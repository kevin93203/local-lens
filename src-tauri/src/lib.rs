use std::{
    collections::{HashMap, HashSet},
    ffi::OsString,
    fs,
    hash::{Hash, Hasher},
    io::BufReader,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{Arc, Condvar, Mutex, OnceLock},
    time::{Instant, SystemTime},
};

use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use fast_image_resize as fir;
use fastembed::{
    EmbeddingModel, ImageEmbedding, ImageEmbeddingModel, ImageInitOptions, InitOptions,
    TextEmbedding,
};
use hf_hub::api::sync::ApiBuilder;
use image::{codecs::jpeg::JpegEncoder, imageops::FilterType, ImageFormat};
use exif::{In, Reader as ExifReader, Tag};
#[cfg(windows)]
use ort::execution_providers::{DirectMLExecutionProvider, ExecutionProvider};
use serde::{Deserialize, Serialize};
use rusqlite::{params, Connection};
use tauri::{Emitter, Manager, State};
use walkdir::WalkDir;

mod face;
use face::{Face, FaceEngine};

const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "webp", "gif", "bmp"];
const DEFAULT_MAX_INDEXED_IMAGES: usize = 3_000;
const DEFAULT_INDEX_BATCH_SIZE: usize = 50;
const MAX_BROWSE_RESULTS: usize = 200;
const MAX_SEARCH_RESULTS: usize = 60;
const MIN_SEMANTIC_SIMILARITY: f32 = 0.20;
const SEMANTIC_BEST_MARGIN: f32 = 0.07;
const FACE_MATCH_THRESHOLD: f32 = 0.45;
const FACE_MODEL_REPOSITORY: &str = "WePrompt/buffalo_sc";

struct CachedTextModel {
    requested_gpu: bool,
    model: TextEmbedding,
}

// Keep the text encoder alive after its first use. Loading the ONNX session
// for every keystroke/search would make subsequent searches unnecessarily slow.
static TEXT_MODEL: OnceLock<Mutex<Option<CachedTextModel>>> = OnceLock::new();
static DIRECTML_STATUS: OnceLock<Result<(), String>> = OnceLock::new();

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
struct ModelSettings {
    max_indexed_images: Option<usize>,
    index_batch_size: usize,
    thumbnail_gpu: bool,
    ocr_gpu: bool,
    clip_gpu: bool,
    face_gpu: bool,
}

impl Default for ModelSettings {
    fn default() -> Self {
        Self {
            max_indexed_images: Some(DEFAULT_MAX_INDEXED_IMAGES),
            index_batch_size: DEFAULT_INDEX_BATCH_SIZE,
            thumbnail_gpu: false,
            ocr_gpu: false,
            clip_gpu: false,
            face_gpu: false,
        }
    }
}

#[derive(Serialize)]
struct SettingsInfo {
    settings: ModelSettings,
    directml_available: bool,
    directml_error: Option<String>,
    thumbnail_gpu_available: bool,
    ocr_gpu_experimental: bool,
}

#[derive(Debug, Clone, Serialize)]
struct ImageRecord {
    id: String,
    path: String,
    filename: String,
    modified_at: String,
    captured_at: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    thumbnail: String,
    // OCR text is kept alongside the image record for local text search.
    ocr_text: String,
    people: Vec<String>,
    score: f32,
    #[serde(skip)]
    embedding: Option<Vec<f32>>,
    #[serde(skip)]
    face_group_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedImageRecord {
    filename: String,
    modified_at: String,
    #[serde(default)]
    captured_at: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    thumbnail: String,
    ocr_text: String,
    people: Vec<String>,
    embedding: Option<Vec<f32>>,
    face_group_ids: Vec<String>,
}

impl CachedImageRecord {
    fn from_record(record: &ImageRecord) -> Self {
        Self {
            filename: record.filename.clone(),
            modified_at: record.modified_at.clone(),
            captured_at: record.captured_at.clone(),
            width: record.width,
            height: record.height,
            thumbnail: record.thumbnail.clone(),
            ocr_text: record.ocr_text.clone(),
            people: record.people.clone(),
            embedding: record.embedding.clone(),
            face_group_ids: record.face_group_ids.clone(),
        }
    }

    fn into_record(self, path: String) -> ImageRecord {
        ImageRecord {
            id: path.clone(),
            path,
            filename: self.filename,
            modified_at: self.modified_at,
            captured_at: self.captured_at,
            width: self.width,
            height: self.height,
            thumbnail: self.thumbnail,
            ocr_text: self.ocr_text,
            people: self.people,
            score: 1.0,
            embedding: self.embedding,
            face_group_ids: self.face_group_ids,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct FileFingerprint {
    bytes: u64,
    modified_ns: u128,
}

#[derive(Debug, Clone)]
struct CachedImage {
    fingerprint: FileFingerprint,
    record: CachedImageRecord,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FaceCluster {
    id: String,
    name: Option<String>,
    centroid: Vec<f32>,
    face_count: usize,
    image_ids: HashSet<String>,
    preview: String,
}

#[derive(Debug, Clone, Serialize)]
struct FaceGroupSummary {
    id: String,
    name: Option<String>,
    face_count: usize,
    image_count: usize,
    preview: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct KnownPerson {
    id: String,
    name: String,
    embedding: Vec<f32>,
}

#[derive(Default)]
struct IndexData {
    images: Vec<ImageRecord>,
    face_groups: Vec<FaceCluster>,
}

#[derive(Clone, Default)]
struct AppIndex {
    data: Arc<Mutex<IndexData>>,
    people_file: Arc<Mutex<Option<PathBuf>>>,
    settings: Arc<Mutex<ModelSettings>>,
    settings_file: Arc<Mutex<Option<PathBuf>>>,
    cache_file: Arc<Mutex<Option<PathBuf>>>,
    scan_control: Arc<ScanControl>,
}

#[derive(Default)]
struct ScanControl {
    paused: Mutex<bool>,
    wake: Condvar,
}

#[derive(Serialize)]
struct ScanResult {
    root: String,
    indexed: usize,
    reused: usize,
    skipped: usize,
    ocr_available: bool,
    semantic_available: bool,
    face_available: bool,
    faces_detected: usize,
    face_groups: usize,
    clip_gpu_active: bool,
    face_gpu_active: bool,
    thumbnail_gpu_requested: bool,
    thumbnail_gpu_active: bool,
    ocr_gpu_requested: bool,
    ocr_gpu_active: bool,
    gpu_warning: Option<String>,
}

#[derive(Clone, Serialize)]
struct ScanProgress {
    processed: usize,
    total: usize,
    eta_seconds: Option<u64>,
    indexed: usize,
    reused: usize,
    skipped: usize,
    ocr_available: bool,
    semantic_available: bool,
    face_available: bool,
    faces_detected: usize,
    clip_gpu_active: bool,
    face_gpu_active: bool,
    thumbnail_gpu_requested: bool,
    thumbnail_gpu_active: bool,
    ocr_gpu_requested: bool,
    ocr_gpu_active: bool,
    gpu_warning: Option<String>,
}

fn is_supported_image(path: &Path) -> bool {
    path.extension()
        .and_then(|value| value.to_str())
        .map(|extension| IMAGE_EXTENSIONS.contains(&extension.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

fn set_scan_paused(control: &ScanControl, paused: bool) -> Result<(), String> {
    let mut state = control.paused.lock().map_err(|_| "掃描控制暫時無法使用。")?;
    *state = paused;
    if !paused {
        control.wake.notify_all();
    }
    Ok(())
}

fn wait_for_scan_resume(control: &ScanControl) -> Result<(), String> {
    let mut paused = control.paused.lock().map_err(|_| "掃描控制暫時無法使用。")?;
    while *paused {
        paused = control.wake.wait(paused).map_err(|_| "掃描控制暫時無法使用。")?;
    }
    Ok(())
}

fn estimate_remaining_seconds(started_at: Instant, processed: usize, total: usize) -> Option<u64> {
    if processed == 0 || total == 0 || processed >= total {
        return (processed >= total && total > 0).then_some(0);
    }
    let elapsed = started_at.elapsed().as_secs_f64();
    if elapsed < 0.5 {
        return None;
    }
    let seconds = ((total - processed) as f64 * elapsed / processed as f64).ceil();
    Some(seconds.max(0.0) as u64)
}

fn thumbnail_dimensions(width: u32, height: u32) -> (u32, u32) {
    const THUMBNAIL_BOUND: u32 = 480;
    if width >= height {
        (
            THUMBNAIL_BOUND,
            ((height as u64 * THUMBNAIL_BOUND as u64 + width as u64 / 2) / width as u64)
                .max(1) as u32,
        )
    } else {
        (
            ((width as u64 * THUMBNAIL_BOUND as u64 + height as u64 / 2) / height as u64)
                .max(1) as u32,
            THUMBNAIL_BOUND,
        )
    }
}

fn make_thumbnail(path: &Path, _use_gpu: bool) -> Result<(String, u32, u32), String> {
    let image = image::open(path).map_err(|error| error.to_string())?;
    let (width, height) = (image.width(), image.height());
    let source = image.to_rgb8();
    let (thumbnail_width, thumbnail_height) = thumbnail_dimensions(width, height);
    let source = fir::images::Image::from_vec_u8(
        width,
        height,
        source.into_raw(),
        fir::PixelType::U8x3,
    )
    .map_err(|error| format!("無法準備縮圖像素：{error}"))?;
    let mut destination = fir::images::Image::new(
        thumbnail_width,
        thumbnail_height,
        fir::PixelType::U8x3,
    );
    fir::Resizer::new()
        .resize(
            &source,
            &mut destination,
            &fir::ResizeOptions::new().resize_alg(fir::ResizeAlg::Convolution(
                fir::FilterType::Bilinear,
            )),
        )
        .map_err(|error| format!("縮放縮圖失敗：{error}"))?;
    let thumbnail = image::RgbImage::from_raw(
        thumbnail_width,
        thumbnail_height,
        destination.into_vec(),
    )
    .ok_or_else(|| "縮圖像素格式不正確。".to_owned())?;
    let mut bytes = Vec::new();
    JpegEncoder::new_with_quality(&mut bytes, 78)
        .encode_image(&thumbnail)
        .map_err(|error| error.to_string())?;
    Ok((
        format!("data:image/jpeg;base64,{}", BASE64.encode(bytes)),
        width,
        height,
    ))
}

fn clip_execution_providers(use_gpu: bool) -> Vec<fastembed::ExecutionProviderDispatch> {
    #[cfg(windows)]
    if use_gpu {
        return vec![DirectMLExecutionProvider::default()
            .build()
            .error_on_failure()];
    }
    let _ = use_gpu;
    Vec::new()
}

fn directml_available() -> bool {
    directml_status().is_ok()
}

fn directml_status() -> Result<(), String> {
    DIRECTML_STATUS
        .get_or_init(|| {
        #[cfg(windows)]
        {
            return DirectMLExecutionProvider::default()
                .is_available()
                .map_err(|error| format!("無法列舉 ONNX Runtime 執行後端：{error}"))
                .and_then(|available| {
                    available.then_some(()).ok_or_else(|| {
                        "目前載入的 ONNX Runtime 沒有 DirectML 執行後端".to_owned()
                    })
                });
        }
        #[allow(unreachable_code)]
        Err("DirectML 僅支援 Windows".to_owned())
        })
        .clone()
}

fn directml_error() -> Option<String> {
    directml_status().err()
}

fn load_image_embedding(use_gpu: bool) -> (Option<ImageEmbedding>, bool, Option<String>) {
    let mut gpu_error = None;
    if use_gpu {
        let options = ImageInitOptions::new(ImageEmbeddingModel::ClipVitB32)
            .with_execution_providers(clip_execution_providers(true))
            .with_show_download_progress(false);
        match ImageEmbedding::try_new(options) {
            Ok(model) => return (Some(model), true, None),
            Err(error) => gpu_error = Some(format!("CLIP GPU 初始化失敗：{error}")),
        }
    }
    let cpu_model = ImageEmbedding::try_new(
        ImageInitOptions::new(ImageEmbeddingModel::ClipVitB32).with_show_download_progress(false),
    )
    .ok();
    (cpu_model, false, gpu_error)
}

fn load_face_engine(use_gpu: bool) -> Result<(FaceEngine, bool, Option<String>), String> {
    let detector_override = std::env::var_os("LOCAL_LENS_FACE_DETECTOR").map(PathBuf::from);
    let recognizer_override = std::env::var_os("LOCAL_LENS_FACE_RECOGNIZER").map(PathBuf::from);
    let (detector_path, recognizer_path) = match (detector_override, recognizer_override) {
        (Some(detector), Some(recognizer)) if detector.is_file() && recognizer.is_file() => {
            (detector, recognizer)
        }
        _ => {
            let cache_root = std::env::var_os("FASTEMBED_CACHE_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(".fastembed_cache"));
            let api = ApiBuilder::from_env()
                .with_cache_dir(cache_root.join("faces"))
                .with_progress(false)
                .build()
                .map_err(|error| format!("無法初始化人臉模型下載器：{error}"))?;
            let repository = api.model(FACE_MODEL_REPOSITORY.to_owned());
            let detector = repository
                .get("det_500m.onnx")
                .map_err(|error| format!("無法取得人臉偵測模型：{error}"))?;
            let recognizer = repository
                .get("w600k_mbf.onnx")
                .map_err(|error| format!("無法取得人臉辨識模型：{error}"))?;
            (detector, recognizer)
        }
    };
    let mut gpu_error = None;
    if use_gpu {
        match FaceEngine::new(&detector_path, &recognizer_path, true) {
            Ok(engine) => return Ok((engine, true, None)),
            Err(error) => gpu_error = Some(format!("Face GPU 初始化失敗：{error}")),
        }
    }
    FaceEngine::new(detector_path, recognizer_path, false)
        .map(|engine| (engine, false, gpu_error))
        .map_err(|error| format!("無法載入人臉模型：{error}"))
}

fn normalize_embedding(values: &[f32]) -> Vec<f32> {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm == 0.0 {
        values.to_vec()
    } else {
        values.iter().map(|value| value / norm).collect()
    }
}

fn make_face_group_id(embedding: &[f32]) -> String {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for value in embedding.iter().take(64) {
        value.to_bits().hash(&mut hasher);
    }
    format!("face-{:016x}", hasher.finish())
}

fn make_face_thumbnail(image: &image::DynamicImage, face: &Face) -> Option<String> {
    let width = image.width() as f32;
    let height = image.height() as f32;
    let face_width = (face.bbox.x2 - face.bbox.x1).max(1.0);
    let face_height = (face.bbox.y2 - face.bbox.y1).max(1.0);
    let margin_x = face_width * 0.22;
    let margin_y = face_height * 0.28;
    let x1 = (face.bbox.x1 - margin_x).clamp(0.0, width - 1.0) as u32;
    let y1 = (face.bbox.y1 - margin_y).clamp(0.0, height - 1.0) as u32;
    let x2 = (face.bbox.x2 + margin_x).clamp(x1 as f32 + 1.0, width) as u32;
    let y2 = (face.bbox.y2 + margin_y).clamp(y1 as f32 + 1.0, height) as u32;
    let crop = image
        .crop_imm(x1, y1, x2.saturating_sub(x1), y2.saturating_sub(y1))
        .resize(180, 180, FilterType::Lanczos3)
        .to_rgb8();
    let mut bytes = Vec::new();
    JpegEncoder::new_with_quality(&mut bytes, 82)
        .encode_image(&crop)
        .ok()?;
    Some(format!("data:image/jpeg;base64,{}", BASE64.encode(bytes)))
}

fn people_file(index: &AppIndex) -> Option<PathBuf> {
    index.people_file.lock().ok()?.clone()
}

fn settings_file(index: &AppIndex) -> Option<PathBuf> {
    index.settings_file.lock().ok()?.clone()
}

fn cache_file(index: &AppIndex) -> Option<PathBuf> {
    index.cache_file.lock().ok()?.clone()
}

fn file_fingerprint(path: &Path) -> Option<FileFingerprint> {
    let metadata = path.metadata().ok()?;
    let modified_ns = metadata
        .modified()
        .ok()?
        .duration_since(SystemTime::UNIX_EPOCH)
        .ok()?
        .as_nanos();
    Some(FileFingerprint {
        bytes: metadata.len(),
        modified_ns,
    })
}

fn open_index_cache(path: &Path) -> Result<Connection, String> {
    let connection = Connection::open(path).map_err(|error| format!("無法開啟 SQLite 索引快取：{error}"))?;
    connection
        .execute_batch(
            "
            PRAGMA journal_mode = WAL;
            CREATE TABLE IF NOT EXISTS image_cache (
                root TEXT NOT NULL,
                path TEXT PRIMARY KEY NOT NULL,
                bytes INTEGER NOT NULL,
                modified_ns TEXT NOT NULL,
                captured_at TEXT,
                record_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS image_cache_root ON image_cache(root);
            CREATE TABLE IF NOT EXISTS scan_state (
                root TEXT PRIMARY KEY NOT NULL,
                face_groups_json TEXT NOT NULL
            );
            ",
        )
        .map_err(|error| format!("無法初始化 SQLite 索引快取：{error}"))?;
    // Existing installations were created before EXIF capture time was added.
    // SQLite has no IF NOT EXISTS form for ADD COLUMN, so an already-present
    // column simply produces an ignored duplicate-column error here.
    let _ = connection.execute("ALTER TABLE image_cache ADD COLUMN captured_at TEXT", []);
    Ok(connection)
}

fn load_cached_images(path: Option<&Path>, root: &str) -> HashMap<String, CachedImage> {
    let Some(path) = path else { return HashMap::new() };
    let Ok(connection) = open_index_cache(path) else { return HashMap::new() };
    let Ok(mut statement) = connection.prepare(
        "SELECT path, bytes, modified_ns, record_json FROM image_cache WHERE root = ?1",
    ) else {
        return HashMap::new();
    };
    let Ok(rows) = statement.query_map(params![root], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, u64>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
        ))
    }) else {
        return HashMap::new();
    };
    rows.filter_map(Result::ok)
        .filter_map(|(path, bytes, modified_ns, record_json)| {
            Some((
                path,
                CachedImage {
                    fingerprint: FileFingerprint {
                        bytes,
                        modified_ns: modified_ns.parse().ok()?,
                    },
                    record: serde_json::from_str(&record_json).ok()?,
                },
            ))
        })
        .collect()
}

fn load_cached_face_groups(path: Option<&Path>, root: &str) -> Vec<FaceCluster> {
    let Some(path) = path else { return Vec::new() };
    let Ok(connection) = open_index_cache(path) else { return Vec::new() };
    connection
        .query_row(
            "SELECT face_groups_json FROM scan_state WHERE root = ?1",
            params![root],
            |row| row.get::<_, String>(0),
        )
        .ok()
        .and_then(|json| serde_json::from_str(&json).ok())
        .unwrap_or_default()
}

fn reset_index_cache(path: Option<&Path>, root: &str) -> Result<(), String> {
    let Some(path) = path else { return Ok(()) };
    let connection = open_index_cache(path)?;
    connection
        .execute("DELETE FROM image_cache WHERE root = ?1", params![root])
        .map_err(|error| error.to_string())?;
    connection
        .execute("DELETE FROM scan_state WHERE root = ?1", params![root])
        .map_err(|error| error.to_string())?;
    Ok(())
}

fn append_index_cache(
    path: Option<&Path>,
    root: &str,
    records: &[ImageRecord],
    fingerprints: &HashMap<String, FileFingerprint>,
) -> Result<(), String> {
    let Some(path) = path else { return Ok(()) };
    let mut connection = open_index_cache(path)?;
    let transaction = connection.transaction().map_err(|error| error.to_string())?;
    {
        let mut insert = transaction
            .prepare("INSERT INTO image_cache(root, path, bytes, modified_ns, captured_at, record_json) VALUES (?1, ?2, ?3, ?4, ?5, ?6)")
            .map_err(|error| error.to_string())?;
        for record in records {
            let Some(fingerprint) = fingerprints.get(&record.path) else { continue };
            let payload = serde_json::to_string(&CachedImageRecord::from_record(record))
                .map_err(|error| error.to_string())?;
            insert
                .execute(params![
                    root,
                    record.path,
                    fingerprint.bytes,
                    fingerprint.modified_ns.to_string(),
                    record.captured_at.as_deref(),
                    payload
                ])
                .map_err(|error| error.to_string())?;
        }
    }
    transaction.commit().map_err(|error| error.to_string())
}

fn save_index_cache_state(
    path: Option<&Path>,
    root: &str,
    face_groups: &[FaceCluster],
) -> Result<(), String> {
    let Some(path) = path else { return Ok(()) };
    let connection = open_index_cache(path)?;
    let face_groups_json = serde_json::to_string(face_groups).map_err(|error| error.to_string())?;
    connection
        .execute(
            "INSERT INTO scan_state(root, face_groups_json) VALUES (?1, ?2) \
             ON CONFLICT(root) DO UPDATE SET face_groups_json = excluded.face_groups_json",
            params![root, face_groups_json],
        )
        .map_err(|error| error.to_string())?;
    Ok(())
}

fn refresh_face_cluster_counts(face_clusters: &mut Vec<FaceCluster>, records: &[ImageRecord]) {
    for cluster in face_clusters.iter_mut() {
        cluster.face_count = 0;
        cluster.image_ids.clear();
    }
    for record in records {
        for group_id in &record.face_group_ids {
            if let Some(cluster) = face_clusters.iter_mut().find(|cluster| cluster.id == *group_id) {
                cluster.face_count += 1;
                cluster.image_ids.insert(record.id.clone());
            }
        }
    }
    face_clusters.retain(|cluster| cluster.face_count > 0);
}

fn current_settings(index: &AppIndex) -> ModelSettings {
    index
        .settings
        .lock()
        .map(|value| value.clone())
        .unwrap_or_default()
}

fn load_settings(path: &Path) -> ModelSettings {
    fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default()
}

fn save_settings(path: &Path, settings: &ModelSettings) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let bytes = serde_json::to_vec_pretty(settings).map_err(|error| error.to_string())?;
    fs::write(path, bytes).map_err(|error| error.to_string())
}

fn load_known_people(index: &AppIndex) -> Vec<KnownPerson> {
    let Some(path) = people_file(index) else {
        return Vec::new();
    };
    fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default()
}

fn save_known_people(path: &Path, people: &[KnownPerson]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let bytes = serde_json::to_vec_pretty(people).map_err(|error| error.to_string())?;
    fs::write(path, bytes).map_err(|error| error.to_string())
}

fn update_face_centroid(cluster: &mut FaceCluster, embedding: &[f32]) {
    let count = cluster.face_count as f32;
    for (centroid, value) in cluster.centroid.iter_mut().zip(embedding.iter()) {
        *centroid = (*centroid * count + value) / (count + 1.0);
    }
    cluster.centroid = normalize_embedding(&cluster.centroid);
    cluster.face_count += 1;
}

fn assign_face_cluster(
    clusters: &mut Vec<FaceCluster>,
    known_people: &[KnownPerson],
    image_id: &str,
    preview: String,
    raw_embedding: &[f32],
) -> (String, Option<String>) {
    let embedding = normalize_embedding(raw_embedding);
    let known_match = known_people
        .iter()
        .filter_map(|person| {
            let score = cosine_similarity(&embedding, &person.embedding);
            (score >= FACE_MATCH_THRESHOLD).then_some((person, score))
        })
        .max_by(|left, right| left.1.total_cmp(&right.1));

    let target_index = if let Some((person, _)) = known_match {
        clusters.iter().position(|cluster| cluster.id == person.id)
    } else {
        clusters
            .iter()
            .enumerate()
            .filter_map(|(index, cluster)| {
                let score = cosine_similarity(&embedding, &cluster.centroid);
                (score >= FACE_MATCH_THRESHOLD).then_some((index, score))
            })
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .map(|(index, _)| index)
    };

    let index = if let Some(index) = target_index {
        index
    } else {
        let (id, name) = known_match
            .map(|(person, _)| (person.id.clone(), Some(person.name.clone())))
            .unwrap_or_else(|| (make_face_group_id(&embedding), None));
        clusters.push(FaceCluster {
            id,
            name,
            centroid: embedding.clone(),
            face_count: 0,
            image_ids: HashSet::new(),
            preview,
        });
        clusters.len() - 1
    };

    let cluster = &mut clusters[index];
    update_face_centroid(cluster, &embedding);
    cluster.image_ids.insert(image_id.to_owned());
    (cluster.id.clone(), cluster.name.clone())
}

fn face_group_summaries(data: &IndexData) -> Vec<FaceGroupSummary> {
    let mut groups: Vec<FaceGroupSummary> = data
        .face_groups
        .iter()
        .map(|cluster| FaceGroupSummary {
            id: cluster.id.clone(),
            name: cluster.name.clone(),
            face_count: cluster.face_count,
            image_count: cluster.image_ids.len(),
            preview: cluster.preview.clone(),
        })
        .collect();
    groups.sort_by(|left, right| {
        right
            .name
            .is_some()
            .cmp(&left.name.is_some())
            .then_with(|| right.face_count.cmp(&left.face_count))
    });
    groups
}

fn format_modified(time: SystemTime) -> String {
    time.duration_since(SystemTime::UNIX_EPOCH)
        .map(|duration| duration.as_secs().to_string())
        .unwrap_or_default()
}

fn normalize_exif_datetime(value: &str) -> Option<String> {
    let value = value
        .trim()
        .trim_matches('\0')
        .trim_matches('"')
        .trim();
    let parts: Vec<u32> = value
        .split(|character: char| character == ':' || character == '-' || character == ' ' || character == 'T')
        .filter(|part| !part.is_empty())
        .take(6)
        .map(str::parse)
        .collect::<Result<_, _>>()
        .ok()?;
    if parts.len() != 6
        || !(1900..=2200).contains(&parts[0])
        || !(1..=12).contains(&parts[1])
        || !(1..=31).contains(&parts[2])
        || parts[3] > 23
        || parts[4] > 59
        || parts[5] > 60
    {
        return None;
    }
    Some(format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
        parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
    ))
}

fn extract_exif_capture_time(path: &Path) -> Option<String> {
    let file = fs::File::open(path).ok()?;
    let mut reader = BufReader::new(file);
    let metadata = ExifReader::new().read_from_container(&mut reader).ok()?;
    [Tag::DateTimeOriginal, Tag::DateTimeDigitized, Tag::DateTime]
        .into_iter()
        .find_map(|tag| {
            metadata
                .get_field(tag, In::PRIMARY)
                .and_then(|field| normalize_exif_datetime(&field.display_value().to_string()))
        })
}

fn tesseract_program() -> OsString {
    std::env::var_os("LOCAL_LENS_TESSERACT").unwrap_or_else(|| OsString::from("tesseract"))
}

fn tesseract_language() -> OsString {
    std::env::var_os("LOCAL_LENS_TESSERACT_LANG").unwrap_or_else(|| OsString::from("eng+chi_tra"))
}

fn has_tesseract(program: &OsString) -> bool {
    Command::new(program)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn tesseract_opencl_available(program: &OsString) -> bool {
    Command::new(program)
        .arg("--version")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| {
            let details = format!(
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            details.to_ascii_lowercase().contains("opencl")
        })
        .unwrap_or(false)
}

struct OcrCandidate {
    text: String,
    confidence: f32,
}

fn make_ocr_image(path: &Path) -> Option<PathBuf> {
    let source = image::open(path).ok()?;
    let largest_side = source.width().max(source.height());
    let target_side = if largest_side < 1800 {
        (largest_side.saturating_mul(2)).min(2600)
    } else {
        largest_side.min(3200)
    };
    let resized = if target_side != largest_side {
        source.resize(target_side, target_side, FilterType::Lanczos3)
    } else {
        source
    };
    let enhanced = resized.grayscale().adjust_contrast(18.0);
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    path.to_string_lossy().hash(&mut hasher);
    let output = std::env::temp_dir().join(format!(
        "local-lens-ocr-{}-{:x}.png",
        std::process::id(),
        hasher.finish()
    ));
    enhanced.save_with_format(&output, ImageFormat::Png).ok()?;
    Some(output)
}

fn run_tesseract(
    path: &Path,
    program: &OsString,
    language: &OsString,
    page_segmentation_mode: u8,
    use_gpu: bool,
) -> Option<OcrCandidate> {
    let mut command = Command::new(program);
    command
        .arg(path)
        .arg("stdout")
        .arg("--oem")
        .arg("1")
        .arg("--psm")
        .arg(page_segmentation_mode.to_string())
        .arg("-l")
        .arg(language)
        .arg("-c")
        .arg("user_defined_dpi=300")
        .arg("-c")
        .arg("preserve_interword_spaces=1")
        .arg("tsv");
    if use_gpu {
        // This is honored only by a Tesseract build compiled with its
        // experimental OpenCL backend. Standard builds safely ignore it.
        command.env("TESSERACT_OPENCL_DEVICE", "1");
    }
    let output = command.output().ok()?;
    if !output.status.success() {
        return None;
    }

    let mut words = Vec::new();
    let mut confidence_total = 0.0;
    let mut confidence_count = 0;
    for line in String::from_utf8_lossy(&output.stdout).lines().skip(1) {
        let fields: Vec<&str> = line.splitn(12, '\t').collect();
        if fields.len() < 12 {
            continue;
        }
        let word = fields[11].trim();
        if word.is_empty() {
            continue;
        }
        if let Ok(confidence) = fields[10].parse::<f32>() {
            if confidence >= 0.0 {
                confidence_total += confidence;
                confidence_count += 1;
            }
        }
        words.push(word.to_owned());
    }
    if words.is_empty() {
        return None;
    }
    let confidence = if confidence_count == 0 {
        0.0
    } else {
        confidence_total / confidence_count as f32
    };
    // Low-confidence OCR is usually visual noise. Do not put it into the
    // search index where it would create false-positive matches.
    if confidence_count > 0 && confidence < 20.0 {
        return None;
    }
    Some(OcrCandidate {
        text: words.join(" "),
        confidence,
    })
}

fn extract_ocr_text(path: &Path, program: &OsString, use_gpu: bool) -> String {
    let enhanced_path = make_ocr_image(path);
    let enhanced = enhanced_path.as_deref().unwrap_or(path);
    let configured_language = tesseract_language();
    let english = OsString::from("eng");
    let mut candidates = Vec::new();
    for (input, mode) in [(enhanced, 6_u8), (path, 11_u8)] {
        if let Some(candidate) = run_tesseract(input, program, &configured_language, mode, use_gpu)
        {
            candidates.push(candidate);
        } else if configured_language != english {
            if let Some(candidate) = run_tesseract(input, program, &english, mode, use_gpu) {
                candidates.push(candidate);
            }
        }
    }
    if let Some(temp_path) = enhanced_path {
        let _ = fs::remove_file(temp_path);
    }
    candidates
        .into_iter()
        .max_by(|left, right| {
            let left_score = left.confidence + (left.text.chars().count().min(200) as f32 * 0.08);
            let right_score =
                right.confidence + (right.text.chars().count().min(200) as f32 * 0.08);
            left_score.total_cmp(&right_score)
        })
        .map(|candidate| candidate.text)
        .unwrap_or_default()
}

fn build_index(
    folder: String,
    app: tauri::AppHandle,
    index: AppIndex,
) -> Result<ScanResult, String> {
    let root = PathBuf::from(&folder);
    if !root.is_dir() {
        return Err("選擇的位置不是資料夾。".into());
    }

    let candidates: Vec<PathBuf> = WalkDir::new(&root)
        .follow_links(false)
        .into_iter()
        .filter_map(Result::ok)
        .map(|entry| entry.into_path())
        .filter(|path| path.is_file() && is_supported_image(path))
        .collect();
    let candidates = if let Some(limit) = current_settings(&index).max_indexed_images {
        candidates.into_iter().take(limit).collect()
    } else {
        candidates
    };
    let total = candidates.len();
    let cache_path = cache_file(&index);
    let cached_images = load_cached_images(cache_path.as_deref(), &folder);
    let fingerprints: HashMap<String, FileFingerprint> = candidates
        .iter()
        .filter_map(|path| {
            let path_text = path.to_string_lossy().into_owned();
            file_fingerprint(path).map(|fingerprint| (path_text, fingerprint))
        })
        .collect();
    let needs_processing = candidates.iter().any(|path| {
        let path_text = path.to_string_lossy();
        !matches!(
            (fingerprints.get(path_text.as_ref()), cached_images.get(path_text.as_ref())),
            (Some(current), Some(cached))
                if current.bytes == cached.fingerprint.bytes
                    && current.modified_ns == cached.fingerprint.modified_ns
        )
    });
    let mut records = Vec::new();
    let mut reused = 0;
    let mut face_clusters = load_cached_face_groups(cache_path.as_deref(), &folder);
    let known_people = load_known_people(&index);
    let mut faces_detected = 0;
    let mut skipped = 0;
    let settings = current_settings(&index);
    let _ = reset_index_cache(cache_path.as_deref(), &folder);
    let mut cache_written = 0usize;
    let batch_size = settings.index_batch_size.max(1);
    let ocr_program = tesseract_program();
    let ocr_available = has_tesseract(&ocr_program);
    let ocr_gpu_active = settings.ocr_gpu && ocr_available && tesseract_opencl_available(&ocr_program);
    // The first initialization may download the local ONNX models into the
    // FastEmbed cache. It runs inside the blocking worker, never on the UI loop.
    let gpu_available = directml_available();
    let mut gpu_warnings = Vec::new();
    if settings.thumbnail_gpu {
        gpu_warnings.push("縮圖 GPU：目前影像解碼、縮放與 JPEG 編碼仍由 CPU 執行；此選項尚無可用的 GPU 後端".to_owned());
    }
    if settings.ocr_gpu && !ocr_gpu_active {
        gpu_warnings.push("OCR GPU：目前 Tesseract 未回報 OpenCL 支援，已使用 CPU".to_owned());
    }
    if !gpu_available {
        let reason = directml_error().unwrap_or_else(|| "找不到可用的 DirectML".to_owned());
        if settings.clip_gpu {
            gpu_warnings.push(format!("CLIP GPU：{reason}，已使用 CPU"));
        }
        if settings.face_gpu {
            gpu_warnings.push(format!("Face GPU：{reason}，已使用 CPU"));
        }
    }
    let (mut image_model, clip_gpu_active, clip_gpu_error) = if needs_processing {
        load_image_embedding(settings.clip_gpu && gpu_available)
    } else {
        (None, false, None)
    };
    if let Some(error) = clip_gpu_error {
        gpu_warnings.push(error);
    }
    let mut semantic_available = image_model.is_some()
        || cached_images.values().any(|cached| cached.record.embedding.is_some());
    let (mut face_engine, face_gpu_active, face_gpu_error) = if needs_processing {
        match load_face_engine(settings.face_gpu && gpu_available) {
            Ok((engine, active, error)) => (Some(engine), active, error),
            Err(error) => (None, false, Some(error)),
        }
    } else {
        (None, false, None)
    };
    if let Some(error) = face_gpu_error {
        gpu_warnings.push(error);
    }
    let gpu_warning = (!gpu_warnings.is_empty()).then(|| gpu_warnings.join("；"));
    let face_available = face_engine.is_some() || !face_clusters.is_empty();
    let scan_started_at = Instant::now();
    app.emit(
        "scan-progress",
        ScanProgress {
            processed: 0,
            total,
            eta_seconds: estimate_remaining_seconds(scan_started_at, 0, total),
            indexed: 0,
            reused: 0,
            skipped: 0,
            ocr_available,
            semantic_available,
            face_available,
            faces_detected,
            clip_gpu_active,
            face_gpu_active,
            thumbnail_gpu_requested: settings.thumbnail_gpu,
            thumbnail_gpu_active: false,
            ocr_gpu_requested: settings.ocr_gpu,
            ocr_gpu_active,
            gpu_warning: gpu_warning.clone(),
        },
    )
    .map_err(|error| error.to_string())?;
    for (position, path) in candidates.iter().enumerate() {
        wait_for_scan_resume(&index.scan_control)?;
        let path_text = path.to_string_lossy().into_owned();
        if let (Some(current), Some(cached)) = (fingerprints.get(&path_text), cached_images.get(&path_text)) {
            if current.bytes == cached.fingerprint.bytes && current.modified_ns == cached.fingerprint.modified_ns {
                faces_detected += cached.record.face_group_ids.len();
                records.push(cached.record.clone().into_record(path_text));
                reused += 1;
                if records.len().saturating_sub(cache_written) >= batch_size {
                    if append_index_cache(
                        cache_path.as_deref(),
                        &folder,
                        &records[cache_written..],
                        &fingerprints,
                    )
                    .is_ok()
                    {
                        cache_written = records.len();
                    }
                }
                let processed = position + 1;
                if processed == total || processed % 5 == 0 {
                    app.emit(
                        "scan-progress",
                        ScanProgress {
                            processed,
                            total,
                            eta_seconds: estimate_remaining_seconds(scan_started_at, processed, total),
                            indexed: records.len(),
                            reused,
                            skipped,
                            ocr_available,
                            semantic_available: semantic_available || records.iter().any(|record| record.embedding.is_some()),
                            face_available,
                            faces_detected,
                            clip_gpu_active,
                            face_gpu_active,
                            thumbnail_gpu_requested: settings.thumbnail_gpu,
                            thumbnail_gpu_active: false,
                            ocr_gpu_requested: settings.ocr_gpu,
                            ocr_gpu_active,
                            gpu_warning: gpu_warning.clone(),
                        },
                    )
                    .map_err(|error| error.to_string())?;
                }
                continue;
            }
        }
        match make_thumbnail(path, settings.thumbnail_gpu) {
            Ok((thumbnail, width, height)) => {
                let embedding = if let Some(model) = image_model.as_mut() {
                    match model.embed(vec![path], None) {
                        Ok(mut embeddings) => embeddings.pop(),
                        Err(_) => {
                            // A single unreadable/unsupported image should not abort the
                            // complete scan. Disable semantic indexing for the remainder and
                            // keep the filename/OCR index usable.
                            image_model = None;
                            semantic_available = false;
                            None
                        }
                    }
                } else {
                    None
                };
                let mut face_group_ids = Vec::new();
                let mut people = Vec::new();
                if let (Some(engine), Ok(source)) = (face_engine.as_mut(), image::open(path)) {
                    let rgb = source.to_rgb8();
                    if let Ok(faces) = engine.run(&rgb) {
                        for face in faces {
                            let face_width = face.bbox.x2 - face.bbox.x1;
                            let face_height = face.bbox.y2 - face.bbox.y1;
                            if face_width < 24.0 || face_height < 24.0 {
                                continue;
                            }
                            let Some(face_preview) = make_face_thumbnail(&source, &face) else {
                                continue;
                            };
                            let (group_id, person_name) = assign_face_cluster(
                                &mut face_clusters,
                                &known_people,
                                &path_text,
                                face_preview,
                                &face.embedding,
                            );
                            faces_detected += 1;
                            if !face_group_ids.contains(&group_id) {
                                face_group_ids.push(group_id);
                            }
                            if let Some(name) = person_name {
                                if !people.contains(&name) {
                                    people.push(name);
                                }
                            }
                        }
                    }
                }
                records.push(ImageRecord {
                    id: path_text.clone(),
                    path: path_text,
                    filename: path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("圖片")
                        .to_owned(),
                    modified_at: path
                        .metadata()
                        .and_then(|metadata| metadata.modified())
                        .map(format_modified)
                        .unwrap_or_default(),
                    captured_at: extract_exif_capture_time(path),
                    width: Some(width),
                    height: Some(height),
                    thumbnail,
                    ocr_text: if ocr_available {
                        extract_ocr_text(path, &ocr_program, settings.ocr_gpu)
                    } else {
                        String::new()
                    },
                    people,
                    score: 1.0,
                    embedding,
                    face_group_ids,
                });
            }
            Err(_) => skipped += 1,
        }
        if records.len().saturating_sub(cache_written) >= batch_size {
            if append_index_cache(
                cache_path.as_deref(),
                &folder,
                &records[cache_written..],
                &fingerprints,
            )
            .is_ok()
            {
                cache_written = records.len();
            }
        }
        let processed = position + 1;
        // Emit in small batches rather than once per file; large folders remain responsive.
        if processed == total || processed % 5 == 0 {
            app.emit(
                "scan-progress",
                ScanProgress {
                    processed,
                    total,
                    eta_seconds: estimate_remaining_seconds(scan_started_at, processed, total),
                    indexed: records.len(),
                    reused,
                    skipped,
                    ocr_available,
                    semantic_available,
                    face_available,
                    faces_detected,
                    clip_gpu_active,
                    face_gpu_active,
                    thumbnail_gpu_requested: settings.thumbnail_gpu,
                    thumbnail_gpu_active: false,
                    ocr_gpu_requested: settings.ocr_gpu,
                    ocr_gpu_active,
                    gpu_warning: gpu_warning.clone(),
                },
            )
            .map_err(|error| error.to_string())?;
        }
    }
    records.sort_by(|a, b| a.filename.to_lowercase().cmp(&b.filename.to_lowercase()));
    refresh_face_cluster_counts(&mut face_clusters, &records);
    // A cache write failure must not discard the freshly built in-memory index.
    let _ = append_index_cache(
        cache_path.as_deref(),
        &folder,
        &records[cache_written..],
        &fingerprints,
    );
    let _ = save_index_cache_state(cache_path.as_deref(), &folder, &face_clusters);
    let indexed = records.len();
    let face_groups = face_clusters.len();
    *index.data.lock().map_err(|_| "索引暫時無法使用。")? = IndexData {
        images: records,
        face_groups: face_clusters,
    };
    Ok(ScanResult {
        root: folder,
        indexed,
        reused,
        skipped,
        ocr_available,
        semantic_available,
        face_available,
        faces_detected,
        face_groups,
        clip_gpu_active,
        face_gpu_active,
        thumbnail_gpu_requested: settings.thumbnail_gpu,
        thumbnail_gpu_active: false,
        ocr_gpu_requested: settings.ocr_gpu,
        ocr_gpu_active,
        gpu_warning,
    })
}

#[tauri::command]
async fn scan_folder(
    folder: String,
    app: tauri::AppHandle,
    index: State<'_, AppIndex>,
) -> Result<ScanResult, String> {
    set_scan_paused(&index.scan_control, false)?;
    let index = index.inner().clone();
    // Decoding images and encoding thumbnails are CPU / disk heavy.  Use a
    // blocking worker so the WebView and native window keep processing events.
    tauri::async_runtime::spawn_blocking(move || build_index(folder, app, index))
        .await
        .map_err(|error| format!("索引工作意外中止：{error}"))?
}

#[tauri::command]
fn pause_scan(index: State<'_, AppIndex>) -> Result<(), String> {
    set_scan_paused(&index.scan_control, true)
}

#[tauri::command]
fn resume_scan(index: State<'_, AppIndex>) -> Result<(), String> {
    set_scan_paused(&index.scan_control, false)
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    if left.len() != right.len() || left.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut left_norm = 0.0;
    let mut right_norm = 0.0;
    for (left_value, right_value) in left.iter().zip(right.iter()) {
        dot += left_value * right_value;
        left_norm += left_value * left_value;
        right_norm += right_value * right_value;
    }
    if left_norm == 0.0 || right_norm == 0.0 {
        0.0
    } else {
        dot / (left_norm.sqrt() * right_norm.sqrt())
    }
}

fn embed_queries(query: &str, use_gpu: bool) -> Vec<Vec<f32>> {
    let model_lock = TEXT_MODEL.get_or_init(|| Mutex::new(None));
    let Ok(mut model) = model_lock.lock() else {
        return Vec::new();
    };
    if model
        .as_ref()
        .is_some_and(|cached| cached.requested_gpu != use_gpu)
    {
        *model = None;
    }
    if model.is_none() {
        let gpu_model = (use_gpu && directml_available())
            .then(|| {
                TextEmbedding::try_new(
                    InitOptions::new(EmbeddingModel::ClipVitB32)
                        .with_execution_providers(clip_execution_providers(true))
                        .with_show_download_progress(false),
                )
                .ok()
            })
            .flatten();
        let loaded = gpu_model.or_else(|| {
            TextEmbedding::try_new(
                InitOptions::new(EmbeddingModel::ClipVitB32).with_show_download_progress(false),
            )
            .ok()
        });
        *model = loaded.map(|model| CachedTextModel {
            requested_gpu: use_gpu,
            model,
        });
    }
    let Some(cached) = model.as_mut() else {
        return Vec::new();
    };
    // CLIP was trained with short image captions. Comparing both the user's
    // original wording and a photo-oriented prompt is more stable for natural
    // language queries than relying on only one phrasing.
    cached
        .model
        .embed(
            vec![query.to_owned(), format!("a photo of {query}")],
            Some(2),
        )
        .unwrap_or_default()
}

fn semantic_threshold(best_score: Option<f32>) -> f32 {
    best_score
        .map(|score| (score - SEMANTIC_BEST_MARGIN).max(MIN_SEMANTIC_SIMILARITY))
        .unwrap_or(MIN_SEMANTIC_SIMILARITY)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FacePresence {
    Any,
    Required,
    Forbidden,
}

impl Default for FacePresence {
    fn default() -> Self {
        Self::Any
    }
}

#[derive(Debug, Clone, Default)]
struct DateRange {
    from: Option<String>,
    to: Option<String>,
}

#[derive(Debug, Clone, Default)]
struct TextQueryPlan {
    should: Vec<String>,
    must: Vec<String>,
    must_not: Vec<String>,
}

#[derive(Debug, Clone, Default)]
struct QueryPlan {
    semantic_query: String,
    text: TextQueryPlan,
    people_include: Vec<String>,
    people_exclude: Vec<String>,
    face_presence: FacePresence,
    date: DateRange,
    extensions: Vec<String>,
    limit: Option<usize>,
}

const QUERY_STOP_WORDS: &[&str] = &[
    "的", "和", "與", "在", "有", "一張", "一些", "照片", "相片", "圖片", "影像", "找", "找出",
    "請", "幫我", "顯示", "列出", "相關", "最", "看", "只", "其中", "裡", "中", "張",
];

fn date_bounds(year: i32, month: Option<u32>, day: Option<u32>) -> Option<DateRange> {
    if !(1900..=2200).contains(&year) {
        return None;
    }
    let month = month.unwrap_or(1);
    if !(1..=12).contains(&month) {
        return None;
    }
    let max_day = days_in_month(year, month);
    let first_day = day.unwrap_or(1);
    if first_day == 0 || first_day > max_day {
        return None;
    }
    let last_day = day.unwrap_or(max_day);
    Some(DateRange {
        from: Some(format!("{year:04}-{month:02}-{first_day:02} 00:00:00")),
        to: Some(format!("{year:04}-{month:02}-{last_day:02} 23:59:59")),
    })
}

fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        2 if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

fn digits_before(value: &str, end: usize) -> Option<i32> {
    let prefix = &value[..end];
    let start = prefix
        .char_indices()
        .rev()
        .find(|(_, character)| !character.is_ascii_digit())
        .map(|(index, character)| index + character.len_utf8())
        .unwrap_or(0);
    prefix[start..].parse().ok()
}

fn parse_date_range(query: &str) -> DateRange {
    let now_year = time::OffsetDateTime::now_utc().year();
    if query.contains("去年夏天") {
        return DateRange {
            from: Some(format!("{:04}-06-01 00:00:00", now_year - 1)),
            to: Some(format!("{:04}-08-31 23:59:59", now_year - 1)),
        };
    }
    if query.contains("今年夏天") {
        return DateRange {
            from: Some(format!("{now_year:04}-06-01 00:00:00")),
            to: Some(format!("{now_year:04}-08-31 23:59:59")),
        };
    }
    for (phrase, year_offset) in [("前年", -2), ("去年", -1), ("今年", 0), ("明年", 1)] {
        if query.contains(phrase) {
            return date_bounds(now_year + year_offset, None, None).unwrap_or_default();
        }
    }
    let year_index = query.find('年');
    let Some(year_index) = year_index else {
        return DateRange::default();
    };
    let Some(year) = digits_before(query, year_index) else {
        return DateRange::default();
    };
    let month = query
        .find('月')
        .and_then(|index| digits_before(query, index))
        .map(|month| month as u32);
    let day = query
        .find('日')
        .and_then(|index| digits_before(query, index))
        .map(|day| day as u32);
    date_bounds(year, month, day).unwrap_or_default()
}

fn remove_date_words(value: &str) -> String {
    let mut result = value.to_owned();
    for phrase in [
        "去年夏天", "今年夏天", "前年", "去年", "今年", "明年", "夏天", "年", "月", "日",
    ] {
        result = result.replace(phrase, " ");
    }
    if result.chars().any(|character| character.is_ascii_digit()) {
        result = result
            .chars()
            .map(|character| if character.is_ascii_digit() { ' ' } else { character })
            .collect();
    }
    result
}

fn parse_limit(query: &str) -> Option<usize> {
    for marker in ["最多", "前", "取"] {
        let Some(index) = query.find(marker) else { continue };
        let suffix = query[index + marker.len()..].trim_start();
        let digits: String = suffix
            .chars()
            .take_while(|character| character.is_ascii_digit())
            .collect();
        if let Ok(limit) = digits.parse::<usize>() {
            if limit > 0 {
                return Some(limit.min(MAX_SEARCH_RESULTS));
            }
        }
    }
    None
}

fn tokenize_query(value: &str) -> Vec<String> {
    value
        .split(|character: char| {
            character.is_whitespace()
                || matches!(character, '、' | '，' | ',' | '。' | '.' | '！' | '!' | '?' | '？')
        })
        .flat_map(|term| term.split(|character: char| matches!(character, '的' | '和' | '與' | '在' | '有')))
        .map(str::trim)
        .filter(|term| !term.is_empty())
        .map(str::to_owned)
        .filter(|term| !QUERY_STOP_WORDS.contains(&term.as_str()))
        .collect()
}

fn extract_negative_terms(value: &mut String) -> Vec<String> {
    let mut negative_terms = Vec::new();
    for marker in ["不要", "不含", "排除", "沒有"] {
        let mut search_from = 0;
        while let Some(relative_index) = value[search_from..].find(marker) {
            let index = search_from + relative_index;
            let start = index + marker.len();
            let tail = &value[start..];
            let candidate: String = tail
                .chars()
                .take_while(|character| {
                    !character.is_whitespace()
                        && !matches!(
                            character,
                            '的' | '和' | '與' | '在' | '有' | '或' | '、' | '，' | ',' | '。'
                        )
                })
                .collect();
            if !candidate.is_empty() && !QUERY_STOP_WORDS.contains(&candidate.as_str()) {
                negative_terms.push(candidate.clone());
                value.replace_range(index..start + candidate.len(), " ");
                search_from = index;
            } else {
                value.replace_range(index..start, " ");
                search_from = index;
            }
        }
    }
    negative_terms
}

fn parse_query(query: &str, known_people: &[String]) -> QueryPlan {
    let normalized = query.trim().to_lowercase();
    let mut residual = normalized.clone();
    let date = parse_date_range(&normalized);
    residual = remove_date_words(&residual);
    let limit = parse_limit(&normalized);
    for marker in ["最多", "前", "取"] {
        if let Some(index) = residual.find(marker) {
            let end = index + marker.len();
            let whitespace = residual[end..]
                .chars()
                .take_while(|character| character.is_whitespace())
                .map(char::len_utf8)
                .sum::<usize>();
            let digits = residual[end + whitespace..]
                .chars()
                .take_while(|character| character.is_ascii_digit())
                .count();
            residual.replace_range(index..end + whitespace + digits, " ");
        }
    }

    let mut face_presence = FacePresence::Any;
    for phrase in ["不要有人", "沒有人的", "沒有臉", "無人臉", "無人物", "不要人"] {
        if normalized.contains(phrase) {
            face_presence = FacePresence::Forbidden;
            residual = residual.replace(phrase, " ");
        }
    }
    for phrase in ["有人臉", "有人", "人物", "人像", "合照"] {
        if face_presence == FacePresence::Any && normalized.contains(phrase) {
            face_presence = FacePresence::Required;
            residual = residual.replace(phrase, " ");
        }
    }

    let mut people_include = Vec::new();
    let mut people_exclude = Vec::new();
    for person in known_people {
        let person = person.trim();
        if person.is_empty() || !normalized.contains(&person.to_lowercase()) {
            continue;
        }
        let person_lower = person.to_lowercase();
        let negative = ["不要", "不含", "排除", "沒有", "無"]
            .iter()
            .any(|marker| normalized.contains(&format!("{marker}{person_lower}")));
        if negative {
            people_exclude.push(person.to_owned());
        } else {
            people_include.push(person.to_owned());
            if face_presence == FacePresence::Any {
                face_presence = FacePresence::Required;
            }
        }
        residual = residual.replace(&person_lower, " ");
    }

    let mut extensions = Vec::new();
    for extension in IMAGE_EXTENSIONS {
        if normalized.contains(extension) {
            extensions.push((*extension).to_owned());
            residual = residual.replace(extension, " ");
        }
    }
    let mut text = TextQueryPlan::default();
    text.must_not = extract_negative_terms(&mut residual);
    let mut positive_terms = Vec::new();
    for term in tokenize_query(&residual) {
        if ["不要", "不含", "排除", "沒有", "無"].contains(&term.as_str()) {
            continue;
        }
        positive_terms.push(term);
    }
    text.should = positive_terms.clone();
    let semantic_query = positive_terms.join(" ");
    QueryPlan {
        semantic_query,
        text,
        people_include,
        people_exclude,
        face_presence,
        date,
        extensions,
        limit,
    }
}

fn record_date_key(record: &ImageRecord) -> Option<String> {
    if let Some(captured_at) = &record.captured_at {
        return Some(captured_at.clone());
    }
    let timestamp = record.modified_at.parse::<i64>().ok()?;
    let date = time::OffsetDateTime::from_unix_timestamp(timestamp).ok()?;
    Some(format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
        date.year(),
        date.month() as u8,
        date.day(),
        date.hour(),
        date.minute(),
        date.second()
    ))
}

fn record_matches_plan(record: &ImageRecord, plan: &QueryPlan) -> bool {
    if !plan.extensions.is_empty() {
        let extension = Path::new(&record.path)
            .extension()
            .and_then(|extension| extension.to_str())
            .map(str::to_lowercase);
        if extension.is_none_or(|extension| !plan.extensions.contains(&extension)) {
            return false;
        }
    }
    if plan.face_presence == FacePresence::Required && record.face_group_ids.is_empty() {
        return false;
    }
    if plan.face_presence == FacePresence::Forbidden && !record.face_group_ids.is_empty() {
        return false;
    }
    let people = record.people.iter().map(|person| person.to_lowercase()).collect::<Vec<_>>();
    if plan.people_include.iter().any(|person| !people.contains(&person.to_lowercase())) {
        return false;
    }
    if plan.people_exclude.iter().any(|person| people.contains(&person.to_lowercase())) {
        return false;
    }
    if plan.date.from.is_some() || plan.date.to.is_some() {
        let Some(date) = record_date_key(record) else { return false };
        if plan.date.from.as_ref().is_some_and(|from| date < *from) {
            return false;
        }
        if plan.date.to.as_ref().is_some_and(|to| date > *to) {
            return false;
        }
    }
    true
}

fn search_images_sync(query: String, index: AppIndex) -> Result<Vec<ImageRecord>, String> {
    let settings = current_settings(&index);
    let data = index.data.lock().map_err(|_| "索引暫時無法使用。")?;
    let records = &data.images;
    if query.trim().is_empty() {
        return Ok(records.iter().take(MAX_BROWSE_RESULTS).cloned().collect());
    }
    let mut known_people: Vec<String> = records
        .iter()
        .flat_map(|record| record.people.iter().cloned())
        .collect();
    known_people.sort_unstable();
    known_people.dedup();
    let plan = parse_query(&query, &known_people);
    let query_embeddings = if !plan.semantic_query.is_empty()
        && records.iter().any(|record| record.embedding.is_some())
    {
        embed_queries(&plan.semantic_query, settings.clip_gpu)
    } else {
        Vec::new()
    };
    let semantic_search = !query_embeddings.is_empty();
    let has_positive_query = !plan.semantic_query.is_empty()
        || !plan.text.should.is_empty()
        || !plan.text.must.is_empty();

    struct SearchCandidate {
        record: ImageRecord,
        lexical_score: f32,
        lexical_allowed: bool,
        semantic_score: Option<f32>,
    }

    let candidates: Vec<SearchCandidate> = records
        .iter()
        .filter(|record| record_matches_plan(record, &plan))
        .map(|record| {
            let haystack = format!(
                "{} {} {}",
                record.filename,
                record.ocr_text,
                record.people.join(" ")
            )
            .to_lowercase();
            let compact_haystack: String = haystack
                .chars()
                .filter(|character| !character.is_whitespace())
                .collect();
            let matched = plan.text.should
                .iter()
                .filter(|term| {
                    haystack.contains(term.as_str())
                        || compact_haystack.contains(
                            &term
                                .chars()
                                .filter(|character| !character.is_whitespace())
                                .collect::<String>(),
                        )
                })
                .count();
            let excluded = plan.text.must_not.iter().any(|term| {
                haystack.contains(term.as_str())
                    || compact_haystack.contains(
                        &term
                            .chars()
                            .filter(|character| !character.is_whitespace())
                            .collect::<String>(),
                    )
            });
            let lexical_score = if plan.text.should.is_empty() || excluded {
                0.0
            } else {
                matched as f32 / plan.text.should.len() as f32
            };
            let semantic_score = record.embedding.as_ref().and_then(|image_vector| {
                query_embeddings
                    .iter()
                    .map(|query_vector| cosine_similarity(query_vector, image_vector))
                    .max_by(f32::total_cmp)
            });
            SearchCandidate {
                record: record.clone(),
                lexical_score,
                lexical_allowed: !excluded,
                semantic_score,
            }
        })
        .collect();

    let best_semantic_score = candidates
        .iter()
        .filter_map(|candidate| candidate.semantic_score)
        .max_by(f32::total_cmp);
    let threshold = semantic_threshold(best_semantic_score);
    let mut matches: Vec<ImageRecord> = candidates
        .into_iter()
        .filter_map(|candidate| {
            let semantic_match = semantic_search
                && candidate.lexical_allowed
                && candidate
                    .semantic_score
                    .is_some_and(|score| score >= threshold);
            let lexical_match = candidate.lexical_score > 0.0
                && candidate.lexical_allowed;
            (candidate.lexical_allowed
                && (!has_positive_query || lexical_match || semantic_match))
                .then(|| {
                let mut record = candidate.record;
                // Lexical/OCR hits are explicit evidence and should rank above a merely
                // similar-looking image. Pure semantic matches keep their raw cosine score.
                record.score = if candidate.lexical_score > 0.0 {
                    1.0 + candidate.lexical_score
                        + candidate.semantic_score.unwrap_or(0.0).max(0.0) * 0.15
                } else {
                    candidate.semantic_score.unwrap_or(0.0)
                };
                record
            })
        })
        .collect();
    matches.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.filename.cmp(&b.filename))
    });
    // Semantic search intentionally returns a compact high-confidence set.
    // Thumbnails are data URLs, so this also keeps the IPC response small.
    matches.truncate(plan.limit.unwrap_or(MAX_SEARCH_RESULTS));
    Ok(matches)
}

#[tauri::command]
async fn search_images(
    query: String,
    index: State<'_, AppIndex>,
) -> Result<Vec<ImageRecord>, String> {
    let index = index.inner().clone();
    // Text model initialization and vector ranking can also be expensive on the
    // first query, so keep them off the WebView thread.
    tauri::async_runtime::spawn_blocking(move || search_images_sync(query, index))
        .await
        .map_err(|error| format!("搜尋工作意外中止：{error}"))?
}

#[tauri::command]
fn get_model_settings(index: State<'_, AppIndex>) -> SettingsInfo {
    SettingsInfo {
        settings: current_settings(index.inner()),
        directml_available: directml_available(),
        directml_error: directml_error(),
        thumbnail_gpu_available: false,
        ocr_gpu_experimental: true,
    }
}

#[tauri::command]
fn update_model_settings(
    settings: ModelSettings,
    index: State<'_, AppIndex>,
) -> Result<SettingsInfo, String> {
    if settings.max_indexed_images == Some(0) {
        return Err("圖片上限至少必須是 1，或選擇無上限。".to_owned());
    }
    if settings.index_batch_size == 0 {
        return Err("SQLite 每批寫入張數至少必須是 1。".to_owned());
    }
    if let Some(path) = settings_file(index.inner()) {
        save_settings(&path, &settings)?;
    }
    let changed_clip_backend = index
        .settings
        .lock()
        .map_err(|_| "模型設定暫時無法使用。")?
        .clip_gpu
        != settings.clip_gpu;
    *index
        .settings
        .lock()
        .map_err(|_| "模型設定暫時無法使用。")? = settings.clone();
    if changed_clip_backend {
        if let Some(model_lock) = TEXT_MODEL.get() {
            if let Ok(mut model) = model_lock.lock() {
                *model = None;
            }
        }
    }
    Ok(SettingsInfo {
        settings,
        directml_available: directml_available(),
        directml_error: directml_error(),
        thumbnail_gpu_available: false,
        ocr_gpu_experimental: true,
    })
}

#[tauri::command]
fn list_face_groups(index: State<'_, AppIndex>) -> Result<Vec<FaceGroupSummary>, String> {
    let data = index.data.lock().map_err(|_| "人物索引暫時無法使用。")?;
    Ok(face_group_summaries(&data))
}

#[tauri::command]
fn label_face_group(
    group_id: String,
    name: String,
    index: State<'_, AppIndex>,
) -> Result<Vec<FaceGroupSummary>, String> {
    let normalized_name = name.trim().to_owned();
    let mut known_people = load_known_people(index.inner());
    let (person, summaries) = {
        let mut data = index.data.lock().map_err(|_| "人物索引暫時無法使用。")?;
        let group = data
            .face_groups
            .iter_mut()
            .find(|group| group.id == group_id)
            .ok_or_else(|| "找不到指定的人物群組。".to_owned())?;
        group.name = (!normalized_name.is_empty()).then_some(normalized_name.clone());
        let person = KnownPerson {
            id: group.id.clone(),
            name: normalized_name.clone(),
            embedding: group.centroid.clone(),
        };

        let named_groups: Vec<(String, String)> = data
            .face_groups
            .iter()
            .filter_map(|group| {
                group
                    .name
                    .as_ref()
                    .map(|name| (group.id.clone(), name.clone()))
            })
            .collect();
        for image in &mut data.images {
            image.people = named_groups
                .iter()
                .filter(|(id, _)| image.face_group_ids.contains(id))
                .map(|(_, name)| name.clone())
                .collect();
            image.people.sort();
            image.people.dedup();
        }
        (person, face_group_summaries(&data))
    };

    known_people.retain(|known| known.id != group_id);
    if !normalized_name.is_empty() {
        known_people.push(person);
    }
    if let Some(path) = people_file(index.inner()) {
        save_known_people(&path, &known_people)?;
    }
    Ok(summaries)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f32, expected: f32) {
        assert!((actual - expected).abs() < 0.0001, "{actual} != {expected}");
    }

    #[test]
    fn cosine_similarity_keeps_the_original_scale() {
        assert_close(cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]), 1.0);
        assert_close(cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]), -1.0);
        assert_close(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]), 0.0);
    }

    #[test]
    fn semantic_threshold_rejects_low_confidence_results() {
        assert_close(semantic_threshold(Some(0.33)), 0.26);
        assert_close(semantic_threshold(Some(0.24)), 0.20);
        assert_close(semantic_threshold(Some(0.18)), 0.20);
        assert_close(semantic_threshold(None), 0.20);
    }

    #[test]
    fn exif_datetime_is_normalized_for_date_queries() {
        assert_eq!(
            normalize_exif_datetime("2025:07:01 12:34:56"),
            Some("2025-07-01 12:34:56".to_owned())
        );
        assert_eq!(normalize_exif_datetime("not-a-date"), None);
        assert_eq!(normalize_exif_datetime("2025:13:01 12:34:56"), None);
    }

    #[test]
    fn rule_query_parser_extracts_date_person_and_semantic_terms() {
        let plan = parse_query("去年夏天在海邊的小明照片", &["小明".to_owned()]);
        let previous_year = time::OffsetDateTime::now_utc().year() - 1;
        assert_eq!(plan.semantic_query, "海邊");
        assert_eq!(plan.people_include, vec!["小明"]);
        assert_eq!(plan.face_presence, FacePresence::Required);
        assert_eq!(plan.date.from, Some(format!("{previous_year:04}-06-01 00:00:00")));
        assert_eq!(plan.date.to, Some(format!("{previous_year:04}-08-31 23:59:59")));
    }

    #[test]
    fn rule_query_parser_extracts_negative_face_extension_and_limit() {
        let plan = parse_query("沒有人的 jpg 照片 最多 20 張", &[]);
        assert_eq!(plan.face_presence, FacePresence::Forbidden);
        assert_eq!(plan.extensions, vec!["jpg"]);
        assert_eq!(plan.limit, Some(20));
        assert!(plan.semantic_query.is_empty());

        let negative = parse_query("不要狗", &[]);
        assert_eq!(negative.text.must_not, vec!["狗"]);
        assert!(negative.semantic_query.is_empty());
        let negative_without = parse_query("沒有狗", &[]);
        assert_eq!(negative_without.text.must_not, vec!["狗"]);
    }

    #[test]
    fn face_embeddings_are_grouped_only_when_they_are_similar() {
        let mut clusters = Vec::new();
        let (first_id, _) = assign_face_cluster(
            &mut clusters,
            &[],
            "image-1",
            "preview-1".to_owned(),
            &[1.0, 0.0, 0.0],
        );
        let (same_id, _) = assign_face_cluster(
            &mut clusters,
            &[],
            "image-2",
            "preview-2".to_owned(),
            &[0.98, 0.08, 0.0],
        );
        let (different_id, _) = assign_face_cluster(
            &mut clusters,
            &[],
            "image-3",
            "preview-3".to_owned(),
            &[0.0, 1.0, 0.0],
        );
        assert_eq!(first_id, same_id);
        assert_ne!(first_id, different_id);
        assert_eq!(clusters.len(), 2);
        assert_eq!(clusters[0].face_count, 2);
        assert_eq!(clusters[0].image_ids.len(), 2);
    }

    #[test]
    fn model_settings_are_backward_compatible() {
        let settings: ModelSettings = serde_json::from_str(r#"{"clip_gpu":true}"#).unwrap();
        assert!(settings.clip_gpu);
        assert_eq!(settings.max_indexed_images, Some(DEFAULT_MAX_INDEXED_IMAGES));
        assert_eq!(settings.index_batch_size, DEFAULT_INDEX_BATCH_SIZE);
        assert!(!settings.thumbnail_gpu);
        assert!(!settings.face_gpu);
        assert!(!settings.ocr_gpu);
    }

    #[test]
    fn thumbnail_dimensions_fit_the_longest_edge() {
        assert_eq!(thumbnail_dimensions(4_000, 3_000), (480, 360));
        assert_eq!(thumbnail_dimensions(3_000, 4_000), (360, 480));
        assert_eq!(thumbnail_dimensions(1, 1), (480, 480));
    }

    #[test]
    fn sqlite_index_cache_round_trip() {
        let cache_path = std::env::temp_dir().join(format!(
            "local-lens-cache-test-{}-{}.sqlite3",
            std::process::id(),
            SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos()
        ));
        let root = "C:\\Photos";
        let path = "C:\\Photos\\sample.jpg".to_owned();
        let record = ImageRecord {
            id: path.clone(),
            path: path.clone(),
            filename: "sample.jpg".to_owned(),
            modified_at: "2026-08-19".to_owned(),
            captured_at: Some("2025-07-01 12:34:56".to_owned()),
            width: Some(1200),
            height: Some(800),
            thumbnail: "data:image/jpeg;base64,test".to_owned(),
            ocr_text: "receipt".to_owned(),
            people: vec!["Tony".to_owned()],
            score: 1.0,
            embedding: Some(vec![0.25, 0.75]),
            face_group_ids: vec!["face-1".to_owned()],
        };
        let fingerprints = HashMap::from([(
            path.clone(),
            FileFingerprint {
                bytes: 42,
                modified_ns: 123,
            },
        )]);
        reset_index_cache(Some(&cache_path), root).unwrap();
        append_index_cache(Some(&cache_path), root, &[record], &fingerprints).unwrap();
        save_index_cache_state(Some(&cache_path), root, &[]).unwrap();
        let cached = load_cached_images(Some(&cache_path), root);
        let cached_record = cached.get(&path).unwrap();
        assert_eq!(cached_record.fingerprint.bytes, 42);
        assert_eq!(
            cached_record.record.captured_at.as_deref(),
            Some("2025-07-01 12:34:56")
        );
        assert_eq!(cached_record.record.ocr_text, "receipt");
        assert_eq!(cached_record.record.embedding, Some(vec![0.25, 0.75]));
        let _ = fs::remove_file(cache_path);
    }

    #[cfg(windows)]
    #[test]
    fn directml_capability_probe_is_safe() {
        let _ = directml_available();
    }
}

pub fn run() {
    tauri::Builder::default()
        .manage(AppIndex::default())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            // Installed applications may not be allowed to write beside the executable.
            // Keep FastEmbed's downloaded models in the platform app-data directory while
            // still allowing FASTEMBED_CACHE_DIR to override it for development.
            if let Ok(app_data) = app.path().app_data_dir() {
                let _ = fs::create_dir_all(&app_data);
                if std::env::var_os("FASTEMBED_CACHE_DIR").is_none() {
                    let model_dir = app_data.join("models");
                    let _ = fs::create_dir_all(&model_dir);
                    std::env::set_var("FASTEMBED_CACHE_DIR", model_dir);
                }
                let index = app.state::<AppIndex>();
                if let Ok(mut people_file) = index.people_file.lock() {
                    *people_file = Some(app_data.join("people.json"));
                };
                let model_settings_file = app_data.join("settings.json");
                let model_settings = load_settings(&model_settings_file);
                if let Ok(mut settings) = index.settings.lock() {
                    *settings = model_settings;
                }
                if let Ok(mut settings_file) = index.settings_file.lock() {
                    *settings_file = Some(model_settings_file);
                };
                if let Ok(mut cache_file) = index.cache_file.lock() {
                    *cache_file = Some(app_data.join("index.sqlite3"));
                };
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            scan_folder,
            pause_scan,
            resume_scan,
            search_images,
            get_model_settings,
            update_model_settings,
            list_face_groups,
            label_face_group
        ])
        .run(tauri::generate_context!())
        .expect("error while running Local Lens");
}
