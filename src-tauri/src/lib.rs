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
use rusqlite::{
    ffi::sqlite3_auto_extension,
    params,
    Connection,
    OptionalExtension,
    Transaction,
};
use tauri::{Emitter, Manager, State};
use walkdir::WalkDir;

mod face;
mod cache;
mod indexer;
mod media;
mod people;
mod query;
mod runtime;
mod search;
mod thumbnail;

use cache::*;
use face::{Face, FaceEngine};
use indexer::*;
use media::*;
use people::*;
use query::*;
use runtime::*;
use search::*;
use thumbnail::*;

const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "webp", "gif", "bmp"];
const DEFAULT_MAX_INDEXED_IMAGES: usize = 3_000;
const DEFAULT_INDEX_BATCH_SIZE: usize = 50;
// Keep only a bounded number of encoded thumbnails in records while a cache
// batch is waiting to be committed. The normal SQLite batch size is the same,
// but this separate cap also protects users who configure a very large batch.
const MAX_PENDING_THUMBNAILS: usize = 50;
const MAX_PENDING_EMBEDDINGS: usize = 50;
const MAX_BROWSE_RESULTS: usize = 200;
const MAX_SEARCH_RESULTS: usize = 60;
const MIN_SEMANTIC_SIMILARITY: f32 = 0.20;
const SEMANTIC_BEST_MARGIN: f32 = 0.07;
const FACE_MATCH_THRESHOLD: f32 = 0.45;
const VECTOR_DIMENSION: usize = 512;
const FACE_MODEL_REPOSITORY: &str = "WePrompt/buffalo_sc";

struct CachedTextModel {
    requested_gpu: bool,
    model: TextEmbedding,
}

// Keep the text encoder alive after its first use. Loading the ONNX session
// for every keystroke/search would make subsequent searches unnecessarily slow.
static TEXT_MODEL: OnceLock<Mutex<Option<CachedTextModel>>> = OnceLock::new();
static DIRECTML_STATUS: OnceLock<Result<(), String>> = OnceLock::new();
static SQLITE_VEC_STATUS: OnceLock<Result<(), String>> = OnceLock::new();

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
    // CLIP image vectors live exclusively in sqlite-vec; only face group IDs
    // remain in the in-memory record.
    #[serde(skip)]
    face_group_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedImageRecord {
    // This metadata payload intentionally excludes thumbnails and CLIP vectors.
    filename: String,
    modified_at: String,
    #[serde(default)]
    captured_at: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    ocr_text: String,
    people: Vec<String>,
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
            ocr_text: record.ocr_text.clone(),
            people: record.people.clone(),
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
            // Thumbnails are loaded lazily from image_thumbnails for the small
            // result set returned to the WebView.
            thumbnail: String::new(),
            ocr_text: self.ocr_text,
            people: self.people,
            score: 1.0,
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
    thumbnail_available: bool,
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
    root: Arc<Mutex<Option<String>>>,
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

fn people_file(index: &AppIndex) -> Option<PathBuf> {
    index.people_file.lock().ok()?.clone()
}

fn settings_file(index: &AppIndex) -> Option<PathBuf> {
    index.settings_file.lock().ok()?.clone()
}

fn cache_file(index: &AppIndex) -> Option<PathBuf> {
    index.cache_file.lock().ok()?.clone()
}

fn current_root(index: &AppIndex) -> Option<String> {
    index.root.lock().ok()?.clone()
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
            thumbnail: "data:image/jpeg;base64,AA==".to_owned(),
            ocr_text: "receipt".to_owned(),
            people: vec!["Tony".to_owned()],
            score: 1.0,
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
        append_index_cache(
            Some(&cache_path),
            root,
            &[record],
            &fingerprints,
            &HashMap::new(),
        )
        .unwrap();
        save_index_cache_state(Some(&cache_path), root, &[]).unwrap();
        let connection = open_index_cache(&cache_path).unwrap();
        let payload: String = connection
            .query_row(
                "SELECT record_json FROM image_cache WHERE path = ?1",
                params![path],
                |row| row.get(0),
            )
            .unwrap();
        assert!(!payload.contains("\"embedding\""));
        drop(connection);
        let cached = load_cached_images(Some(&cache_path), root);
        let cached_record = cached.get(&path).unwrap();
        assert_eq!(cached_record.fingerprint.bytes, 42);
        assert_eq!(
            cached_record.record.captured_at.as_deref(),
            Some("2025-07-01 12:34:56")
        );
        assert_eq!(cached_record.record.ocr_text, "receipt");
        assert!(cached_record.thumbnail_available);
        assert!(cached_record.record.clone().into_record(path.clone()).thumbnail.is_empty());
        let mut hydrated = vec![cached_record.record.clone().into_record(path.clone())];
        hydrate_thumbnails(Some(&cache_path), &mut hydrated);
        assert_eq!(hydrated[0].thumbnail, "data:image/jpeg;base64,AA==");
        let _ = fs::remove_file(cache_path);
    }

    #[test]
    fn legacy_json_embeddings_are_migrated_to_sqlite_vec() {
        let cache_path = std::env::temp_dir().join(format!(
            "local-lens-legacy-embedding-test-{}-{}.sqlite3",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let root = "C:\\Photos";
        let path = "C:\\Photos\\legacy.jpg";
        let legacy_json = serde_json::json!({
            "filename": "legacy.jpg",
            "modified_at": "1",
            "captured_at": null,
            "width": 1200,
            "height": 800,
            "thumbnail": "",
            "ocr_text": "",
            "people": [],
            "embedding": vec![1.0; VECTOR_DIMENSION],
            "face_group_ids": []
        })
        .to_string();
        let connection = open_index_cache(&cache_path).unwrap();
        connection
            .execute(
                "INSERT INTO image_cache(root, path, bytes, modified_ns, record_json) VALUES (?1, ?2, ?3, ?4, ?5)",
                params![root, path, 42_u64, "1", legacy_json],
            )
            .unwrap();
        drop(connection);

        let cached = load_cached_images(Some(&cache_path), root);
        assert!(cached.contains_key(path));
        assert!(has_clip_vectors(Some(&cache_path), Some(root)));
        let scores = sqlite_vec_search_scores(
            Some(&cache_path),
            Some(root),
            "clip",
            &[vec![1.0; VECTOR_DIMENSION]],
            1,
        )
        .unwrap();
        assert!(scores.get(path).is_some_and(|score| *score > 0.99));
        let _ = fs::remove_file(cache_path);
    }

    #[test]
    fn sqlite_fts5_and_vec_indexes_round_trip() {
        let cache_path = std::env::temp_dir().join(format!(
            "local-lens-fts-vec-test-{}-{}.sqlite3",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let root = "C:\\Photos";
        let path = "C:\\Photos\\receipt.jpg".to_owned();
        let record = ImageRecord {
            id: path.clone(),
            path: path.clone(),
            filename: "receipt.jpg".to_owned(),
            modified_at: "2026-08-19".to_owned(),
            captured_at: None,
            width: Some(1200),
            height: Some(800),
            thumbnail: String::new(),
            ocr_text: "receipt invoice".to_owned(),
            people: vec!["Tony".to_owned()],
            score: 1.0,
            face_group_ids: Vec::new(),
        };
        let fingerprints = HashMap::from([(
            path.clone(),
            FileFingerprint {
                bytes: 42,
                modified_ns: 123,
            },
        )]);

        let clip_embeddings = HashMap::from([(path.clone(), Some(vec![1.0; VECTOR_DIMENSION]))]);
        append_index_cache(
            Some(&cache_path),
            root,
            &[record],
            &fingerprints,
            &clip_embeddings,
        )
        .unwrap();
        let text = TextQueryPlan {
            should: vec!["receipt".to_owned()],
            must: Vec::new(),
            must_not: Vec::new(),
        };
        let text_scores = fts5_search_scores(Some(&cache_path), Some(root), &text).unwrap();
        assert_eq!(text_scores.get(&path), Some(&1.0));

        let vector_scores = sqlite_vec_search_scores(
            Some(&cache_path),
            Some(root),
            "clip",
            &[vec![1.0; VECTOR_DIMENSION]],
            10,
        )
        .unwrap();
        assert!(vector_scores.get(&path).is_some_and(|score| *score > 0.99));

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
