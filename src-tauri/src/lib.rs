use std::{
    collections::HashSet,
    ffi::OsString,
    fs,
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{Arc, Mutex, OnceLock},
    time::SystemTime,
};

use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use fastembed::{
    EmbeddingModel, ImageEmbedding, ImageEmbeddingModel, ImageInitOptions, InitOptions,
    TextEmbedding,
};
use hf_hub::api::sync::ApiBuilder;
use image::{codecs::jpeg::JpegEncoder, imageops::FilterType, ImageFormat};
use serde::{Deserialize, Serialize};
use tauri::{Emitter, Manager, State};
use walkdir::WalkDir;

mod face;
use face::{Face, FaceEngine};

const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "webp", "gif", "bmp"];
const MAX_INDEXED_IMAGES: usize = 3_000;
const MAX_BROWSE_RESULTS: usize = 200;
const MAX_SEARCH_RESULTS: usize = 60;
const MIN_SEMANTIC_SIMILARITY: f32 = 0.20;
const SEMANTIC_BEST_MARGIN: f32 = 0.07;
const FACE_MATCH_THRESHOLD: f32 = 0.45;
const FACE_MODEL_REPOSITORY: &str = "WePrompt/buffalo_sc";

// Keep the text encoder alive after its first use. Loading the ONNX session
// for every keystroke/search would make subsequent searches unnecessarily slow.
static TEXT_MODEL: OnceLock<Mutex<Option<TextEmbedding>>> = OnceLock::new();

#[derive(Debug, Clone, Serialize)]
struct ImageRecord {
    id: String,
    path: String,
    filename: String,
    modified_at: String,
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

#[derive(Debug, Clone)]
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
}

#[derive(Serialize)]
struct ScanResult {
    root: String,
    indexed: usize,
    skipped: usize,
    ocr_available: bool,
    semantic_available: bool,
    face_available: bool,
    faces_detected: usize,
    face_groups: usize,
}

#[derive(Clone, Serialize)]
struct ScanProgress {
    processed: usize,
    total: usize,
    indexed: usize,
    skipped: usize,
    ocr_available: bool,
    semantic_available: bool,
    face_available: bool,
    faces_detected: usize,
}

fn is_supported_image(path: &Path) -> bool {
    path.extension()
        .and_then(|value| value.to_str())
        .map(|extension| IMAGE_EXTENSIONS.contains(&extension.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

fn make_thumbnail(path: &Path) -> Result<(String, u32, u32), String> {
    let image = image::open(path).map_err(|error| error.to_string())?;
    let (width, height) = (image.width(), image.height());
    let thumbnail = image.resize(480, 480, FilterType::Triangle).to_rgb8();
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

fn load_face_engine() -> Result<FaceEngine, String> {
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
    FaceEngine::new(detector_path, recognizer_path)
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
) -> Option<OcrCandidate> {
    let output = Command::new(program)
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
        .arg("tsv")
        .output()
        .ok()?;
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

fn extract_ocr_text(path: &Path, program: &OsString) -> String {
    let enhanced_path = make_ocr_image(path);
    let enhanced = enhanced_path.as_deref().unwrap_or(path);
    let configured_language = tesseract_language();
    let english = OsString::from("eng");
    let mut candidates = Vec::new();
    for (input, mode) in [(enhanced, 6_u8), (path, 11_u8)] {
        if let Some(candidate) = run_tesseract(input, program, &configured_language, mode) {
            candidates.push(candidate);
        } else if configured_language != english {
            if let Some(candidate) = run_tesseract(input, program, &english, mode) {
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
        .take(MAX_INDEXED_IMAGES)
        .collect();
    let total = candidates.len();
    let mut records = Vec::new();
    let mut face_clusters = Vec::new();
    let known_people = load_known_people(&index);
    let mut faces_detected = 0;
    let mut skipped = 0;
    let ocr_program = tesseract_program();
    let ocr_available = has_tesseract(&ocr_program);
    // The first initialization may download the local ONNX models into the
    // FastEmbed cache. It runs inside the blocking worker, never on the UI loop.
    let mut image_model = ImageEmbedding::try_new(
        ImageInitOptions::new(ImageEmbeddingModel::ClipVitB32).with_show_download_progress(false),
    )
    .ok();
    let mut semantic_available = image_model.is_some();
    let mut face_engine = load_face_engine().ok();
    let face_available = face_engine.is_some();
    app.emit(
        "scan-progress",
        ScanProgress {
            processed: 0,
            total,
            indexed: 0,
            skipped: 0,
            ocr_available,
            semantic_available,
            face_available,
            faces_detected,
        },
    )
    .map_err(|error| error.to_string())?;
    for (position, path) in candidates.iter().enumerate() {
        match make_thumbnail(path) {
            Ok((thumbnail, width, height)) => {
                let path_text = path.to_string_lossy().into_owned();
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
                    width: Some(width),
                    height: Some(height),
                    thumbnail,
                    ocr_text: if ocr_available {
                        extract_ocr_text(path, &ocr_program)
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
        let processed = position + 1;
        // Emit in small batches rather than once per file; large folders remain responsive.
        if processed == total || processed % 5 == 0 {
            app.emit(
                "scan-progress",
                ScanProgress {
                    processed,
                    total,
                    indexed: records.len(),
                    skipped,
                    ocr_available,
                    semantic_available,
                    face_available,
                    faces_detected,
                },
            )
            .map_err(|error| error.to_string())?;
        }
    }
    records.sort_by(|a, b| a.filename.to_lowercase().cmp(&b.filename.to_lowercase()));
    let indexed = records.len();
    let face_groups = face_clusters.len();
    *index.data.lock().map_err(|_| "索引暫時無法使用。")? = IndexData {
        images: records,
        face_groups: face_clusters,
    };
    Ok(ScanResult {
        root: folder,
        indexed,
        skipped,
        ocr_available,
        semantic_available,
        face_available,
        faces_detected,
        face_groups,
    })
}

#[tauri::command]
async fn scan_folder(
    folder: String,
    app: tauri::AppHandle,
    index: State<'_, AppIndex>,
) -> Result<ScanResult, String> {
    let index = index.inner().clone();
    // Decoding images and encoding thumbnails are CPU / disk heavy.  Use a
    // blocking worker so the WebView and native window keep processing events.
    tauri::async_runtime::spawn_blocking(move || build_index(folder, app, index))
        .await
        .map_err(|error| format!("索引工作意外中止：{error}"))?
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

fn embed_queries(query: &str) -> Vec<Vec<f32>> {
    let model_lock = TEXT_MODEL.get_or_init(|| Mutex::new(None));
    let Ok(mut model) = model_lock.lock() else {
        return Vec::new();
    };
    if model.is_none() {
        *model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::ClipVitB32).with_show_download_progress(false),
        )
        .ok();
    }
    let Some(model) = model.as_mut() else {
        return Vec::new();
    };
    // CLIP was trained with short image captions. Comparing both the user's
    // original wording and a photo-oriented prompt is more stable for natural
    // language queries than relying on only one phrasing.
    model
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

fn search_images_sync(query: String, index: AppIndex) -> Result<Vec<ImageRecord>, String> {
    let normalized_query = query.to_lowercase();
    let terms: Vec<String> = normalized_query
        .split(|character: char| {
            character.is_whitespace()
                || matches!(
                    character,
                    '的' | '和' | '與' | '在' | '有' | '、' | '，' | ','
                )
        })
        .filter(|term| !term.is_empty())
        .map(str::to_owned)
        .collect();
    let data = index.data.lock().map_err(|_| "索引暫時無法使用。")?;
    let records = &data.images;
    if query.trim().is_empty() {
        return Ok(records.iter().take(MAX_BROWSE_RESULTS).cloned().collect());
    }
    let query_embeddings = if records.iter().any(|record| record.embedding.is_some()) {
        embed_queries(&query)
    } else {
        Vec::new()
    };
    let semantic_search = !query_embeddings.is_empty();

    struct SearchCandidate {
        record: ImageRecord,
        lexical_score: f32,
        semantic_score: Option<f32>,
    }

    let candidates: Vec<SearchCandidate> = records
        .iter()
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
            let matched = terms
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
            let lexical_score = if terms.is_empty() {
                0.0
            } else {
                matched as f32 / terms.len() as f32
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
                && candidate
                    .semantic_score
                    .is_some_and(|score| score >= threshold);
            (candidate.lexical_score > 0.0 || semantic_match).then(|| {
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
    matches.truncate(MAX_SEARCH_RESULTS);
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
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            scan_folder,
            search_images,
            list_face_groups,
            label_face_group
        ])
        .run(tauri::generate_context!())
        .expect("error while running Local Lens");
}
