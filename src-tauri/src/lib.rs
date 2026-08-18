use std::{
    ffi::OsString,
    fs,
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{Arc, Mutex},
    time::SystemTime,
};

use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use image::{codecs::jpeg::JpegEncoder, imageops::FilterType, ImageFormat};
use serde::Serialize;
use tauri::{Emitter, State};
use walkdir::WalkDir;

const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "webp", "gif", "bmp"];
const MAX_INDEXED_IMAGES: usize = 3_000;
const MAX_SEARCH_RESULTS: usize = 200;

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
}

#[derive(Clone, Default)]
struct AppIndex(Arc<Mutex<Vec<ImageRecord>>>);

#[derive(Serialize)]
struct ScanResult {
    root: String,
    indexed: usize,
    skipped: usize,
    ocr_available: bool,
}

#[derive(Clone, Serialize)]
struct ScanProgress {
    processed: usize,
    total: usize,
    indexed: usize,
    skipped: usize,
    ocr_available: bool,
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
    let mut skipped = 0;
    let ocr_program = tesseract_program();
    let ocr_available = has_tesseract(&ocr_program);
    app.emit(
        "scan-progress",
        ScanProgress {
            processed: 0,
            total,
            indexed: 0,
            skipped: 0,
            ocr_available,
        },
    )
    .map_err(|error| error.to_string())?;
    for (position, path) in candidates.iter().enumerate() {
        match make_thumbnail(path) {
            Ok((thumbnail, width, height)) => {
                let path_text = path.to_string_lossy().into_owned();
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
                    people: vec![],
                    score: 1.0,
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
                },
            )
            .map_err(|error| error.to_string())?;
        }
    }
    records.sort_by(|a, b| a.filename.to_lowercase().cmp(&b.filename.to_lowercase()));
    let indexed = records.len();
    *index.0.lock().map_err(|_| "索引暫時無法使用。")? = records;
    Ok(ScanResult {
        root: folder,
        indexed,
        skipped,
        ocr_available,
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

#[tauri::command]
fn search_images(query: String, index: State<'_, AppIndex>) -> Result<Vec<ImageRecord>, String> {
    let terms: Vec<String> = query
        .to_lowercase()
        .split_whitespace()
        .map(str::to_owned)
        .collect();
    let records = index.0.lock().map_err(|_| "索引暫時無法使用。")?;
    let mut matches: Vec<ImageRecord> = records
        .iter()
        .filter_map(|record| {
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
            (terms.is_empty() || matched > 0).then(|| {
                let mut copy = record.clone();
                copy.score = if terms.is_empty() {
                    1.0
                } else {
                    matched as f32 / terms.len() as f32
                };
                copy
            })
        })
        .collect();
    matches.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.filename.cmp(&b.filename))
    });
    // Thumbnails are data URLs. Keep the IPC response bounded, otherwise the
    // renderer can freeze while receiving thousands of images at once.
    matches.truncate(MAX_SEARCH_RESULTS);
    Ok(matches)
}

pub fn run() {
    tauri::Builder::default()
        .manage(AppIndex::default())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![scan_folder, search_images])
        .run(tauri::generate_context!())
        .expect("error while running Local Lens");
}
