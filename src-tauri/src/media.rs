use super::*;

pub(super) fn format_modified(time: SystemTime) -> String {
    time.duration_since(SystemTime::UNIX_EPOCH)
        .map(|duration| duration.as_secs().to_string())
        .unwrap_or_default()
}

pub(super) fn normalize_exif_datetime(value: &str) -> Option<String> {
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

pub(super) fn extract_exif_capture_time(path: &Path) -> Option<String> {
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

pub(super) fn tesseract_program() -> OsString {
    std::env::var_os("LOCAL_LENS_TESSERACT").unwrap_or_else(|| OsString::from("tesseract"))
}

pub(super) fn tesseract_language() -> OsString {
    std::env::var_os("LOCAL_LENS_TESSERACT_LANG").unwrap_or_else(|| OsString::from("eng+chi_tra"))
}

pub(super) fn has_tesseract(program: &OsString) -> bool {
    Command::new(program)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

pub(super) fn tesseract_opencl_available(program: &OsString) -> bool {
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

pub(super) fn make_ocr_image(path: &Path) -> Option<PathBuf> {
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

pub(super) fn extract_ocr_text(path: &Path, program: &OsString, use_gpu: bool) -> String {
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
