use super::*;

pub(super) fn build_index(
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
    let _ = sync_face_vectors(cache_path.as_deref(), &folder, &face_clusters);
    let mut cache_written = 0usize;
    let mut pending_thumbnails = 0usize;
    let mut pending_clip_embeddings: HashMap<String, Option<Vec<f32>>> = HashMap::new();
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
    let mut semantic_available =
        image_model.is_some() || has_clip_vectors(cache_path.as_deref(), Some(&folder));
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
                let mut record = cached.record.clone().into_record(path_text);
                // Older installations stored the thumbnail inside record_json.
                // Recreate it once and move it to the dedicated BLOB table;
                // subsequent scans can reuse it without loading it into the
                // full in-memory index.
                if !cached.thumbnail_available {
                    if let Ok((thumbnail, _, _)) = make_thumbnail(path, settings.thumbnail_gpu) {
                        record.thumbnail = thumbnail;
                        pending_thumbnails += 1;
                    }
                }
                records.push(record);
                reused += 1;
                if records.len().saturating_sub(cache_written) >= batch_size
                    || pending_thumbnails >= MAX_PENDING_THUMBNAILS
                    || pending_clip_embeddings.len() >= MAX_PENDING_EMBEDDINGS
                {
                    if append_index_cache_and_release(
                        cache_path.as_deref(),
                        &folder,
                        &mut records[cache_written..],
                        &fingerprints,
                        &mut pending_clip_embeddings,
                    )
                    .is_ok()
                    {
                        cache_written = records.len();
                        pending_thumbnails = 0;
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
                            let vector_hint = find_nearest_face_group(
                                cache_path.as_deref(),
                                &folder,
                                &face.embedding,
                            )
                            .ok()
                            .flatten();
                            let (group_id, person_name) = assign_face_cluster_with_hint(
                                &mut face_clusters,
                                &known_people,
                                &path_text,
                                face_preview,
                                &face.embedding,
                                vector_hint,
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
                    face_group_ids,
                });
                if cache_path.is_some() {
                    pending_clip_embeddings.insert(
                        records.last().map(|record| record.path.clone()).unwrap_or_default(),
                        embedding,
                    );
                }
                if cache_path.is_some()
                    && records
                        .last()
                        .is_some_and(|record| !record.thumbnail.is_empty())
                {
                    pending_thumbnails += 1;
                }
            }
            Err(_) => skipped += 1,
        }
        if records.len().saturating_sub(cache_written) >= batch_size
            || pending_thumbnails >= MAX_PENDING_THUMBNAILS
            || pending_clip_embeddings.len() >= MAX_PENDING_EMBEDDINGS
        {
            if append_index_cache_and_release(
                cache_path.as_deref(),
                &folder,
                &mut records[cache_written..],
                &fingerprints,
                &mut pending_clip_embeddings,
            )
            .is_ok()
            {
                cache_written = records.len();
                pending_thumbnails = 0;
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
    refresh_face_cluster_counts(&mut face_clusters, &records);
    // A cache write failure must not discard the freshly built in-memory index.
    // Keep the processing order until this final flush so cache_written still
    // identifies the unwritten suffix; sort only after persistence completes.
    let _ = append_index_cache_and_release(
        cache_path.as_deref(),
        &folder,
        &mut records[cache_written..],
        &fingerprints,
        &mut pending_clip_embeddings,
    );
    let _ = cleanup_stale_thumbnails(cache_path.as_deref(), &folder);
    let _ = cleanup_stale_clip_vectors(cache_path.as_deref(), &folder);
    semantic_available = has_clip_vectors(cache_path.as_deref(), Some(&folder));
    let _ = sync_face_vectors(cache_path.as_deref(), &folder, &face_clusters);
    let _ = save_index_cache_state(cache_path.as_deref(), &folder, &face_clusters);
    records.sort_by(|a, b| a.filename.to_lowercase().cmp(&b.filename.to_lowercase()));
    let indexed = records.len();
    let face_groups = face_clusters.len();
    *index.data.lock().map_err(|_| "索引暫時無法使用。")? = IndexData {
        images: records,
        face_groups: face_clusters,
    };
    if let Ok(mut current_root) = index.root.lock() {
        *current_root = Some(folder.clone());
    }
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
