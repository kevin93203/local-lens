use super::*;

pub(super) fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
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

pub(super) fn embed_queries(query: &str, use_gpu: bool) -> Vec<Vec<f32>> {
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

pub(super) fn semantic_threshold(best_score: Option<f32>) -> f32 {
    best_score
        .map(|score| (score - SEMANTIC_BEST_MARGIN).max(MIN_SEMANTIC_SIMILARITY))
        .unwrap_or(MIN_SEMANTIC_SIMILARITY)
}

pub(super) fn fts5_search_scores(path: Option<&Path>, text: &TextQueryPlan) -> Option<HashMap<String, f32>> {
    let path = path?;
    let connection = open_index_cache(path).ok()?;
    let mut scores = HashMap::new();
    let terms = text.should.iter().chain(text.must.iter());
    for term in terms {
        let escaped = term.replace('"', "\"\"");
        let match_query = format!("\"{escaped}\"");
        let mut statement = connection
            .prepare(
                "SELECT path FROM image_ocr_fts \
                 WHERE image_ocr_fts MATCH ?1",
            )
            .ok()?;
        let rows = statement
            .query_map(params![match_query], |row| row.get::<_, String>(0))
            .ok()?;
        for row in rows.flatten() {
            *scores.entry(row).or_insert(0.0) += 1.0;
        }
    }
    for term in &text.must_not {
        let escaped = term.replace('"', "\"\"");
        let match_query = format!("\"{escaped}\"");
        let mut statement = connection
            .prepare("SELECT path FROM image_ocr_fts WHERE image_ocr_fts MATCH ?1")
            .ok()?;
        let rows = statement
            .query_map(params![match_query], |row| row.get::<_, String>(0))
            .ok()?;
        for row in rows.flatten() {
            scores.remove(&row);
        }
    }
    (!scores.is_empty()).then_some(scores)
}

pub(super) fn sqlite_vec_search_scores(
    path: Option<&Path>,
    kind: &str,
    embeddings: &[Vec<f32>],
    limit: usize,
) -> Option<HashMap<String, f32>> {
    let path = path?;
    let connection = open_index_cache(path).ok()?;
    let mut scores = HashMap::new();
    for embedding in embeddings {
        if embedding.len() != VECTOR_DIMENSION {
            continue;
        }
        let blob = embedding_blob(embedding);
        let mut statement = connection
            .prepare(
                "SELECT rowid, distance FROM image_vectors \
                 WHERE embedding MATCH ?1 AND k = ?2 \
                 AND rowid IN (SELECT vector_rowid FROM vector_rows WHERE kind = ?3) \
                 ORDER BY distance",
            )
            .ok()?;
        let rows = statement
            .query_map(params![blob, limit as i64, kind], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, f64>(1)?))
            })
            .ok()?;
        for row in rows.flatten() {
            let (rowid, distance) = row;
            let path_value = connection
                .query_row(
                    "SELECT path FROM vector_rows WHERE vector_rowid = ?1 AND kind = ?2",
                    params![rowid, kind],
                    |row| row.get::<_, String>(0),
                )
                .ok();
            let Some(path_value) = path_value.filter(|value| !value.is_empty()) else {
                continue;
            };
            let score = (1.0 - distance).clamp(-1.0, 1.0) as f32;
            scores
                .entry(path_value)
                .and_modify(|current: &mut f32| *current = current.max(score))
                .or_insert(score);
        }
    }
    (!scores.is_empty()).then_some(scores)
}

pub(super) fn search_images_sync(query: String, index: AppIndex) -> Result<Vec<ImageRecord>, String> {
    let settings = current_settings(&index);
    let data = index.data.lock().map_err(|_| "索引暫時無法使用。")?;
    let records = &data.images;
    let cache_path = cache_file(&index);
    if query.trim().is_empty() {
        let mut results: Vec<ImageRecord> = records.iter().take(MAX_BROWSE_RESULTS).cloned().collect();
        hydrate_thumbnails(cache_path.as_deref(), &mut results);
        return Ok(results);
    }
    let mut known_people: Vec<String> = records
        .iter()
        .flat_map(|record| record.people.iter().cloned())
        .collect();
    known_people.sort_unstable();
    known_people.dedup();
    let plan = parse_query(&query, &known_people);
    let fts_scores = fts5_search_scores(cache_path.as_deref(), &plan.text);
    let query_embeddings = if !plan.semantic_query.is_empty()
        && records.iter().any(|record| record.embedding.is_some())
    {
        embed_queries(&plan.semantic_query, settings.clip_gpu)
    } else {
        Vec::new()
    };
    let vector_scores = sqlite_vec_search_scores(
        cache_path.as_deref(),
        "clip",
        &query_embeddings,
        MAX_SEARCH_RESULTS.saturating_mul(5),
    );
    let semantic_search = vector_scores
        .as_ref()
        .map_or(!query_embeddings.is_empty(), |scores| !scores.is_empty());
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
            let positive_terms = plan.text.should.iter().chain(plan.text.must.iter());
            let matched = positive_terms
                .clone()
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
            let total_terms = plan.text.should.len() + plan.text.must.len();
            let memory_lexical_score = if total_terms == 0 {
                0.0
            } else {
                matched as f32 / total_terms as f32
            };
            let lexical_score = if excluded {
                0.0
            } else if let Some(scores) = &fts_scores {
                scores
                    .get(&record.path)
                    .copied()
                    .unwrap_or(0.0)
                    .max(memory_lexical_score)
            } else {
                memory_lexical_score
            };
            let semantic_score = record.embedding.as_ref().and_then(|image_vector| {
                if let Some(scores) = &vector_scores {
                    scores.get(&record.path).copied()
                } else {
                    query_embeddings
                        .iter()
                        .map(|query_vector| cosine_similarity(query_vector, image_vector))
                        .max_by(f32::total_cmp)
                }
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
    // Only this bounded result set receives Base64 thumbnail data URLs for IPC.
    matches.truncate(plan.limit.unwrap_or(MAX_SEARCH_RESULTS));
    hydrate_thumbnails(cache_path.as_deref(), &mut matches);
    Ok(matches)
}
