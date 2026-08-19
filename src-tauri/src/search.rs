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

pub(super) fn fts5_search_scores(
    path: Option<&Path>,
    root: Option<&str>,
    text: &TextQueryPlan,
) -> Option<HashMap<String, f32>> {
    let path = path?;
    let root = root?;
    let connection = open_index_cache(path).ok()?;
    let mut scores = HashMap::new();
    let terms = text.should.iter().chain(text.must.iter());
    for term in terms {
        let escaped = term.replace('"', "\"\"");
        let match_query = format!("\"{escaped}\"");
        let mut statement = connection
            .prepare(
                "SELECT path FROM image_ocr_fts \
                 WHERE image_ocr_fts MATCH ?1 AND root = ?2",
            )
            .ok()?;
        let rows = statement
            .query_map(params![match_query, root], |row| row.get::<_, String>(0))
            .ok()?;
        for row in rows.flatten() {
            *scores.entry(row).or_insert(0.0) += 1.0;
        }
    }
    for term in &text.must_not {
        let escaped = term.replace('"', "\"\"");
        let match_query = format!("\"{escaped}\"");
        let mut statement = connection
            .prepare("SELECT path FROM image_ocr_fts WHERE image_ocr_fts MATCH ?1 AND root = ?2")
            .ok()?;
        let rows = statement
            .query_map(params![match_query, root], |row| row.get::<_, String>(0))
            .ok()?;
        for row in rows.flatten() {
            scores.remove(&row);
        }
    }
    (!scores.is_empty()).then_some(scores)
}

pub(super) fn sqlite_vec_search_scores(
    path: Option<&Path>,
    root: Option<&str>,
    kind: &str,
    embeddings: &[Vec<f32>],
    limit: usize,
) -> Option<HashMap<String, f32>> {
    let path = path?;
    let root = root?;
    let connection = open_index_cache(path).ok()?;
    let knn_limit = if limit == usize::MAX {
        connection
            .query_row(
                "SELECT COUNT(*) FROM vector_rows WHERE kind = ?1",
                params![kind],
                |row| row.get::<_, i64>(0),
            )
            .ok()
            .and_then(|count| usize::try_from(count).ok())
            .unwrap_or(1)
            .max(1)
    } else {
        limit.max(1)
    };
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
                 AND rowid IN (SELECT vector_rowid FROM vector_rows WHERE kind = ?3 AND root = ?4) \
                 ORDER BY distance",
            )
            .ok()?;
        let rows = statement
            .query_map(params![blob, knn_limit as i64, kind, root], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, f64>(1)?))
            })
            .ok()?;
        for row in rows.flatten() {
            let (rowid, distance) = row;
            let path_value = connection
                .query_row(
                    "SELECT path FROM vector_rows WHERE vector_rowid = ?1 AND kind = ?2 AND root = ?3",
                    params![rowid, kind, root],
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

#[derive(Serialize)]
pub(super) struct SearchPage {
    pub(super) images: Vec<ImageRecord>,
    pub(super) total: usize,
    pub(super) has_more: bool,
}

struct ScoredCandidate {
    index: usize,
    lexical_score: f32,
    lexical_allowed: bool,
    semantic_score: Option<f32>,
}

fn score_record(
    index: usize,
    record: &ImageRecord,
    plan: &QueryPlan,
    fts_scores: Option<&HashMap<String, f32>>,
    vector_scores: Option<&HashMap<String, f32>>,
) -> Option<ScoredCandidate> {
    if !record_matches_plan(record, plan) {
        return None;
    }
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
    } else if let Some(scores) = fts_scores {
        scores
            .get(&record.path)
            .copied()
            .unwrap_or(0.0)
            .max(memory_lexical_score)
    } else {
        memory_lexical_score
    };
    // sqlite-vec is the only source of image embeddings. Search candidates
    // carry only an index and scores, never a complete ImageRecord copy.
    let semantic_score = vector_scores.and_then(|scores| scores.get(&record.path).copied());
    Some(ScoredCandidate {
        index,
        lexical_score,
        lexical_allowed: !excluded,
        semantic_score,
    })
}

fn page_from_result_refs(
    results: &[SearchResultRef],
    records: &[ImageRecord],
    offset: usize,
    page_size: usize,
    cache_path: Option<&Path>,
) -> SearchPage {
    let total = results.len();
    let start = offset.min(total);
    let end = start.saturating_add(page_size).min(total);
    let mut images: Vec<ImageRecord> = results[start..end]
        .iter()
        .map(|result| {
            let mut record = records[result.index].clone();
            record.score = result.score;
            record
        })
        .collect();
    hydrate_thumbnails(cache_path, &mut images);
    SearchPage {
        images,
        total,
        has_more: end < total,
    }
}

pub(super) fn search_images_page_sync(
    query: String,
    index: AppIndex,
    offset: usize,
    page_size: usize,
) -> Result<SearchPage, String> {
    let settings = current_settings(&index);
    let data = index.data.lock().map_err(|_| "索引暫時無法使用。")?;
    let records = &data.images;
    let cache_path = cache_file(&index);
    let root = current_root(&index);
    let page_size = page_size.max(1).min(MAX_RESULT_PAGE_SIZE);
    if query.trim().is_empty() {
        let total = records.len();
        let start = offset.min(total);
        let end = start.saturating_add(page_size).min(total);
        let mut results: Vec<ImageRecord> = records[start..end].to_vec();
        hydrate_thumbnails(cache_path.as_deref(), &mut results);
        return Ok(SearchPage {
            images: results,
            total,
            has_more: end < total,
        });
    }
    if let Ok(session) = index.search_session.lock() {
        if let Some(session) = session.as_ref().filter(|session| {
            session.root.as_ref() == root.as_ref()
                && session.query.as_str() == query.as_str()
        }) {
            return Ok(page_from_result_refs(
                &session.results,
                records,
                offset,
                page_size,
                cache_path.as_deref(),
            ));
        }
    }
    let mut known_people: Vec<String> = records
        .iter()
        .flat_map(|record| record.people.iter().cloned())
        .collect();
    known_people.sort_unstable();
    known_people.dedup();
    let plan = parse_query(&query, &known_people);
    let clip_vectors_available = has_clip_vectors(cache_path.as_deref(), root.as_deref());
    let fts_scores = fts5_search_scores(cache_path.as_deref(), root.as_deref(), &plan.text);
    let query_embeddings = if !plan.semantic_query.is_empty() && clip_vectors_available {
        embed_queries(&plan.semantic_query, settings.clip_gpu)
    } else {
        Vec::new()
    };
    let vector_scores = sqlite_vec_search_scores(
        cache_path.as_deref(),
        root.as_deref(),
        "clip",
        &query_embeddings,
        // Fetch every vector row so the adaptive threshold is applied to the
        // complete result set, then filter back to this root. Only the
        // requested page is hydrated below.
        usize::MAX,
    );
    let semantic_search = vector_scores
        .as_ref()
        .is_some_and(|scores| !scores.is_empty());
    let has_positive_query = !plan.semantic_query.is_empty()
        || !plan.text.should.is_empty()
        || !plan.text.must.is_empty();

    // First pass only retains the best semantic score needed by the adaptive
    // threshold. No ImageRecord is cloned or stored for all candidates.
    let best_semantic_score = records
        .iter()
        .enumerate()
        .filter_map(|(index, record)| {
            score_record(
                index,
                record,
                &plan,
                fts_scores.as_ref(),
                vector_scores.as_ref(),
            )
        })
        .filter_map(|candidate| candidate.semantic_score)
        .max_by(f32::total_cmp);
    let threshold = semantic_threshold(best_semantic_score);
    let mut ranked_matches = Vec::with_capacity(records.len());

    // Second pass keeps only index + score for every qualifying result. This
    // minimal session is reused by subsequent pages; complete ImageRecords are
    // cloned only for the page requested by the UI.
    for (index, record) in records.iter().enumerate() {
        let Some(candidate) = score_record(
            index,
            record,
            &plan,
            fts_scores.as_ref(),
            vector_scores.as_ref(),
        ) else {
            continue;
        };
        let semantic_match = semantic_search
            && candidate.lexical_allowed
            && candidate
                .semantic_score
                .is_some_and(|score| score >= threshold);
        let lexical_match = candidate.lexical_score > 0.0 && candidate.lexical_allowed;
        if candidate.lexical_allowed
            && (!has_positive_query || lexical_match || semantic_match)
        {
            let score = if candidate.lexical_score > 0.0 {
                1.0 + candidate.lexical_score
                    + candidate.semantic_score.unwrap_or(0.0).max(0.0) * 0.15
            } else {
                candidate.semantic_score.unwrap_or(0.0)
            };
            ranked_matches.push(SearchResultRef {
                index: candidate.index,
                score,
            });
        }
    }

    ranked_matches.sort_unstable_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| records[left.index].filename.cmp(&records[right.index].filename))
            .then_with(|| left.index.cmp(&right.index))
    });
    if let Some(limit) = plan.limit {
        ranked_matches.truncate(limit);
    }
    let page = page_from_result_refs(
        &ranked_matches,
        records,
        offset,
        page_size,
        cache_path.as_deref(),
    );
    if let Ok(mut session) = index.search_session.lock() {
        *session = Some(SearchSessionState {
            root,
            query,
            results: ranked_matches,
        });
    }
    Ok(page)
}

pub(super) fn search_images_sync(
    query: String,
    index: AppIndex,
) -> Result<Vec<ImageRecord>, String> {
    search_images_page_sync(query, index, 0, DEFAULT_RESULT_PAGE_SIZE).map(|page| page.images)
}
