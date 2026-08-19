use super::*;

pub(super) fn register_sqlite_vec() -> Result<(), String> {
    SQLITE_VEC_STATUS
        .get_or_init(|| {
            let result = unsafe {
                sqlite3_auto_extension(Some(std::mem::transmute(
                    sqlite_vec::sqlite3_vec_init as *const (),
                )))
            };
            (result == 0)
                .then_some(())
                .ok_or_else(|| format!("無法註冊 sqlite-vec SQLite 擴充功能（錯誤碼 {result}）。"))
        })
        .clone()
}

pub(super) fn open_index_cache(path: &Path) -> Result<Connection, String> {
    register_sqlite_vec()?;
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
            -- Keep binary thumbnails outside record_json so cached metadata can
            -- be loaded without retaining every thumbnail in memory.
            CREATE TABLE IF NOT EXISTS image_thumbnails (
                path TEXT PRIMARY KEY NOT NULL,
                root TEXT NOT NULL,
                jpeg BLOB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS image_thumbnails_root ON image_thumbnails(root);
            CREATE VIRTUAL TABLE IF NOT EXISTS image_ocr_fts USING fts5(
                path UNINDEXED,
                root UNINDEXED,
                filename,
                ocr_text,
                people,
                tokenize = 'unicode61 remove_diacritics 0'
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS image_vectors USING vec0(
                embedding float[512] distance_metric=cosine
            );
            CREATE TABLE IF NOT EXISTS vector_rows (
                vector_rowid INTEGER PRIMARY KEY NOT NULL,
                kind TEXT NOT NULL,
                root TEXT NOT NULL,
                path TEXT NOT NULL DEFAULT '',
                group_id TEXT NOT NULL DEFAULT '',
                UNIQUE(kind, root, path, group_id)
            );
            CREATE INDEX IF NOT EXISTS vector_rows_scope ON vector_rows(kind, root);
            CREATE TABLE IF NOT EXISTS scan_state (
                root TEXT PRIMARY KEY NOT NULL,
                face_groups_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS clip_vector_migrations (
                root TEXT PRIMARY KEY NOT NULL
            );
            CREATE TABLE IF NOT EXISTS scan_seen (
                root TEXT NOT NULL,
                path TEXT NOT NULL,
                PRIMARY KEY(root, path)
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

pub(super) fn embedding_blob(values: &[f32]) -> Vec<u8> {
    let mut blob = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
    for value in values {
        blob.extend_from_slice(&value.to_le_bytes());
    }
    blob
}

pub(super) fn delete_vector_entry(
    transaction: &Transaction<'_>,
    kind: &str,
    root: &str,
    path: &str,
    group_id: &str,
) -> Result<(), String> {
    let rowid = transaction
        .query_row(
            "SELECT vector_rowid FROM vector_rows WHERE kind = ?1 AND root = ?2 AND path = ?3 AND group_id = ?4",
            params![kind, root, path, group_id],
            |row| row.get::<_, i64>(0),
        )
        .optional()
        .map_err(|error| error.to_string())?;
    if let Some(rowid) = rowid {
        transaction
            .execute("DELETE FROM image_vectors WHERE rowid = ?1", params![rowid])
            .map_err(|error| error.to_string())?;
        transaction
            .execute("DELETE FROM vector_rows WHERE vector_rowid = ?1", params![rowid])
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

pub(super) fn upsert_vector_entry(
    transaction: &Transaction<'_>,
    kind: &str,
    root: &str,
    path: &str,
    group_id: &str,
    embedding: &[f32],
) -> Result<(), String> {
    if embedding.len() != VECTOR_DIMENSION {
        return delete_vector_entry(transaction, kind, root, path, group_id);
    }
    let existing_rowid = transaction
        .query_row(
            "SELECT vector_rowid FROM vector_rows WHERE kind = ?1 AND root = ?2 AND path = ?3 AND group_id = ?4",
            params![kind, root, path, group_id],
            |row| row.get::<_, i64>(0),
        )
        .optional()
        .map_err(|error| error.to_string())?;
    if let Some(rowid) = existing_rowid {
        transaction
            .execute("DELETE FROM image_vectors WHERE rowid = ?1", params![rowid])
            .map_err(|error| error.to_string())?;
        transaction
            .execute("DELETE FROM vector_rows WHERE vector_rowid = ?1", params![rowid])
            .map_err(|error| error.to_string())?;
    }
    let blob = embedding_blob(embedding);
    let rowid = if let Some(rowid) = existing_rowid {
        transaction
            .execute(
                "INSERT INTO image_vectors(rowid, embedding) VALUES (?1, ?2)",
                params![rowid, blob],
            )
            .map_err(|error| error.to_string())?;
        rowid
    } else {
        transaction
            .execute("INSERT INTO image_vectors(embedding) VALUES (?1)", params![blob])
            .map_err(|error| error.to_string())?;
        transaction.last_insert_rowid()
    };
    transaction
        .execute(
            "INSERT INTO vector_rows(vector_rowid, kind, root, path, group_id) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![rowid, kind, root, path, group_id],
        )
        .map_err(|error| error.to_string())?;
    Ok(())
}

#[derive(Deserialize)]
struct LegacyClipCacheRecord {
    #[serde(default)]
    embedding: Option<Vec<f32>>,
}

fn flush_legacy_clip_embeddings(
    connection: &mut Connection,
    root: &str,
    pending: &mut Vec<(String, Vec<f32>)>,
) -> Result<(), String> {
    if pending.is_empty() {
        return Ok(());
    }
    let transaction = connection.transaction().map_err(|error| error.to_string())?;
    for (path, embedding) in pending.drain(..) {
        upsert_vector_entry(&transaction, "clip", root, &path, "", &embedding)?;
    }
    transaction.commit().map_err(|error| error.to_string())
}

fn migrate_legacy_clip_vectors(path: &Path, root: &str) -> Result<(), String> {
    let connection = open_index_cache(path)?;
    let migrated: bool = connection
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM clip_vector_migrations WHERE root = ?1)",
            params![root],
            |row| row.get::<_, i64>(0),
        )
        .map(|value| value != 0)
        .map_err(|error| error.to_string())?;
    drop(connection);
    if migrated {
        return Ok(());
    }

    // Read legacy JSON one row at a time and retain at most a small batch of
    // vectors. New records never contain this field; this is only for caches
    // created before sqlite-vec became the authoritative vector store.
    let read_connection = open_index_cache(path)?;
    let mut write_connection = open_index_cache(path)?;
    let mut statement = read_connection
        .prepare("SELECT path, record_json FROM image_cache WHERE root = ?1")
        .map_err(|error| error.to_string())?;
    let rows = statement
        .query_map(params![root], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .map_err(|error| error.to_string())?;
    let mut pending = Vec::new();
    for row in rows {
        let (path, record_json) = row.map_err(|error| error.to_string())?;
        let Some(embedding) = serde_json::from_str::<LegacyClipCacheRecord>(&record_json)
            .ok()
            .and_then(|record| record.embedding)
            .filter(|embedding| embedding.len() == VECTOR_DIMENSION)
        else {
            continue;
        };
        pending.push((path, embedding));
        if pending.len() >= MAX_PENDING_EMBEDDINGS {
            flush_legacy_clip_embeddings(&mut write_connection, root, &mut pending)?;
        }
    }
    drop(statement);
    drop(read_connection);
    flush_legacy_clip_embeddings(&mut write_connection, root, &mut pending)?;
    write_connection
        .execute(
            "INSERT OR REPLACE INTO clip_vector_migrations(root) VALUES (?1)",
            params![root],
        )
        .map_err(|error| error.to_string())?;
    Ok(())
}

pub(super) struct CacheReader {
    connection: Connection,
    root: String,
}

impl CacheReader {
    pub(super) fn lookup(&self, path: &str) -> Option<CachedImage> {
        let row = self
            .connection
            .query_row(
                "SELECT c.bytes, c.modified_ns, c.record_json, \
                        EXISTS(SELECT 1 FROM image_thumbnails t \
                               WHERE t.path = c.path AND t.root = c.root) \
                 FROM image_cache c WHERE c.root = ?1 AND c.path = ?2",
                params![self.root, path],
                |row| {
                    Ok((
                        row.get::<_, u64>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, i64>(3)? != 0,
                    ))
                },
            )
            .optional()
            .ok()
            .flatten()?;
        Some(CachedImage {
            fingerprint: FileFingerprint {
                bytes: row.0,
                modified_ns: row.1.parse().ok()?,
            },
            // Older record_json values may still contain a thumbnail field.
            // Serde ignores that legacy field; it is recreated lazily when
            // the cache row is reused.
            record: serde_json::from_str(&row.2).ok()?,
            thumbnail_available: row.3,
        })
    }
}

pub(super) fn open_cache_reader(path: Option<&Path>, root: &str) -> Option<CacheReader> {
    let path = path?;
    let _ = migrate_legacy_clip_vectors(path, root);
    Some(CacheReader {
        connection: open_index_cache(path).ok()?,
        root: root.to_owned(),
    })
}

pub(super) fn load_cached_face_groups(path: Option<&Path>, root: &str) -> Vec<FaceCluster> {
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

pub(super) fn begin_scan_cache(path: Option<&Path>, root: &str) -> Result<(), String> {
    let Some(path) = path else { return Ok(()) };
    let connection = open_index_cache(path)?;
    // Keep existing metadata, FTS rows and CLIP vectors available for lookup
    // while this scan is running. scan_seen is the bounded reconciliation set.
    connection
        .execute("DELETE FROM scan_seen WHERE root = ?1", params![root])
        .map_err(|error| error.to_string())?;
    connection
        .execute("DELETE FROM scan_state WHERE root = ?1", params![root])
        .map_err(|error| error.to_string())?;
    Ok(())
}

pub(super) fn append_index_cache(
    path: Option<&Path>,
    root: &str,
    records: &[ImageRecord],
    fingerprints: &HashMap<String, FileFingerprint>,
    clip_embeddings: &HashMap<String, Option<Vec<f32>>>,
) -> Result<(), String> {
    let Some(path) = path else { return Ok(()) };
    let mut connection = open_index_cache(path)?;
    let transaction = connection.transaction().map_err(|error| error.to_string())?;
    for record in records {
        let Some(fingerprint) = fingerprints.get(&record.path) else { continue };
        let payload = serde_json::to_string(&CachedImageRecord::from_record(record))
            .map_err(|error| error.to_string())?;
        transaction
            .execute(
                "DELETE FROM image_ocr_fts WHERE path = ?1",
                params![record.path],
            )
            .map_err(|error| error.to_string())?;
        transaction
            .execute(
                "INSERT INTO image_ocr_fts(path, root, filename, ocr_text, people) VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    record.path,
                    root,
                    record.filename,
                    record.ocr_text,
                    record.people.join(" ")
                ],
            )
            .map_err(|error| error.to_string())?;
        transaction
            .execute(
                "INSERT OR REPLACE INTO image_cache(root, path, bytes, modified_ns, captured_at, record_json) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    root,
                    record.path,
                    fingerprint.bytes,
                    fingerprint.modified_ns.to_string(),
                    record.captured_at.as_deref(),
                    payload
                ],
            )
            .map_err(|error| error.to_string())?;
        transaction
            .execute(
                "INSERT OR REPLACE INTO scan_seen(root, path) VALUES (?1, ?2)",
                params![root, record.path],
            )
            .map_err(|error| error.to_string())?;
        if let Some(jpeg) = decode_thumbnail_data_url(&record.thumbnail) {
            transaction
                .execute(
                    "INSERT OR REPLACE INTO image_thumbnails(path, root, jpeg) VALUES (?1, ?2, ?3)",
                    params![record.path, root, jpeg],
                )
                .map_err(|error| error.to_string())?;
        }
    }
    for (path, embedding) in clip_embeddings {
        if let Some(embedding) = embedding {
            upsert_vector_entry(&transaction, "clip", root, path, "", embedding)?;
        } else {
            delete_vector_entry(&transaction, "clip", root, path, "")?;
        }
    }
    transaction.commit().map_err(|error| error.to_string())
}

pub(super) fn append_index_cache_and_release(
    path: Option<&Path>,
    root: &str,
    records: &mut [ImageRecord],
    fingerprints: &mut HashMap<String, FileFingerprint>,
    clip_embeddings: &mut HashMap<String, Option<Vec<f32>>>,
) -> Result<(), String> {
    append_index_cache(path, root, records, fingerprints, clip_embeddings)?;
    if path.is_some() {
        // The durable copy is now in image_thumbnails. Do not retain the
        // Base64 data URL in the full in-memory index.
        for record in records {
            if fingerprints.contains_key(&record.path) {
                record.thumbnail.clear();
            }
        }
        fingerprints.clear();
        clip_embeddings.clear();
    }
    Ok(())
}

pub(super) fn cleanup_stale_cache(path: Option<&Path>, root: &str) -> Result<(), String> {
    let Some(path) = path else { return Ok(()) };
    let mut connection = open_index_cache(path)?;
    let transaction = connection.transaction().map_err(|error| error.to_string())?;
    transaction
        .execute(
            "DELETE FROM image_ocr_fts
             WHERE root = ?1
               AND path NOT IN (SELECT path FROM scan_seen WHERE root = ?1)",
            params![root],
        )
        .map_err(|error| error.to_string())?;
    transaction
        .execute(
            "DELETE FROM image_cache
             WHERE root = ?1
               AND path NOT IN (SELECT path FROM scan_seen WHERE root = ?1)",
            params![root],
        )
        .map_err(|error| error.to_string())?;
    transaction
        .execute("DELETE FROM scan_seen WHERE root = ?1", params![root])
        .map_err(|error| error.to_string())?;
    transaction.commit().map_err(|error| error.to_string())
}

pub(super) fn cleanup_stale_thumbnails(path: Option<&Path>, root: &str) -> Result<(), String> {
    let Some(path) = path else { return Ok(()) };
    let connection = open_index_cache(path)?;
    connection
        .execute(
            "DELETE FROM image_thumbnails
             WHERE root = ?1
               AND path NOT IN (SELECT path FROM image_cache WHERE root = ?1)",
            params![root],
        )
        .map_err(|error| error.to_string())?;
    Ok(())
}

pub(super) fn hydrate_thumbnails(path: Option<&Path>, records: &mut [ImageRecord]) {
    let Some(path) = path else { return };
    if records.iter().all(|record| !record.thumbnail.is_empty()) {
        return;
    }
    let Ok(connection) = open_index_cache(path) else { return };
    let Ok(mut statement) = connection.prepare(
        "SELECT jpeg FROM image_thumbnails WHERE path = ?1",
    ) else {
        return;
    };
    for record in records {
        if !record.thumbnail.is_empty() {
            continue;
        }
        let bytes = statement
            .query_row(params![record.path], |row| row.get::<_, Vec<u8>>(0))
            .optional()
            .ok()
            .flatten();
        if let Some(bytes) = bytes {
            record.thumbnail = thumbnail_data_url(&bytes);
        }
    }
}

pub(super) fn cleanup_stale_clip_vectors(path: Option<&Path>, root: &str) -> Result<(), String> {
    let Some(path) = path else { return Ok(()) };
    let connection = open_index_cache(path)?;
    let stale_ids: Vec<i64> = connection
        .prepare(
            "SELECT vector_rowid FROM vector_rows
             WHERE kind = 'clip' AND root = ?1
               AND path NOT IN (SELECT path FROM image_cache WHERE root = ?1)",
        )
        .and_then(|mut statement| {
            statement
                .query_map(params![root], |row| row.get::<_, i64>(0))
                .map(|rows| rows.filter_map(Result::ok).collect())
        })
        .map_err(|error| error.to_string())?;
    for rowid in stale_ids {
        connection
            .execute("DELETE FROM image_vectors WHERE rowid = ?1", params![rowid])
            .map_err(|error| error.to_string())?;
    }
    connection
        .execute(
            "DELETE FROM vector_rows
             WHERE kind = 'clip' AND root = ?1
               AND path NOT IN (SELECT path FROM image_cache WHERE root = ?1)",
            params![root],
        )
        .map_err(|error| error.to_string())?;
    Ok(())
}

pub(super) fn has_clip_vectors(path: Option<&Path>, root: Option<&str>) -> bool {
    let Some(path) = path else { return false };
    let Some(root) = root else { return false };
    let Ok(connection) = open_index_cache(path) else { return false };
    connection
        .query_row(
            "SELECT EXISTS(
                 SELECT 1 FROM vector_rows
                 WHERE kind = 'clip' AND root = ?1
             )",
            params![root],
            |row| row.get::<_, i64>(0),
        )
        .map(|value| value != 0)
        .unwrap_or(false)
}

pub(super) fn save_index_cache_state(
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

pub(super) fn sync_face_vectors(
    path: Option<&Path>,
    root: &str,
    face_clusters: &[FaceCluster],
) -> Result<(), String> {
    let Some(path) = path else { return Ok(()) };
    let mut connection = open_index_cache(path)?;
    let transaction = connection.transaction().map_err(|error| error.to_string())?;
    let old_ids: Vec<i64> = transaction
        .prepare("SELECT vector_rowid FROM vector_rows WHERE kind = 'face' AND root = ?1")
        .and_then(|mut statement| {
            statement
                .query_map(params![root], |row| row.get::<_, i64>(0))
                .map(|rows| rows.filter_map(Result::ok).collect())
        })
        .map_err(|error| error.to_string())?;
    for rowid in old_ids {
        transaction
            .execute("DELETE FROM image_vectors WHERE rowid = ?1", params![rowid])
            .map_err(|error| error.to_string())?;
    }
    transaction
        .execute(
            "DELETE FROM vector_rows WHERE kind = 'face' AND root = ?1",
            params![root],
        )
        .map_err(|error| error.to_string())?;
    for cluster in face_clusters {
        upsert_vector_entry(
            &transaction,
            "face",
            root,
            "",
            &cluster.id,
            &cluster.centroid,
        )?;
    }
    transaction.commit().map_err(|error| error.to_string())
}

pub(super) fn find_nearest_face_group(
    path: Option<&Path>,
    root: &str,
    embedding: &[f32],
) -> Result<Option<(String, f32)>, String> {
    let Some(path) = path else { return Ok(None) };
    if embedding.len() != VECTOR_DIMENSION {
        return Ok(None);
    }
    let connection = open_index_cache(path)?;
    let blob = embedding_blob(embedding);
    let nearest = connection
        .query_row(
            "SELECT rowid, distance FROM image_vectors \
             WHERE embedding MATCH ?1 AND k = 1 \
             AND rowid IN (SELECT vector_rowid FROM vector_rows WHERE kind = 'face' AND root = ?2) \
             ORDER BY distance LIMIT 1",
            params![blob, root],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, f64>(1)?)),
        )
        .optional()
        .map_err(|error| error.to_string())?;
    let Some((rowid, distance)) = nearest else { return Ok(None) };
    let group_id = connection
        .query_row(
            "SELECT group_id FROM vector_rows WHERE vector_rowid = ?1 AND kind = 'face' AND root = ?2",
            params![rowid, root],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(|error| error.to_string())?;
    Ok(group_id.map(|group_id| (group_id, (1.0 - distance) as f32)))
}
