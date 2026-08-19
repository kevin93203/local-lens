use super::*;

pub(super) fn normalize_embedding(values: &[f32]) -> Vec<f32> {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm == 0.0 {
        values.to_vec()
    } else {
        values.iter().map(|value| value / norm).collect()
    }
}

pub(super) fn make_face_group_id(embedding: &[f32]) -> String {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for value in embedding.iter().take(64) {
        value.to_bits().hash(&mut hasher);
    }
    format!("face-{:016x}", hasher.finish())
}

pub(super) fn refresh_face_cluster_counts(face_clusters: &mut Vec<FaceCluster>, records: &[ImageRecord]) {
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

pub(super) fn current_settings(index: &AppIndex) -> ModelSettings {
    index
        .settings
        .lock()
        .map(|value| value.clone())
        .unwrap_or_default()
}

pub(super) fn load_settings(path: &Path) -> ModelSettings {
    fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default()
}

pub(super) fn save_settings(path: &Path, settings: &ModelSettings) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let bytes = serde_json::to_vec_pretty(settings).map_err(|error| error.to_string())?;
    fs::write(path, bytes).map_err(|error| error.to_string())
}

pub(super) fn load_known_people(index: &AppIndex) -> Vec<KnownPerson> {
    let Some(path) = people_file(index) else {
        return Vec::new();
    };
    fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default()
}

pub(super) fn save_known_people(path: &Path, people: &[KnownPerson]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let bytes = serde_json::to_vec_pretty(people).map_err(|error| error.to_string())?;
    fs::write(path, bytes).map_err(|error| error.to_string())
}

pub(super) fn update_face_centroid(cluster: &mut FaceCluster, embedding: &[f32]) {
    let count = cluster.face_count as f32;
    for (centroid, value) in cluster.centroid.iter_mut().zip(embedding.iter()) {
        *centroid = (*centroid * count + value) / (count + 1.0);
    }
    cluster.centroid = normalize_embedding(&cluster.centroid);
    cluster.face_count += 1;
}

#[allow(dead_code)]
pub(super) fn assign_face_cluster(
    clusters: &mut Vec<FaceCluster>,
    known_people: &[KnownPerson],
    image_id: &str,
    preview: String,
    raw_embedding: &[f32],
) -> (String, Option<String>) {
    assign_face_cluster_with_hint(
        clusters,
        known_people,
        image_id,
        preview,
        raw_embedding,
        None,
    )
}

pub(super) fn assign_face_cluster_with_hint(
    clusters: &mut Vec<FaceCluster>,
    known_people: &[KnownPerson],
    image_id: &str,
    preview: String,
    raw_embedding: &[f32],
    vector_hint: Option<(String, f32)>,
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
        vector_hint
            .filter(|(_, score)| *score >= FACE_MATCH_THRESHOLD)
            .and_then(|(group_id, _)| clusters.iter().position(|cluster| cluster.id == group_id))
            .or_else(|| {
                clusters
                    .iter()
                    .enumerate()
                    .filter_map(|(index, cluster)| {
                        let score = cosine_similarity(&embedding, &cluster.centroid);
                        (score >= FACE_MATCH_THRESHOLD).then_some((index, score))
                    })
                    .max_by(|left, right| left.1.total_cmp(&right.1))
                    .map(|(index, _)| index)
            })
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

pub(super) fn face_group_summaries(data: &IndexData) -> Vec<FaceGroupSummary> {
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
