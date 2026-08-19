use super::*;

pub(super) fn clip_execution_providers(use_gpu: bool) -> Vec<fastembed::ExecutionProviderDispatch> {
    #[cfg(windows)]
    if use_gpu {
        return vec![DirectMLExecutionProvider::default()
            .build()
            .error_on_failure()];
    }
    let _ = use_gpu;
    Vec::new()
}

pub(super) fn directml_available() -> bool {
    directml_status().is_ok()
}

pub(super) fn directml_status() -> Result<(), String> {
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

pub(super) fn directml_error() -> Option<String> {
    directml_status().err()
}

pub(super) fn load_image_embedding(use_gpu: bool) -> (Option<ImageEmbedding>, bool, Option<String>) {
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

pub(super) fn load_face_engine(use_gpu: bool) -> Result<(FaceEngine, bool, Option<String>), String> {
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
