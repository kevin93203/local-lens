use super::*;

pub(super) fn thumbnail_dimensions(width: u32, height: u32) -> (u32, u32) {
    const THUMBNAIL_BOUND: u32 = 480;
    if width >= height {
        (
            THUMBNAIL_BOUND,
            ((height as u64 * THUMBNAIL_BOUND as u64 + width as u64 / 2) / width as u64)
                .max(1) as u32,
        )
    } else {
        (
            ((width as u64 * THUMBNAIL_BOUND as u64 + height as u64 / 2) / height as u64)
                .max(1) as u32,
            THUMBNAIL_BOUND,
        )
    }
}

pub(super) fn make_thumbnail(path: &Path, _use_gpu: bool) -> Result<(String, u32, u32), String> {
    let image = image::open(path).map_err(|error| error.to_string())?;
    let (width, height) = (image.width(), image.height());
    let source = image.to_rgb8();
    let (thumbnail_width, thumbnail_height) = thumbnail_dimensions(width, height);
    let source = fir::images::Image::from_vec_u8(
        width,
        height,
        source.into_raw(),
        fir::PixelType::U8x3,
    )
    .map_err(|error| format!("無法準備縮圖像素：{error}"))?;
    let mut destination = fir::images::Image::new(
        thumbnail_width,
        thumbnail_height,
        fir::PixelType::U8x3,
    );
    fir::Resizer::new()
        .resize(
            &source,
            &mut destination,
            &fir::ResizeOptions::new().resize_alg(fir::ResizeAlg::Convolution(
                fir::FilterType::Bilinear,
            )),
        )
        .map_err(|error| format!("縮放縮圖失敗：{error}"))?;
    let thumbnail = image::RgbImage::from_raw(
        thumbnail_width,
        thumbnail_height,
        destination.into_vec(),
    )
    .ok_or_else(|| "縮圖像素格式不正確。".to_owned())?;
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

pub(super) fn decode_thumbnail_data_url(value: &str) -> Option<Vec<u8>> {
    value
        .strip_prefix("data:image/jpeg;base64,")
        .and_then(|encoded| BASE64.decode(encoded).ok())
}

pub(super) fn thumbnail_data_url(bytes: &[u8]) -> String {
    format!("data:image/jpeg;base64,{}", BASE64.encode(bytes))
}

pub(super) fn make_face_thumbnail(image: &image::DynamicImage, face: &Face) -> Option<String> {
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
