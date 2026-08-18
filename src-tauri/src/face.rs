use std::{path::Path, thread::available_parallelism};

use image::{imageops, Rgb, RgbImage};
use imageproc::geometric_transformations::{warp_into, Border, Interpolation, Projection};
use nalgebra::{Matrix2, Matrix2x3, Vector2};
use ndarray::Array4;
use ort::session::{builder::GraphOptimizationLevel, Session};

const DETECTOR_SIZE: u32 = 640;
const RECOGNIZER_SIZE: u32 = 112;
const SCORE_THRESHOLD: f32 = 0.65;
const NMS_THRESHOLD: f32 = 0.4;
const STRIDES: [u32; 3] = [8, 16, 32];
const ANCHOR_NUM: u32 = 2;
const ARCFACE_TEMPLATE: [[f32; 2]; 5] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];

#[derive(Debug, Clone, Copy)]
pub struct FaceBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl FaceBox {
    fn area(self) -> f32 {
        (self.x2 - self.x1).max(0.0) * (self.y2 - self.y1).max(0.0)
    }

    fn iou(self, other: Self) -> f32 {
        let x1 = self.x1.max(other.x1);
        let y1 = self.y1.max(other.y1);
        let x2 = self.x2.min(other.x2);
        let y2 = self.y2.min(other.y2);
        let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
        let union = self.area() + other.area() - intersection;
        if union <= 0.0 {
            0.0
        } else {
            intersection / union
        }
    }
}

#[derive(Debug, Clone)]
struct DetectedFace {
    bbox: FaceBox,
    landmarks: [[f32; 2]; 5],
    score: f32,
}

pub struct Face {
    pub bbox: FaceBox,
    pub embedding: Vec<f32>,
}

pub struct FaceEngine {
    detector: Session,
    recognizer: Session,
}

impl FaceEngine {
    pub fn new(
        detector_path: impl AsRef<Path>,
        recognizer_path: impl AsRef<Path>,
    ) -> Result<Self, String> {
        // Keep one CPU available for the WebView and cap inference parallelism so
        // indexing a large folder does not make the desktop UI feel unresponsive.
        let threads = available_parallelism()
            .map(|count| count.get().saturating_sub(1).clamp(1, 4))
            .unwrap_or(1);
        let detector = Session::builder()
            .and_then(|builder| builder.with_optimization_level(GraphOptimizationLevel::Level3))
            .and_then(|builder| builder.with_intra_threads(threads))
            .and_then(|builder| builder.commit_from_file(detector_path))
            .map_err(|error| error.to_string())?;
        let recognizer = Session::builder()
            .and_then(|builder| builder.with_optimization_level(GraphOptimizationLevel::Level3))
            .and_then(|builder| builder.with_intra_threads(threads))
            .and_then(|builder| builder.commit_from_file(recognizer_path))
            .map_err(|error| error.to_string())?;
        Ok(Self {
            detector,
            recognizer,
        })
    }

    pub fn run(&mut self, image: &RgbImage) -> Result<Vec<Face>, String> {
        let detected = self.detect(image)?;
        detected
            .into_iter()
            .map(|face| {
                let embedding = self.recognize(image, &face)?;
                Ok(Face {
                    bbox: face.bbox,
                    embedding,
                })
            })
            .collect()
    }

    fn detect(&mut self, image: &RgbImage) -> Result<Vec<DetectedFace>, String> {
        let (original_width, original_height) = image.dimensions();
        let scale = DETECTOR_SIZE as f32 / original_width.max(original_height) as f32;
        let width = (original_width as f32 * scale) as u32;
        let height = (original_height as f32 * scale) as u32;
        let resized = imageops::resize(image, width, height, imageops::FilterType::Triangle);

        let plane_size = DETECTOR_SIZE as usize * DETECTOR_SIZE as usize;
        let mut data = vec![(0.0 - 127.5) / 128.0; plane_size * 3];
        let (red, rest) = data.split_at_mut(plane_size);
        let (green, blue) = rest.split_at_mut(plane_size);
        for y in 0..height as usize {
            for x in 0..width as usize {
                let pixel = resized.get_pixel(x as u32, y as u32);
                let index = y * DETECTOR_SIZE as usize + x;
                red[index] = (pixel[0] as f32 - 127.5) / 128.0;
                green[index] = (pixel[1] as f32 - 127.5) / 128.0;
                blue[index] = (pixel[2] as f32 - 127.5) / 128.0;
            }
        }
        let input =
            Array4::from_shape_vec((1, 3, DETECTOR_SIZE as usize, DETECTOR_SIZE as usize), data)
                .map_err(|error| error.to_string())?;
        let outputs = self
            .detector
            .run(ort::inputs![input].map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
        if outputs.len() < 9 {
            return Err("人臉偵測模型輸出格式不相容。".to_owned());
        }

        let inverse_scale = 1.0 / scale;
        let mut detections = Vec::new();
        for (level, stride) in STRIDES.iter().enumerate() {
            let scores: Vec<f32> = outputs[level]
                .try_extract_tensor::<f32>()
                .map_err(|error| error.to_string())?
                .iter()
                .copied()
                .collect();
            let boxes: Vec<f32> = outputs[level + 3]
                .try_extract_tensor::<f32>()
                .map_err(|error| error.to_string())?
                .iter()
                .copied()
                .collect();
            let landmarks: Vec<f32> = outputs[level + 6]
                .try_extract_tensor::<f32>()
                .map_err(|error| error.to_string())?
                .iter()
                .copied()
                .collect();
            let feature_size = DETECTOR_SIZE / stride;
            let scaling = *stride as f32 * inverse_scale;
            let mut index = 0usize;
            for y in 0..feature_size {
                for x in 0..feature_size {
                    for _ in 0..ANCHOR_NUM {
                        if index >= scores.len()
                            || index * 4 + 3 >= boxes.len()
                            || index * 10 + 9 >= landmarks.len()
                        {
                            break;
                        }
                        let score = scores[index];
                        if score > SCORE_THRESHOLD {
                            let box_offset = index * 4;
                            let landmark_offset = index * 10;
                            let mut points = [[0.0; 2]; 5];
                            for point in 0..5 {
                                points[point][0] =
                                    (landmarks[landmark_offset + point * 2] + x as f32) * scaling;
                                points[point][1] = (landmarks[landmark_offset + point * 2 + 1]
                                    + y as f32)
                                    * scaling;
                            }
                            detections.push(DetectedFace {
                                bbox: FaceBox {
                                    x1: (x as f32 - boxes[box_offset]) * scaling,
                                    y1: (y as f32 - boxes[box_offset + 1]) * scaling,
                                    x2: (x as f32 + boxes[box_offset + 2]) * scaling,
                                    y2: (y as f32 + boxes[box_offset + 3]) * scaling,
                                },
                                landmarks: points,
                                score,
                            });
                        }
                        index += 1;
                    }
                }
            }
        }
        detections.sort_by(|left, right| right.score.total_cmp(&left.score));
        let mut kept: Vec<DetectedFace> = Vec::new();
        for detection in detections {
            if kept
                .iter()
                .all(|existing| existing.bbox.iou(detection.bbox) <= NMS_THRESHOLD)
            {
                kept.push(detection);
            }
        }
        Ok(kept)
    }

    fn recognize(&mut self, image: &RgbImage, face: &DetectedFace) -> Result<Vec<f32>, String> {
        let transform = estimate_similarity_transform(&face.landmarks, &ARCFACE_TEMPLATE);
        let projection = Projection::from_matrix([
            transform[(0, 0)],
            transform[(0, 1)],
            transform[(0, 2)],
            transform[(1, 0)],
            transform[(1, 1)],
            transform[(1, 2)],
            0.0,
            0.0,
            1.0,
        ])
        .ok_or_else(|| "無法對齊人臉。".to_owned())?;
        let mut aligned = RgbImage::new(RECOGNIZER_SIZE, RECOGNIZER_SIZE);
        warp_into(
            image,
            projection,
            Interpolation::Bilinear,
            Border::Constant(Rgb([0, 0, 0])),
            &mut aligned,
        );
        let mut input =
            Array4::<f32>::zeros((1, 3, RECOGNIZER_SIZE as usize, RECOGNIZER_SIZE as usize));
        for y in 0..RECOGNIZER_SIZE {
            for x in 0..RECOGNIZER_SIZE {
                let pixel = aligned.get_pixel(x, y);
                input[[0, 0, y as usize, x as usize]] = (pixel[0] as f32 - 127.5) / 128.0;
                input[[0, 1, y as usize, x as usize]] = (pixel[1] as f32 - 127.5) / 128.0;
                input[[0, 2, y as usize, x as usize]] = (pixel[2] as f32 - 127.5) / 128.0;
            }
        }
        let outputs = self
            .recognizer
            .run(ort::inputs![input].map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
        let embedding: Vec<f32> = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|error| error.to_string())?
            .iter()
            .copied()
            .collect();
        if embedding.len() != 512 {
            return Err("人臉辨識模型沒有回傳 512 維向量。".to_owned());
        }
        Ok(embedding)
    }
}

fn estimate_similarity_transform(
    source: &[[f32; 2]; 5],
    destination: &[[f32; 2]; 5],
) -> Matrix2x3<f32> {
    let count = source.len() as f32;
    let mut source_mean = Vector2::zeros();
    let mut destination_mean = Vector2::zeros();
    for index in 0..5 {
        source_mean += Vector2::new(source[index][0], source[index][1]);
        destination_mean += Vector2::new(destination[index][0], destination[index][1]);
    }
    source_mean /= count;
    destination_mean /= count;

    let mut variance = 0.0;
    let mut covariance = Matrix2::zeros();
    for index in 0..5 {
        let source_delta = Vector2::new(source[index][0], source[index][1]) - source_mean;
        let destination_delta =
            Vector2::new(destination[index][0], destination[index][1]) - destination_mean;
        variance += source_delta.dot(&source_delta);
        covariance += destination_delta * source_delta.transpose();
    }
    variance /= count;
    covariance /= count;

    let decomposition = covariance.svd(true, true);
    let left = decomposition.u.unwrap_or_else(Matrix2::identity);
    let right = decomposition.v_t.unwrap_or_else(Matrix2::identity);
    let singular = decomposition.singular_values;
    let mut reflection = Matrix2::identity();
    if covariance.determinant() < 0.0
        || (covariance.determinant() == 0.0 && left.determinant() * right.determinant() < 0.0)
    {
        reflection[(1, 1)] = -1.0;
    }
    let rotation = left * reflection * right;
    let scale = if variance == 0.0 {
        1.0
    } else {
        (singular[0] * reflection[(0, 0)] + singular[1] * reflection[(1, 1)]) / variance
    };
    let translation = destination_mean - scale * (rotation * source_mean);
    let mut result = Matrix2x3::zeros();
    result
        .fixed_view_mut::<2, 2>(0, 0)
        .copy_from(&(scale * rotation));
    result.set_column(2, &translation);
    result
}
