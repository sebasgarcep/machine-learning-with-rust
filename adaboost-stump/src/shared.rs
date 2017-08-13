use rulinalg::matrix::Matrix;
use integral_image::IntegralImage;

pub const WINDOW_HEIGHT: usize = 24;
pub const WINDOW_WIDTH: usize = 24;
pub const MIN_FEATURE_HEIGHT: usize = 8;
pub const MIN_FEATURE_WIDTH: usize = 8;
pub const NUM_ROUNDS: usize = 50;

pub type ImageData = Matrix<f64>;
pub type Label = f64;

pub struct DataPoint {
    pub image_data: ImageData,
    pub integral_image: IntegralImage,
    pub label: Label,
}
