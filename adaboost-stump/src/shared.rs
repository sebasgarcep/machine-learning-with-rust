use rulinalg::matrix::Matrix;
use integral_image::IntegralImage;

pub const WINDOW_HEIGHT: usize = 19;
pub const WINDOW_WIDTH: usize = 19;
pub const MIN_FEATURE_HEIGHT: usize = 4;
pub const MIN_FEATURE_WIDTH: usize = 4;

pub type ImageData = Matrix<f64>;
pub type Label = f64;

#[derive(Debug)]
pub struct DataPoint {
    pub image_data: ImageData,
    pub integral_image: IntegralImage,
    pub label: Label,
}
