use rulinalg::matrix::Matrix;

pub type ImageData = Matrix<f64>;
pub type IntegralImage = Matrix<f64>;
pub type Label = f64;

pub struct DataPoint {
    pub image_data: ImageData,
    pub integral_image: IntegralImage,
    pub label: Label,
}
