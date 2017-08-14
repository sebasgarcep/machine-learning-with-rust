use rulinalg::matrix::{BaseMatrix, Matrix};
use shared::ImageData;

#[derive(Debug)]
pub struct IntegralImage {
    data: Matrix<f64>,
}

impl IntegralImage {
    pub fn build(image: &ImageData) -> IntegralImage {
        let width = image.cols() + 1;
        let height = image.rows() + 1;
        let mut mat = Matrix::zeros(height as usize, width as usize);

        for i in 0..height {
            for j in 0..width {
                if i == 0 || j == 0 {
                    continue;
                }

                let top_term = if i > 1 {
                    let top_index = [(i - 1) as usize, j as usize];
                    mat[top_index]
                } else {
                    0.0
                };

                let left_term = if j > 1 {
                    let left_index = [i as usize, (j - 1) as usize];
                    mat[left_index]
                } else {
                    0.0
                };

                let diag_term = if i > 1 && j > 1 {
                    let diag_index = [(i - 1) as usize, (j - 1) as usize];
                    mat[diag_index]
                } else {
                    0.0
                };

                let img_index = [(i - 1) as usize, (j - 1) as usize];
                let index = [i as usize, j as usize];
                mat[index] = image[img_index] + top_term + left_term - diag_term;
            }
        }

        IntegralImage { data: mat }
    }

    pub fn sum_region(&self, x: usize, y: usize, width: usize, height: usize) -> f64 {
        // bounding box
        let upper_left = self.data[[x, y]];
        let upper_right = self.data[[x + width, y]];
        let bottom_left = self.data[[x, y + height]];
        let bottom_right = self.data[[x + width, y + height]];

        bottom_right - bottom_left - upper_right + upper_left
    }
}
