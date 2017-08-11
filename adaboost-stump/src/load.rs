use std::fs;
use rand::{thread_rng, Rng};
use shared::{DataPoint, ImageData, IntegralImage};
use rulinalg::matrix::{BaseMatrix, Matrix};
use image::{DynamicImage, ImageBuffer, Pixel, Rgba};

type PreprocessedImage = ImageBuffer<Rgba<u8>, Vec<u8>>;

fn get_luminosity_from_image(img: &PreprocessedImage, i: u32, j: u32) -> f64 {
    let channels = img.get_pixel(i, j).channels();
    let r = channels[0] as f64;
    let g = channels[1] as f64;
    let b = channels[2] as f64;

    (r + g + b) / 3.0
}

fn get_luminosity_matrix(img: &PreprocessedImage) -> ImageData {
    let (width, height) = img.dimensions();
    let mut mat = Matrix::zeros(height as usize, width as usize);

    for i in 0..height {
        for j in 0..width {
            let index = [i as usize, j as usize];
            mat[index] = get_luminosity_from_image(img, i, j);
        }
    }

    mat
}

fn get_integral_image(image: &ImageData) -> IntegralImage {
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

    mat
}

pub fn get_training_data() -> (Vec<DataPoint>, usize, usize) {
    let faces_paths: Vec<_> =
        fs::read_dir("./data/training/faces").unwrap().map(|path| (path, 1.0)).collect();
    let num_faces = faces_paths.len();

    let non_faces_paths: Vec<_> =
        fs::read_dir("./data/training/non-faces").unwrap().map(|path| (path, -1.0)).collect();
    let num_non_faces = non_faces_paths.len();

    let paths = faces_paths.into_iter().chain(non_faces_paths.into_iter());

    let mut training_data: Vec<DataPoint> = paths.map(|(path, label)| {
            extern crate image;

            let img_path = path.unwrap().path();

            let rgba_image = match image::open(img_path).unwrap() {
                DynamicImage::ImageRgba8(rgba_image) => rgba_image,
                _ => unreachable!(),
            };

            let image_data = get_luminosity_matrix(&rgba_image);

            let integral_image = get_integral_image(&image_data);

            DataPoint {
                image_data: image_data,
                integral_image: integral_image,
                label: label,
            }
        })
        .collect();

    {

        let slice = training_data.as_mut_slice();
        thread_rng().shuffle(slice);
    }

    (training_data, num_faces, num_non_faces)
}
