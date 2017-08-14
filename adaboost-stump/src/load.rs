use std::fs;
use rand::{thread_rng, Rng};
use shared::{DataPoint, ImageData};
use rulinalg::matrix::Matrix;
use image::{DynamicImage, Pixel, GrayImage};
use integral_image::IntegralImage;

type PreprocessedImage = GrayImage;

fn get_luminosity_from_image(img: &PreprocessedImage, i: u32, j: u32) -> f64 {
    let channels = img.get_pixel(i, j).channels();

    channels[0] as f64
}

fn get_luminosity_matrix(img: &PreprocessedImage) -> ImageData {
    let (width, height) = img.dimensions();
    let mut mat = Matrix::zeros(height as usize, width as usize);

    let mut max = 1.0;

    for i in 0..height {
        for j in 0..width {
            let index = [i as usize, j as usize];
            let value = get_luminosity_from_image(img, i, j);

            mat[index] = value;

            if value > max {
                max = value;
            }
        }
    }

    for i in 0..height {
        for j in 0..width {
            let index = [i as usize, j as usize];
            let value = mat[index] / max;

            mat[index] = value;
        }
    }

    mat
}

pub fn get_training_data() -> (Vec<DataPoint>, usize, usize) {
    let faces_paths: Vec<_> =
        fs::read_dir("./data/trainset/faces").unwrap().map(|path| (path, 1.0)).collect();
    let num_faces = faces_paths.len();

    let non_faces_paths: Vec<_> =
        fs::read_dir("./data/trainset/non-faces").unwrap().map(|path| (path, -1.0)).collect();
    let num_non_faces = non_faces_paths.len();

    let paths = faces_paths.into_iter().chain(non_faces_paths.into_iter());

    let mut training_data: Vec<DataPoint> = paths.map(|(path, label)| {
            extern crate image;

            let img_path = path.unwrap().path();

            let gray_image = match image::open(img_path.clone()).unwrap() {
                DynamicImage::ImageLuma8(gray_image) => gray_image,
                _ => unreachable!(),
            };

            let image_data = get_luminosity_matrix(&gray_image);

            let integral_image = IntegralImage::build(&image_data);

            DataPoint {
                image_data: image_data,
                integral_image: integral_image,
                label: label,
            }
        })
        .collect();

    // shuffle data to introduce randomness
    {

        let slice = training_data.as_mut_slice();
        thread_rng().shuffle(slice);
    }

    (training_data, num_faces, num_non_faces)
}
