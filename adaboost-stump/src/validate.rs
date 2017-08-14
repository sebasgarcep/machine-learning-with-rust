#[macro_use]
extern crate serde_derive;
extern crate serde;
extern crate serde_json;
extern crate image;
extern crate rulinalg;

mod shared;
mod integral_image;
mod prediction_ensemble;
mod haar_like_feature;

use std::fs::File;
use std::io::Read;
use image::{DynamicImage, Pixel};
use rulinalg::matrix::Matrix;
use integral_image::IntegralImage;
use shared::{WINDOW_HEIGHT, WINDOW_WIDTH};
use prediction_ensemble::PredictionEnsemble;

fn main() {
    let beatles = match image::open("./data/beatles.jpg").unwrap() {
        DynamicImage::ImageLuma8(gray_image) => gray_image,
        _ => unreachable!(),
    };

    let (width, height) = beatles.dimensions();

    let mut data = String::new();
    let mut f = File::open("./foo.json").expect("Unable to open file");
    f.read_to_string(&mut data).expect("Unable to read string");

    let ensemble: PredictionEnsemble = serde_json::from_str(&data).unwrap();

    for y in 0..((height as usize) - WINDOW_HEIGHT) {
        for x in 0..((width as usize) - WINDOW_WIDTH) {
            let mut mat = Matrix::zeros(WINDOW_HEIGHT, WINDOW_WIDTH);
            let mut max = 1.0;

            for i in 0..WINDOW_HEIGHT {
                for j in 0..WINDOW_WIDTH {
                    let pixel = beatles.get_pixel((x + i) as u32, (y + j) as u32);
                    let mat_index = [i as usize, j as usize];
                    let value = (pixel.channels()[0]) as f64;
                    mat[mat_index] = value;

                    if value > max {
                        max = value;
                    }
                }
            }

            mat = mat / max;

            let integral_image = IntegralImage::build(&mat);

            if ensemble.predict(&integral_image) {
                println!("x: {:?} y: {:?}", x, y);
            }
        }
    }
}
