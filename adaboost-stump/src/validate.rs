#[macro_use]
extern crate serde_derive;
extern crate serde;
extern crate serde_json;
extern crate image;
extern crate rulinalg;
extern crate piston_window;

mod shared;
mod integral_image;
mod prediction_ensemble;
mod haar_like_feature;

use std::fs::File;
use std::io::Read;
use image::{ConvertBuffer, DynamicImage, Pixel};
use rulinalg::matrix::Matrix;
use integral_image::IntegralImage;
use shared::{WINDOW_HEIGHT, WINDOW_WIDTH};
use prediction_ensemble::PredictionEnsemble;
use piston_window::{PistonWindow, Texture, WindowSettings, TextureSettings};
use piston_window::rectangle::Rectangle;

fn main() {
    let data = match image::open("./data/got.jpeg").unwrap() {
        DynamicImage::ImageLuma8(gray_image) => gray_image,
        DynamicImage::ImageRgb8(rgb_image) => rgb_image.convert(),
        DynamicImage::ImageRgba8(rgba_image) => rgba_image.convert(),
        _ => unreachable!(),
    };

    let (width, height) = data.dimensions();

    let mut model_raw = String::new();
    let mut f = File::open("./foo.json").expect("Unable to open file");
    f.read_to_string(&mut model_raw).expect("Unable to read string");

    let ensemble: PredictionEnsemble = serde_json::from_str(&model_raw).unwrap();

    let mut coll = Vec::new();

    for y in 0..((height as usize) - WINDOW_HEIGHT) {
        for x in 0..((width as usize) - WINDOW_WIDTH) {
            let mut mat = Matrix::zeros(WINDOW_HEIGHT, WINDOW_WIDTH);
            let mut max = 1.0;

            for i in 0..WINDOW_HEIGHT {
                for j in 0..WINDOW_WIDTH {
                    let pixel = data.get_pixel((x + i) as u32, (y + j) as u32);
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
                coll.push((x, y, WINDOW_WIDTH, WINDOW_HEIGHT));
            }
        }
    }

    let mut window: PistonWindow = WindowSettings::new("piston: image", [width, height])
        .exit_on_esc(true)
        .build()
        .unwrap();

    let img = data.convert();

    let mut tex: Texture<_> =
        Texture::from_image(&mut window.factory, &img, &TextureSettings::new()).ok().unwrap();

    while let Some(e) = window.next() {
        window.draw_2d(&e, |c, g| {
            piston_window::clear([1.0; 4], g);

            piston_window::image(&tex, c.transform, g);

            for &(x, y, ww, wh) in coll.iter() {
                let rect = Rectangle::new_border([0.0, 1.0, 0.0, 0.2], 1.0);
                let pos = [x as f64, y as f64, ww as f64, wh as f64];
                rect.draw(pos, &c.draw_state, c.transform, g)
            }
        });
    }
}
