// FIND v4l2 COMPATIBLE SPECS: v4l2-ctl --list-formats-ext
// Thanks to https://github.com/oli-obk/camera_capture for most of the code

extern crate piston_window;
extern crate image;
extern crate rulinalg;

use std::thread::sleep;
use std::time::Duration;
use std::collections::HashMap;
use piston_window::{PistonWindow, Texture, WindowSettings, TextureSettings, clear};
use image::{ConvertBuffer, DynamicImage, ImageBuffer, Rgba};

use std::io::BufReader;
use std::io::BufRead;
use std::fs;
use std::fs::File;
use std::io::Read;
use rulinalg::matrix::Matrix;
use rulinalg::vector::Vector;
use image::Pixel;

const WINDOW_SIZE: u64 = 24;
const NUM_ROUNDS: usize = 10;

type ImageData = ImageBuffer<Rgba<u8>, Vec<u8>>;
type IntegralImage = Matrix<i64>;
type Label = i64;

#[derive(Debug)]
struct FaceFeature {
    left_eye: (i64, i64),
    right_eye: (i64, i64),
    nose: (i64, i64),
    left_mouth: (i64, i64),
    center_mouth: (i64, i64),
    right_mouth: (i64, i64),
}

#[allow(dead_code)]
fn get_u64_from_img(img: &ImageData, i: u32, j: u32) -> i64 {
    let channels = img.get_pixel(i, j).channels();
    let r = channels[0] as i64;
    let g = channels[1] as i64;
    let b = channels[2] as i64;

    (r + g + b) / 3
}

#[allow(dead_code)]
fn get_integral_img(img: &ImageData) -> IntegralImage {
    let (width, height) = img.dimensions();
    let num_entries = width * height;

    let mut mat = Matrix::new(width as usize,
                              height as usize,
                              vec![0; num_entries as usize]);

    for i in 0..width {
        for j in 0..height {
            let index: [usize; 2];
            let left_index: [usize; 2];
            let top_index: [usize; 2];
            let diag_index: [usize; 2];

            index = [i as usize, j as usize];

            if i == 0 && j == 0 {
                mat[index] = get_u64_from_img(img, i, j);
            } else if i == 0 {
                left_index = [i as usize, (j - 1) as usize];
                mat[index] = mat[left_index] + get_u64_from_img(img, i, j);
            } else if j == 0 {
                top_index = [(i - 1) as usize, j as usize];
                mat[index] = mat[top_index] + get_u64_from_img(img, i, j);
            } else {
                left_index = [i as usize, (j - 1) as usize];
                top_index = [(i - 1) as usize, j as usize];
                diag_index = [(i - 1) as usize, (j - 1) as usize];
                mat[index] = mat[diag_index] + (mat[top_index] - mat[diag_index]) +
                             (mat[left_index] - mat[diag_index]) +
                             get_u64_from_img(img, i, j);
            }
        }
    }

    mat
}

#[allow(dead_code)]
fn get_faces_map() -> HashMap<String, Vec<FaceFeature>> {
    let mut faces_map = HashMap::new();
    let faces_file = File::open("./data/faces.txt").unwrap();
    let faces_buffer = BufReader::new(&faces_file);
    for line in faces_buffer.lines() {
        let l = line.unwrap();
        let tokens: Vec<&str> = l.split(" ").collect();
        let filename = tokens[0];

        let coords: Vec<i64> = tokens[1..]
            .iter()
            .map(|s| {
                let parts: Vec<&str> = s.split(".").collect();
                let integer_part = parts[0];
                integer_part.parse().unwrap()
            })
            .collect();

        let feature = FaceFeature {
            left_eye: (coords[0], coords[1]),
            right_eye: (coords[2], coords[3]),
            nose: (coords[4], coords[5]),
            left_mouth: (coords[6], coords[7]),
            center_mouth: (coords[8], coords[9]),
            right_mouth: (coords[10], coords[11]),
        };

        let keyname = filename.to_string();
        let feature_list = faces_map.entry(keyname).or_insert(Vec::new());

        feature_list.push(feature);
    }

    faces_map
}

fn feature_box_value(integral_img: &IntegralImage,
                     ox: usize,
                     oy: usize,
                     fx: usize,
                     fy: usize)
                     -> i64 {
    // bounding box
    let upper_left = integral_img[[ox, oy]];
    let upper_right = integral_img[[fx, oy]];
    let bottom_left = integral_img[[ox, fy]];
    let bottom_right = integral_img[[fx, fy]];

    bottom_right - bottom_left - upper_right + upper_left
}

fn compute_feature_a(integral_img: &IntegralImage, ox: u32, oy: u32, fx: u32, fy: u32) -> i64 {
    let left_box = feature_box_value(integral_img,
                                     ox as usize,
                                     oy as usize,
                                     ((ox + fx) / 2) as usize,
                                     fy as usize);
    let right_box = feature_box_value(integral_img,
                                      ((ox + fx) / 2) as usize,
                                      oy as usize,
                                      fx as usize,
                                      fy as usize);

    right_box - left_box
}

fn compute_feature_b(integral_img: &IntegralImage, ox: u32, oy: u32, fx: u32, fy: u32) -> i64 {
    let upper_box = feature_box_value(integral_img,
                                      ox as usize,
                                      oy as usize,
                                      fx as usize,
                                      ((oy + fy) / 2) as usize);
    let lower_box = feature_box_value(integral_img,
                                      ox as usize,
                                      ((oy + fy) / 2) as usize,
                                      fx as usize,
                                      fy as usize);

    upper_box - lower_box
}

fn compute_feature_c(integral_img: &IntegralImage, ox: u32, oy: u32, fx: u32, fy: u32) -> i64 {
    let full_box = feature_box_value(integral_img,
                                     ox as usize,
                                     oy as usize,
                                     fx as usize,
                                     fy as usize);
    let mid_box = feature_box_value(integral_img,
                                    ((ox + fx) / 3) as usize,
                                    oy as usize,
                                    ((ox + fx) * 2 / 3) as usize,
                                    fy as usize);

    2 * mid_box - full_box
}

fn compute_feature_d(integral_img: &IntegralImage, ox: u32, oy: u32, fx: u32, fy: u32) -> i64 {
    let full_box = feature_box_value(integral_img,
                                     ox as usize,
                                     oy as usize,
                                     fx as usize,
                                     fy as usize);
    let upper_left_box = feature_box_value(integral_img,
                                           ox as usize,
                                           oy as usize,
                                           ((ox + fx) / 2) as usize,
                                           ((oy + fy) / 2) as usize);
    let bottom_right_box = feature_box_value(integral_img,
                                             ((ox + fx) / 2) as usize,
                                             ((oy + fy) / 2) as usize,
                                             fx as usize,
                                             fy as usize);

    2 * (upper_left_box + bottom_right_box) - full_box
}

fn adaboost(img_collection: Vec<(ImageData, IntegralImage, Label)>) {
    let m = img_collection.len();
    let mut d = Vector::new(vec![1 / m; m]);

    for t in 0..NUM_ROUNDS {
        // let e = d.dot();
        // let w = 0.5 * (1 / e - 1).ln();

        // for i in 0..m {
        //     d[i] = d[i] * (-w * img_collection[i].1).exp();
        // }

        // d = d / d.sum();
    }
}

fn main() {
    let faces_paths = fs::read_dir("./data/training/faces").unwrap().map(|path| (path, 1));
    let non_faces_paths = fs::read_dir("./data/training/non-faces").unwrap().map(|path| (path, -1));
    let paths = faces_paths.chain(non_faces_paths);
    let img_collection: Vec<_> = paths.map(|(path, label)| {
            let img_path = path.unwrap().path();

            let rgba_image = match image::open(img_path).unwrap() {
                DynamicImage::ImageRgba8(rgba_image) => rgba_image,
                _ => unreachable!(),
            };

            let integral_img = get_integral_img(&rgba_image);

            (rgba_image, integral_img, label)
        })
        .collect();

    adaboost(img_collection);
}
