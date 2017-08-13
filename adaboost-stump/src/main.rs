// FIND v4l2 COMPATIBLE SPECS: v4l2-ctl --list-formats-ext
// Thanks to https://github.com/oli-obk/camera_capture for most of the code

extern crate piston_window;
extern crate image;
extern crate rulinalg;
extern crate rand;

mod load;
mod shared;
mod integral_image;
mod haar_like_feature;
mod prediction_hypothesis;

use std::iter::Iterator;
use rulinalg::vector::Vector;
use load::get_training_data;
use integral_image::IntegralImage;
use haar_like_feature::HaarLikeFeature;
use prediction_hypothesis::{PredictionHypothesis, PredictionEnsemble};
use shared::{DataPoint, NUM_ROUNDS};

fn weak_learner(feature_collection: &mut Vec<HaarLikeFeature>, image_collection: &mut Vec<DataPoint>, dist: &Vector<f64>) -> PredictionHypothesis {
    let m = image_collection.len();

    let mut f_star_min = std::f64::INFINITY;
    let mut theta_star_min = std::f64::INFINITY;
    let mut fi_star_min = None;

    let mut f_star_max = -std::f64::INFINITY;
    let mut theta_star_max = std::f64::INFINITY;
    let mut fi_star_max = None;

    for fi in 0..(feature_collection.len()) {
        // calculate <dist, (fi)i=1,2,..,m>, where fi = (yi + 1) / 2
        let mut f: f64 = 0.0;
        for (i, ref data_point) in image_collection.iter().enumerate() {
            if data_point.label > 0.0 {
                f += dist[i];
            }
        }

        let mut scores: Vec<_>;
        {
            let feature_hypothesis = &feature_collection[fi];

            scores = image_collection.iter()
                .map(|ref data_point| feature_hypothesis.get_score(&data_point))
                .collect();
        }

        scores.sort_by(|&a, &b| a.partial_cmp(&b).unwrap());

        if f < f_star_min {
            f_star_min = f;
            theta_star_min = scores[0] - 1.0;
            fi_star_min = Some(fi);
        }

        if f > f_star_max {
            f_star_max = f;
            theta_star_max = scores[0] - 1.0;
            fi_star_max = Some(fi);
        }

        for (i, ref data_point) in image_collection.iter().enumerate() {
            f = f - data_point.label * dist[i];
            let curr_x = scores[i];
            let next_x = if i < m - 1 {
                scores[i + 1]
            } else {
                scores[i] + 1.0
            };

            if curr_x != next_x {
                if f < f_star_min {
                    f_star_min = f;
                    theta_star_min = 0.5 * (curr_x + next_x);
                    fi_star_min = Some(fi);
                }

                if f > f_star_max {
                    f_star_max = f;
                    theta_star_max = 0.5 * (curr_x + next_x);
                    fi_star_max = Some(fi);
                }
            }
        }
    }

    if f_star_min < (1.0 - f_star_max) {
        let feature_star_min = feature_collection.remove(fi_star_min.unwrap());

        PredictionHypothesis {
            haar_like_feature: feature_star_min,
            theta: theta_star_min,
            b: 1.0,
        }
    } else {
        let feature_star_max = feature_collection.remove(fi_star_max.unwrap());

        PredictionHypothesis {
            haar_like_feature: feature_star_max,
            theta: theta_star_max,
            b: -1.0,
        }
    }

}

fn adaboost(mut image_collection: &mut Vec<DataPoint>,
            num_faces: usize,
            num_non_faces: usize)
            -> PredictionEnsemble {
    let weight_array: Vec<_> = image_collection.iter()
        .map(|data_point| {
            if data_point.label > 0.0 {
                1.0 / ((2 * num_faces) as f64)
            } else {
                1.0 / ((2 * num_non_faces) as f64)
            }
        })
        .collect();

    let mut dist = Vector::new(weight_array);

    // generate all the possible haar-like features from the bounding boxes and the feature types
    let mut feature_collection = HaarLikeFeature::generate_all_features();

    let mut ensemble = PredictionEnsemble::new();

    for t in 0..NUM_ROUNDS {
        println!("Begun round: {}", t + 1);

        // normalize weights
        let sum = dist.sum();
        dist = dist / sum;

        let h = weak_learner(&mut feature_collection, image_collection, &dist);

        let label_prediction_tuples: Vec<_> = image_collection.iter()
            .map(|data_point| {
                let label = data_point.label;
                let prediction = h.predict(data_point);

                (label, prediction)
            })
            .collect();

        let e = label_prediction_tuples.iter()
            .enumerate()
            .fold(0.0, |acc_err, value| {
                let (i, &(label, prediction)) = value;

                if label * prediction < 0.0 {
                    acc_err + dist[i]
                } else {
                    acc_err
                }
            });

        let w = 0.5 * (1.0 / e - 1.0).ln();

        println!("w({}) = {:?}", t + 1, w);
        println!("h({}) = {:?}", t + 1, h);

        ensemble.push((w, h));

        for (i, &(label, prediction)) in label_prediction_tuples.iter().enumerate() {
            dist[i] = dist[i] * (-w * label * prediction).exp();
        }

        println!("Finished round: {}", t + 1);
        ensemble.print_mistakes(&image_collection);
    }

    ensemble
}

fn main() {
    let (mut image_collection, num_faces, num_non_faces) = get_training_data();
    adaboost(&mut image_collection, num_faces, num_non_faces);
}
