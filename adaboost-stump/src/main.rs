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
mod prediction_ensemble;

use std::iter::Iterator;
use rulinalg::vector::Vector;
use load::get_training_data;
use haar_like_feature::HaarLikeFeature;
use prediction_ensemble::PredictionEnsemble;
use shared::{DataPoint, NUM_ROUNDS};

fn weak_learner(feature_collection: &mut Vec<HaarLikeFeature>,
                image_collection: &Vec<DataPoint>,
                weights: &Vector<f64>)
                -> HaarLikeFeature {
    let mut error_star = std::f64::INFINITY;
    let mut fi_star = None;
    let mut threshold_star = None;

    for fi in 0..(feature_collection.len()) {
        let ref mut feature_hypothesis = feature_collection[fi];

        let mut scores: Vec<_> = image_collection.iter()
            .enumerate()
            .map(|(index, ref data_point)| (index, feature_hypothesis.get_score(&data_point)))
            .collect();

        scores.sort_by(|&a, &b| a.partial_cmp(&b).unwrap());

        let mut error = image_collection.iter()
            .zip(weights.iter())
            .fold(0.0, |acc, (data_point, weight)| {
                if data_point.label * feature_hypothesis.polarity < 0.0 {
                    acc + weight
                } else {
                    acc
                }
            });

        let m = scores.len();
        for xi in 0..(m + 1) {
            let curr_x = if xi > 0 {
                scores[xi - 1].1
            } else {
                scores[xi].1 - 1.0
            };

            let next_x = if xi < m {
                scores[xi].1
            } else {
                scores[xi - 1].1 + 1.0
            };

            let threshold = (curr_x + next_x) / 2.0;
            feature_hypothesis.threshold = threshold;

            if error < error_star {
                error_star = error;
                fi_star = Some(fi);
                threshold_star = Some(threshold);
            }

            if xi < m {
                // computes the error of the next iteration
                let ref data_point = image_collection[xi];
                let weight = weights[xi];
                error += feature_hypothesis.polarity * data_point.label * weight;
            } else if error < error_star {
                // check after all error updates
                error_star = error;
                fi_star = Some(fi);
                threshold_star = Some(threshold);
            }
        }
    }

    let mut feature = feature_collection.remove(fi_star.unwrap());
    feature.threshold = threshold_star.unwrap();

    feature
}

fn adaboost(image_collection: &Vec<DataPoint>,
            num_faces: usize,
            num_non_faces: usize)
            -> PredictionEnsemble {
    let mut weights: Vector<f64> = image_collection.iter()
        .map(|data_point| {
            if data_point.label > 0.0 {
                1.0 / ((2 * num_faces) as f64)
            } else {
                1.0 / ((2 * num_non_faces) as f64)
            }
        })
        .collect();

    // generate all the possible haar-like features from the bounding boxes and the feature types
    let mut feature_collection = HaarLikeFeature::generate_all_features();

    let mut ensemble = PredictionEnsemble::new();

    for t in 0..NUM_ROUNDS {
        println!("Begun round: {}", t + 1);

        // normalize weights
        let sum = weights.sum();
        weights = weights / sum;

        let mut h = weak_learner(&mut feature_collection, image_collection, &weights);

        let label_prediction_tuples: Vec<_> = image_collection.iter()
            .map(|data_point| (data_point.label, h.predict(data_point)))
            .collect();

        let epsilon = label_prediction_tuples.iter()
            .zip(weights.iter())
            .fold(0.0, |acc, (&(label, prediction), weight)| {
                if label * prediction < 0.0 {
                    acc + weight
                } else {
                    acc
                }
            });

        h.weight = 0.5 * ((1.0 - epsilon) / epsilon).ln();

        println!("h({}) = {:?}", t + 1, h);

        ensemble.push(h);

        /*
        weights = label_prediction_tuples.iter()
            .zip(weights.iter())
            .map(|(&(label, prediction), weight)| weight * (-w * label * prediction).exp())
            .collect();
        */

        weights = label_prediction_tuples.iter()
            .zip(weights.iter())
            .map(|(&(label, prediction), &weight)| {
                if label * prediction > 0.0 {
                    weight * (epsilon / (1.0 - epsilon)).sqrt()
                } else {
                    weight * ((1.0 - epsilon) / epsilon).sqrt()
                }
            })
            .collect();

        println!("Finished round: {}", t + 1);
        ensemble.print_mistakes(&image_collection);
    }

    ensemble
}

fn main() {
    let (image_collection, num_faces, num_non_faces) = get_training_data();
    adaboost(&image_collection, num_faces, num_non_faces);
}
