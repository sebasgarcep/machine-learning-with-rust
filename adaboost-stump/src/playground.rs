// FIND v4l2 COMPATIBLE SPECS: v4l2-ctl --list-formats-ext
// Thanks to https://github.com/oli-obk/camera_capture for most of the code

#[macro_use]
extern crate lazy_static;
extern crate piston_window;
extern crate image;
extern crate rulinalg;
extern crate rand;

mod load;
mod shared;

use std::iter::Iterator;
use rulinalg::vector::Vector;
use load::get_training_data;
use shared::{DataPoint, IntegralImage};

const WINDOW_HEIGHT: usize = 24;
const WINDOW_WIDTH: usize = 24;
const NUM_ROUNDS: usize = 20;

#[derive(Debug, Clone, Copy)]
enum HaarLikeFeatureType {
    TypeA, // two columns vertical
    TypeB, // two columns horizontal
    TypeC, // three columns vertical
    TypeD, // four squares checkerboard
}

#[derive(Debug, Clone, Copy)]
struct HaarLikeFeature {
    feature_type: HaarLikeFeatureType,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
}

lazy_static! {
    // generate all the possible haar-like features from the bounding boxes and the feature types
    static ref HAAR_LIKE_FEATURE_HYPOTHESES: Vec<HaarLikeFeature> = {
        let mut feature_hypotheses = Vec::new();

        for y in 0..(WINDOW_HEIGHT - 2) {
            for x in 0..(WINDOW_WIDTH - 6) {
                for fraction_height in 0..((WINDOW_HEIGHT - y) / 2) {
                    for fraction_width in 0..((WINDOW_WIDTH - x) / 6) {
                        let height = (fraction_height + 1) * 2;
                        let width = (fraction_width + 1) * 6;

                        let mut feature_hypothesis;

                        feature_hypothesis = HaarLikeFeature {
                            feature_type: HaarLikeFeatureType::TypeA,
                            x,
                            y,
                            width,
                            height,
                        };

                        feature_hypotheses.push(feature_hypothesis);

                        feature_hypothesis = HaarLikeFeature {
                            feature_type: HaarLikeFeatureType::TypeB,
                            x,
                            y,
                            width,
                            height,
                        };

                        feature_hypotheses.push(feature_hypothesis);

                        feature_hypothesis = HaarLikeFeature {
                            feature_type: HaarLikeFeatureType::TypeC,
                            x,
                            y,
                            width,
                            height,
                        };

                        feature_hypotheses.push(feature_hypothesis);

                        feature_hypothesis = HaarLikeFeature {
                            feature_type: HaarLikeFeatureType::TypeD,
                            x,
                            y,
                            width,
                            height,
                        };

                        feature_hypotheses.push(feature_hypothesis);
                    }
                }
            }
        }

        feature_hypotheses
    };
}

fn feature_box_value(integral_image: &IntegralImage,
                     x: usize,
                     y: usize,
                     width: usize,
                     height: usize)
                     -> f64 {
    // bounding box
    let upper_left = integral_image[[x, y]];
    let upper_right = integral_image[[x + width, y]];
    let bottom_left = integral_image[[x, y + height]];
    let bottom_right = integral_image[[x + width, y + height]];

    bottom_right - bottom_left - upper_right + upper_left
}

impl HaarLikeFeature {
    fn get_score_a(&self, integral_image: &IntegralImage) -> f64 {
        let left_box = feature_box_value(integral_image,
                                         self.x as usize,
                                         self.y as usize,
                                         self.width / 2 as usize,
                                         self.height as usize);

        let right_box = feature_box_value(integral_image,
                                          self.x + self.width / 2 as usize,
                                          self.y as usize,
                                          self.width / 2 as usize,
                                          self.height as usize);

        right_box - left_box
    }

    fn get_score_b(&self, integral_image: &IntegralImage) -> f64 {
        let upper_box = feature_box_value(integral_image,
                                          self.x as usize,
                                          self.y as usize,
                                          self.width as usize,
                                          self.height / 2 as usize);

        let lower_box = feature_box_value(integral_image,
                                          self.x as usize,
                                          self.y + self.height / 2 as usize,
                                          self.width as usize,
                                          self.height / 2 as usize);

        upper_box - lower_box
    }

    fn get_score_c(&self, integral_image: &IntegralImage) -> f64 {

        let left_box = feature_box_value(integral_image,
                                         self.x as usize,
                                         self.y as usize,
                                         self.width / 3 as usize,
                                         self.height as usize);

        let mid_box = feature_box_value(integral_image,
                                        self.x + self.width / 3 as usize,
                                        self.y as usize,
                                        self.width / 3 as usize,
                                        self.height as usize);

        let right_box = feature_box_value(integral_image,
                                          self.x + 2 * self.width / 3 as usize,
                                          self.y as usize,
                                          self.width / 3 as usize,
                                          self.height as usize);

        mid_box - (left_box + right_box)
    }

    fn get_score_d(&self, integral_image: &IntegralImage) -> f64 {
        let upper_left_box = feature_box_value(integral_image,
                                               self.x as usize,
                                               self.y as usize,
                                               self.width / 2 as usize,
                                               self.height / 2 as usize);

        let upper_right_box = feature_box_value(&integral_image,
                                                self.x + self.width / 2 as usize,
                                                self.y as usize,
                                                self.width / 2 as usize,
                                                self.height / 2 as usize);

        let bottom_right_box = feature_box_value(integral_image,
                                                 self.x as usize,
                                                 self.y + self.height / 2 as usize,
                                                 self.width / 2 as usize,
                                                 self.height / 2 as usize);

        let bottom_left_box = feature_box_value(&integral_image,
                                                self.x + self.width / 2 as usize,
                                                self.y + self.height / 2 as usize,
                                                self.width / 2 as usize,
                                                self.height / 2 as usize);

        (upper_left_box + bottom_right_box) - (upper_right_box + bottom_left_box)
    }

    pub fn get_score(&self, data_point: &DataPoint) -> f64 {
        match self.feature_type {
            HaarLikeFeatureType::TypeA => self.get_score_a(&data_point.integral_image),
            HaarLikeFeatureType::TypeB => self.get_score_b(&data_point.integral_image),
            HaarLikeFeatureType::TypeC => self.get_score_c(&data_point.integral_image),
            HaarLikeFeatureType::TypeD => self.get_score_d(&data_point.integral_image),
        }
    }
}

#[derive(Debug)]
struct PredictionHypothesis {
    pub haar_like_feature: HaarLikeFeature,
    pub theta: f64,
    pub b: f64,
}

impl PredictionHypothesis {
    pub fn predict(&self, data_point: &DataPoint) -> f64 {
        let x = self.haar_like_feature.get_score(data_point);

        (self.theta - x).signum() * self.b
    }
}

fn weak_learner(image_collection: &mut Vec<DataPoint>, dist: &Vector<f64>) -> PredictionHypothesis {
    let m = image_collection.len();

    let mut f_star_min = std::f64::INFINITY;
    let mut theta_star_min = std::f64::INFINITY;
    let mut feature_star_min = None;

    let mut f_star_max = -std::f64::INFINITY;
    let mut theta_star_max = std::f64::INFINITY;
    let mut feature_star_max = None;

    for feature_hypothesis in HAAR_LIKE_FEATURE_HYPOTHESES.iter() {
        // calculate <dist, (fi)i=1,2,..,m>, where fi = (yi + 1) / 2
        let mut f: f64 = 0.0;
        for (i, ref data_point) in image_collection.iter().enumerate() {
            if data_point.label > 0.0 {
                f += dist[i];
            }
        }

        let mut scores: Vec<_> = image_collection.iter()
            .map(|ref data_point| feature_hypothesis.get_score(&data_point))
            .collect();
        scores.sort_by(|&a, &b| a.partial_cmp(&b).unwrap());

        if f < f_star_min {
            f_star_min = f;
            theta_star_min = scores[0] - 1.0;
            feature_star_min = Some(feature_hypothesis);
        }

        if f > f_star_max {
            f_star_max = f;
            theta_star_max = scores[0] - 1.0;
            feature_star_max = Some(feature_hypothesis);
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
                    feature_star_min = Some(feature_hypothesis);
                }

                if f > f_star_max {
                    f_star_max = f;
                    theta_star_max = 0.5 * (curr_x + next_x);
                    feature_star_max = Some(feature_hypothesis);
                }
            }
        }
    }

    if f_star_min < (1.0 - f_star_max) {
        PredictionHypothesis {
            haar_like_feature: *(feature_star_min.unwrap()),
            theta: theta_star_min,
            b: 1.0,
        }
    } else {
        PredictionHypothesis {
            haar_like_feature: *(feature_star_max.unwrap()),
            theta: theta_star_max,
            b: -1.0,
        }
    }

}

fn adaboost(mut image_collection: &mut Vec<DataPoint>,
            num_faces: usize,
            num_non_faces: usize)
            -> Vec<(f64, PredictionHypothesis)> {
    let m = image_collection.len();

    let weight_array: Vec<f64> = image_collection.iter()
        .map(|data_point| {
            if data_point.label > 0.0 {
                1.0 / ((2 * num_faces) as f64)
            } else {
                1.0 / ((2 * num_non_faces) as f64)
            }
        })
        .collect();

    let mut dist = Vector::new(weight_array);
    let first_sum = dist.sum();

    dist = dist / first_sum;

    let mut ensemble = Vec::new();

    for t in 0..NUM_ROUNDS {
        println!("Begun round: {}", t + 1);

        let h = weak_learner(image_collection, &dist);

        let label_prediction_tuples: Vec<_> = image_collection.iter()
            .map(|data_point| {
                let label = data_point.label;
                let prediction = h.predict(data_point);

                (label, prediction)
            })
            .collect();

        let (e, bad_predictions) = label_prediction_tuples.iter()
            .enumerate()
            .fold((0.0, 0), |(acc_err, acc_bac_predictions), value| {
                let (i, &(label, prediction)) = value;

                if label * prediction < 0.0 {
                    (acc_err + dist[i], acc_bac_predictions + 1)
                } else {
                    (acc_err, acc_bac_predictions)
                }
            });

        println!("Bad predictions: {} / {}", bad_predictions, m);

        let w = 0.5 * (1.0 / e - 1.0).ln();

        println!("w({}) = {:?}", t + 1, w);
        println!("h({}) = {:?}", t + 1, h);

        ensemble.push((w, h));

        for (i, &(label, prediction)) in label_prediction_tuples.iter().enumerate() {
            dist[i] = dist[i] * (-w * label * prediction).exp();
        }

        let sum = dist.sum();
        dist = dist / sum;

        println!("Finished round: {}", t + 1);
    }

    ensemble
}

fn main() {
    let (mut image_collection, num_faces, num_non_faces) = get_training_data();

    let ensemble = adaboost(&mut image_collection, num_faces, num_non_faces);

    let mut bad_predictions = 0;

    for data_point in image_collection.iter() {
        let mut prediction = 0.0;

        for &(w, ref h) in ensemble.iter() {
            prediction += w * h.predict(&data_point);
        }

        prediction = prediction.signum();

        if prediction * data_point.label < 0.0 {
            bad_predictions += 1;
        }
    }

    println!("Ensemble bad predictions: {} / {}",
             bad_predictions,
             image_collection.len());
}
