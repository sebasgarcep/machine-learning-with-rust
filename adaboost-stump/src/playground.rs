// FIND v4l2 COMPATIBLE SPECS: v4l2-ctl --list-formats-ext
// Thanks to https://github.com/oli-obk/camera_capture for most of the code

#[macro_use]
extern crate lazy_static;
extern crate piston_window;
extern crate image;
extern crate rulinalg;

mod load;
mod shared;

use std::iter::Iterator;
use rulinalg::vector::Vector;
use load::get_training_data;
use shared::{DataPoint, IntegralImage};

const WINDOW_HEIGHT: usize = 24;
const WINDOW_WIDTH: usize = 24;
const NUM_ROUNDS: usize = 10;

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
    top_left_x: usize,
    top_left_y: usize,
    bottom_right_x: usize,
    bottom_right_y: usize,
}

lazy_static! {
    // generate all the possible haar-like features from the bounding boxes and the feature types
    static ref HAAR_LIKE_FEATURE_HYPOTHESES: Vec<HaarLikeFeature> = {
        let mut feature_hypotheses = Vec::new();

        for i in 0..WINDOW_HEIGHT {
            for j in 0..WINDOW_WIDTH {
                for y in (i + 1)..WINDOW_HEIGHT {
                    for x in (j + 1)..WINDOW_WIDTH {
                        let top_left_x = j;
                        let top_left_y = i;
                        let bottom_right_x = x;
                        let bottom_right_y = y;

                        let mut feature_hypothesis;

                        feature_hypothesis = HaarLikeFeature {
                            feature_type: HaarLikeFeatureType::TypeA,
                            top_left_x,
                            top_left_y,
                            bottom_right_x,
                            bottom_right_y,
                        };

                        feature_hypotheses.push(feature_hypothesis);

                        feature_hypothesis = HaarLikeFeature {
                            feature_type: HaarLikeFeatureType::TypeB,
                            top_left_x,
                            top_left_y,
                            bottom_right_x,
                            bottom_right_y,
                        };

                        feature_hypotheses.push(feature_hypothesis);

                        feature_hypothesis = HaarLikeFeature {
                            feature_type: HaarLikeFeatureType::TypeC,
                            top_left_x,
                            top_left_y,
                            bottom_right_x,
                            bottom_right_y,
                        };

                        feature_hypotheses.push(feature_hypothesis);

                        feature_hypothesis = HaarLikeFeature {
                            feature_type: HaarLikeFeatureType::TypeD,
                            top_left_x,
                            top_left_y,
                            bottom_right_x,
                            bottom_right_y,
                        };

                        feature_hypotheses.push(feature_hypothesis);
                    }
                }
            }
        }

        feature_hypotheses
    };
}

impl HaarLikeFeature {
    // FIXME
    fn feature_box_value (integral_image: &IntegralImage, ox: usize, oy: usize, fx: usize, fy: usize) -> f64 {
        // bounding box
        let upper_left = integral_image[[ox, oy]];
        let upper_right = integral_image[[(fx + 1), oy]];
        let bottom_left = integral_image[[ox, (fy + 1)]];
        let bottom_right = integral_image[[(fx + 1), (fy + 1)]];

        bottom_right - bottom_left - upper_right + upper_left
    }

    fn get_score_a (&self, integral_image: &IntegralImage) -> f64 {
        let left_box = HaarLikeFeature::feature_box_value(integral_image,
                                         self.top_left_x as usize,
                                         self.top_left_y as usize,
                                         ((self.top_left_x + self.bottom_right_x) / 2) as usize,
                                         self.bottom_right_y as usize);
        let right_box = HaarLikeFeature::feature_box_value(integral_image,
                                          ((self.top_left_x + self.bottom_right_x) / 2) as usize,
                                          self.top_left_y as usize,
                                          self.bottom_right_x as usize,
                                          self.bottom_right_y as usize);

        right_box - left_box
    }

    fn get_score_b (&self, integral_image: &IntegralImage) -> f64 {
        let upper_box = HaarLikeFeature::feature_box_value(integral_image,
                                          self.top_left_x as usize,
                                          self.top_left_y as usize,
                                          self.bottom_right_x as usize,
                                          ((self.top_left_y + self.bottom_right_y) / 2) as usize);
        let lower_box = HaarLikeFeature::feature_box_value(integral_image,
                                          self.top_left_x as usize,
                                          ((self.top_left_y + self.bottom_right_y) / 2) as usize,
                                          self.bottom_right_x as usize,
                                          self.bottom_right_y as usize);

        upper_box - lower_box
    }

    fn get_score_c (&self, integral_image: &IntegralImage) -> f64 {
        let full_box = HaarLikeFeature::feature_box_value(integral_image,
                                         self.top_left_x as usize,
                                         self.top_left_y as usize,
                                         self.bottom_right_x as usize,
                                         self.bottom_right_y as usize);

        let mid_box = HaarLikeFeature::feature_box_value(integral_image,
                                        (self.top_left_x * 2 / 3 + self.bottom_right_x * 1 / 3) as usize,
                                        self.top_left_y as usize,
                                        (self.top_left_x * 2 / 3 + self.bottom_right_x * 1 / 3) as usize,
                                        self.bottom_right_y as usize);

        2.0 * mid_box - full_box
    }

    fn get_score_d (&self, integral_image: &IntegralImage) -> f64 {
        let full_box = HaarLikeFeature::feature_box_value(integral_image,
                                         self.top_left_x as usize,
                                         self.top_left_y as usize,
                                         self.bottom_right_x as usize,
                                         self.bottom_right_y as usize);
        let upper_left_box = HaarLikeFeature::feature_box_value(integral_image,
                                                self.top_left_x as usize,
                                                self.top_left_y as usize,
                                               ((self.top_left_x + self.bottom_right_x) / 2) as usize,
                                               ((self.top_left_y + self.bottom_right_y) / 2) as usize);
        let bottom_right_box = HaarLikeFeature::feature_box_value(integral_image,
                                                ((self.top_left_x + self.bottom_right_x) / 2) as usize,
                                                ((self.top_left_y + self.bottom_right_y) / 2) as usize,
                                                 self.bottom_right_x as usize,
                                                 self.bottom_right_y as usize);

        2.0 * (upper_left_box + bottom_right_box) - full_box
    }

    pub fn get_score (&self, data_point: &DataPoint) -> f64 {
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
}

impl PredictionHypothesis {
    pub fn predict(&self, data_point: &DataPoint) -> f64 {
        let x = self.haar_like_feature.get_score(data_point);

        (self.theta - x).signum()
    }
}

fn weak_learner(image_collection: &mut Vec<DataPoint>, dist: &Vector<f64>) -> PredictionHypothesis {
    let m = image_collection.len();

    let mut f_star = std::f64::INFINITY;
    let mut theta_star = std::f64::INFINITY;
    let mut feature_star = None;

    for feature_hypothesis in HAAR_LIKE_FEATURE_HYPOTHESES.iter() {
        // calculate <dist, (fi)i=1,2,..,m>, where fi = (yi + 1) / 2
        let mut f: f64 = 0.0;
        for (i, ref data_point) in image_collection.iter().enumerate() {
            if data_point.label == 1.0 {
                f += dist[i];
            }
        }

        let mut scores: Vec<_> = image_collection
            .iter()
            .map(|ref data_point| feature_hypothesis.get_score(&data_point))
            .collect();
        scores.sort_by(|&a, &b| a.partial_cmp(&b).unwrap());

        if f < f_star {
            f_star = f;
            theta_star = scores[0] - 1.0;
            feature_star = Some(feature_hypothesis);
        }

        for (i, ref data_point) in image_collection.iter().enumerate() {
            f = f - data_point.label * dist[i];
            let curr_x = scores[i];
            let next_x = if i < m - 1 {
                scores[i + 1]
            } else {
                scores[i] + 1.0
            };

            if f < f_star && curr_x != next_x {
                f_star = f;
                theta_star = 0.5 * (curr_x + next_x);
                feature_star = Some(feature_hypothesis);
            }
        }
    }


    PredictionHypothesis {
        haar_like_feature: *(feature_star.unwrap()),
        theta: theta_star,
    }
}

struct HypothesesEnsemble {
    ensemble: Vec<(f64, PredictionHypothesis)>,
}

fn adaboost(mut image_collection: Vec<DataPoint>) -> HypothesesEnsemble {
    let m = image_collection.len();
    let mut dist = Vector::new(vec![1.0 / (m as f64); m]);

    let mut hypotheses = HypothesesEnsemble {
        ensemble: Vec::new(),
    };

    for t in 0..NUM_ROUNDS {
        println!("Begun round: {}", t + 1);

        let h = weak_learner(&mut image_collection, &dist);

        let mut e = 0.0;

        let label_prediction_tuples: Vec<_> = image_collection.iter()
            .map(|data_point| {
                let label = data_point.label;
                let prediction = h.predict(data_point);

                (label, prediction)
            })
            .collect();

        for (i, &(label, prediction)) in label_prediction_tuples.iter().enumerate() {
            if label != prediction {
                e += dist[i];
            }
        }

        let w = 0.5 * (1.0 / e - 1.0).ln();

        hypotheses.ensemble.push((w, h));

        for (i, &(label, prediction)) in label_prediction_tuples.iter().enumerate() {
            dist[i] = dist[i] * (-w * label * prediction).exp();
        }

        let sum = dist.sum();
        dist = dist / sum;

        println!("Finished round: {}", t + 1);
    }

    hypotheses
}

fn main() {
    let image_collection = get_training_data();

    let hypothesis = adaboost(image_collection);

    for (i, ensemble_component) in hypothesis.ensemble.iter().enumerate() {
        println!("w({}) = {:?}", i, ensemble_component.0);
        println!("h({}) = {:?}", i, ensemble_component.1);
    }
}
