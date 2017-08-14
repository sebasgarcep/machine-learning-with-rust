use integral_image::IntegralImage;
use shared::{DataPoint, MIN_FEATURE_HEIGHT, MIN_FEATURE_WIDTH, WINDOW_HEIGHT, WINDOW_WIDTH};

#[derive(Debug, Copy, Clone)]
enum HaarLikeFeatureType {
    TwoVertical, // two columns vertical
    TwoHorizontal, // two columns horizontal
    ThreeHorizontal, // three columns horizontal
    ThreeVertical, // three columns vertical
    FourCheckers, // four squares checkerboard
}

#[derive(Debug)]
pub struct HaarLikeFeature {
    pub threshold: f64,
    pub weight: f64,
    pub polarity: f64,
    feature_type: HaarLikeFeatureType,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
}

impl HaarLikeFeature {
    pub fn generate_all_features() -> Vec<HaarLikeFeature> {
        let mut feature_hypotheses = Vec::new();

        let polarities: Vec<f64> = vec![1.0, -1.0];

        let feature_type_collection = vec![HaarLikeFeatureType::TwoVertical,
                                           HaarLikeFeatureType::TwoHorizontal,
                                           HaarLikeFeatureType::ThreeHorizontal,
                                           HaarLikeFeatureType::ThreeVertical,
                                           HaarLikeFeatureType::FourCheckers];

        for y in 0..(WINDOW_HEIGHT - 2) {
            for x in 0..(WINDOW_WIDTH - 2) {
                for fraction_height in (MIN_FEATURE_HEIGHT - 1)..((WINDOW_HEIGHT - y) / 2) {
                    for fraction_width in (MIN_FEATURE_WIDTH - 1)..((WINDOW_WIDTH - x) / 2) {
                        let height = (fraction_height + 1) * 2;
                        let width = (fraction_width + 1) * 2;
                        for feature_type in feature_type_collection.iter() {
                            for polarity in polarities.iter() {
                                feature_hypotheses.push(HaarLikeFeature {
                                    feature_type: feature_type.clone(),
                                    polarity: polarity.clone(),
                                    threshold: 0.0,
                                    weight: 1.0,
                                    x: x,
                                    y: y,
                                    width: width,
                                    height: height,
                                });
                            }
                        }
                    }
                }
            }
        }

        feature_hypotheses
    }

    fn get_score_two_vertical(&self, integral_image: &IntegralImage) -> f64 {
        let left_box = integral_image.sum_region(self.x, self.y, self.width / 2, self.height);

        let right_box =
            integral_image.sum_region(self.x + self.width / 2, self.y, self.width / 2, self.height);

        left_box - right_box
    }

    fn get_score_two_horizontal(&self, integral_image: &IntegralImage) -> f64 {
        let upper_box = integral_image.sum_region(self.x, self.y, self.width, self.height / 2);

        let lower_box = integral_image.sum_region(self.x,
                                                  self.y + self.height / 2,
                                                  self.width,
                                                  self.height / 2);

        upper_box - lower_box
    }

    fn get_score_three_horizontal(&self, integral_image: &IntegralImage) -> f64 {
        let upper_box = integral_image.sum_region(self.x, self.y, self.width, self.height / 3);

        let mid_box = integral_image.sum_region(self.x,
                                                self.y + self.height / 3,
                                                self.width,
                                                self.height / 3);

        let lower_box = integral_image.sum_region(self.x,
                                                  self.y + 2 * self.height / 3,
                                                  self.width,
                                                  self.height / 3);

        upper_box - mid_box + lower_box
    }

    fn get_score_three_vertical(&self, integral_image: &IntegralImage) -> f64 {
        let left_box = integral_image.sum_region(self.x, self.y, self.width / 3, self.height);

        let mid_box =
            integral_image.sum_region(self.x + self.width / 3, self.y, self.width / 3, self.height);

        let right_box = integral_image.sum_region(self.x + 2 * self.width / 3,
                                                  self.y,
                                                  self.width / 3,
                                                  self.height);

        left_box - mid_box + right_box
    }

    fn get_score_four_checkers(&self, integral_image: &IntegralImage) -> f64 {
        let upper_left_box =
            integral_image.sum_region(self.x, self.y, self.width / 2, self.height / 2);

        let upper_right_box = integral_image.sum_region(self.x + self.width / 2,
                                                        self.y,
                                                        self.width / 2,
                                                        self.height / 2);

        let bottom_right_box = integral_image.sum_region(self.x,
                                                         self.y + self.height / 2,
                                                         self.width / 2,
                                                         self.height / 2);

        let bottom_left_box = integral_image.sum_region(self.x + self.width / 2,
                                                        self.y + self.height / 2,
                                                        self.width / 2,
                                                        self.height / 2);

        upper_left_box - upper_right_box - bottom_left_box + bottom_right_box
    }

    pub fn get_score(&self, data_point: &DataPoint) -> f64 {
        match self.feature_type {
            HaarLikeFeatureType::TwoVertical => {
                self.get_score_two_vertical(&data_point.integral_image)
            }
            HaarLikeFeatureType::TwoHorizontal => {
                self.get_score_two_horizontal(&data_point.integral_image)
            }
            HaarLikeFeatureType::ThreeHorizontal => {
                self.get_score_three_horizontal(&data_point.integral_image)
            }
            HaarLikeFeatureType::ThreeVertical => {
                self.get_score_three_vertical(&data_point.integral_image)
            }
            HaarLikeFeatureType::FourCheckers => {
                self.get_score_four_checkers(&data_point.integral_image)
            }
        }
    }

    pub fn predict(&self, data_point: &DataPoint) -> f64 {
        let score = self.get_score(data_point);
        self.weight * self.polarity * (score - self.threshold).signum()
    }
}
