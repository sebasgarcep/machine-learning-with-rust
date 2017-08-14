use integral_image::IntegralImage;
use haar_like_feature::HaarLikeFeature;

#[derive(Serialize, Deserialize, Debug)]
pub struct PredictionEnsemble {
    ensemble: Vec<Vec<HaarLikeFeature>>,
}

impl PredictionEnsemble {
    pub fn new() -> PredictionEnsemble {
        PredictionEnsemble { ensemble: Vec::new() }
    }

    pub fn predict(&self, integral_image: &IntegralImage) -> bool {
        for composition in self.ensemble.iter() {
            let prediction = composition.iter()
                .fold(0.0, |acc, ref h| acc + h.predict(&integral_image))
                .signum();

            if prediction < 0.0 {
                return false;
            }
        }

        return true;
    }

    pub fn push(&mut self, prediction: Vec<HaarLikeFeature>) {
        self.ensemble.push(prediction);
    }
}
