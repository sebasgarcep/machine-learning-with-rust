use shared::DataPoint;
use haar_like_feature::HaarLikeFeature;

pub struct PredictionEnsemble {
    ensemble: Vec<HaarLikeFeature>,
}

impl PredictionEnsemble {
    pub fn new() -> PredictionEnsemble {
        PredictionEnsemble { ensemble: Vec::new() }
    }

    pub fn print_mistakes(&self, image_collection: &Vec<DataPoint>) {
        let mut bad_predictions = 0;

        for data_point in image_collection.iter() {
            let prediction = self.ensemble
                .iter()
                .fold(0.0, |acc, ref h| acc + h.predict(&data_point))
                .signum();

            if prediction * data_point.label < 0.0 {
                bad_predictions += 1;
            }
        }

        println!("Ensemble bad predictions: {} / {}",
                 bad_predictions,
                 image_collection.len());
    }

    pub fn push(&mut self, prediction: HaarLikeFeature) {
        self.ensemble.push(prediction);
    }
}
