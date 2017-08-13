use shared::DataPoint;
use haar_like_feature::HaarLikeFeature;

#[derive(Debug)]
pub struct PredictionHypothesis {
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

pub struct PredictionEnsemble {
    ensemble: Vec<(f64, PredictionHypothesis)>,
}

impl PredictionEnsemble {
    pub fn new() -> PredictionEnsemble {
        PredictionEnsemble { ensemble: Vec::new() }
    }

    pub fn print_mistakes(&self, image_collection: &Vec<DataPoint>) {
        let mut bad_predictions = 0;

        for data_point in image_collection.iter() {
            let mut prediction = 0.0;

            for &(w, ref h) in self.ensemble.iter() {
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

    pub fn push(&mut self, prediction: (f64, PredictionHypothesis)) {
        self.ensemble.push(prediction);
    }
}
