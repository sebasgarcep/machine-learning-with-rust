extern crate rand;
extern crate rulinalg;

mod load;
mod shared;
mod range;

use shared::{DataPoint, Digit, Label};
use load::load_mnist_data;
use range::Range;
use rand::distributions::{IndependentSample, Normal};
use rand::{Rng, thread_rng};
use rulinalg::matrix::{BaseMatrix, Matrix};
use rulinalg::vector::Vector;
use rulinalg::norm::Euclidean;

fn sigmoid (z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

fn sigmoid_prime(z: f64) -> f64 {
    sigmoid(z) * (1.0 - sigmoid(z))
}

fn bp4 (delta: &Vector<f64>, prev_activation: &Vector<f64>) -> Matrix<f64> {
    Matrix::from_fn(delta.size(), prev_activation.size(), |k, j| {
        prev_activation[k] * delta[j]
    })
}

#[derive(Debug)]
struct Network<C: CostFunction> {
    pub sizes : Vec<usize>,
    pub biases : Vec<Vector<f64>>,
    pub weights : Vec<Matrix<f64>>,
    pub cost: C,
}

trait CostFunction {
    fn get (&self, a: &Vector<f64>, y: &Vector<f64>) -> f64;

    fn delta (&self, z: &Vector<f64>, a: &Vector<f64>, y: &Vector<f64>) -> Vector<f64>;
}

#[allow(dead_code)]
struct QuadraticCost;
impl CostFunction for QuadraticCost  {
    fn get (&self, a: &Vector<f64>, y: &Vector<f64>) -> f64 {
        0.5 * (a - y).norm(Euclidean).powi(2)
    }

    fn delta (&self, z: &Vector<f64>, a: &Vector<f64>, y: &Vector<f64>) -> Vector<f64> {
        (a - y).elemul(&z.clone().apply(&sigmoid_prime))
    }
}

struct CrossEntropyCost;
impl CostFunction for CrossEntropyCost  {
    #[allow(unused_variables)]
    fn get (&self, a: &Vector<f64>, y: &Vector<f64>) -> f64 {
        unimplemented!()
    }

    #[allow(unused_variables)]
    fn delta (&self, z: &Vector<f64>, a: &Vector<f64>, y: &Vector<f64>) -> Vector<f64> {
        (a - y)
    }
}


impl <C: CostFunction> Network <C> {
    fn new (cost: C, sizes: Vec<usize>) -> Network<C> {
        let normal_dist = Normal::new(0.0, 1.0);
        let biases = sizes.iter().skip(1)
            .map(|&y| Vector::from_fn(y, |_| normal_dist.ind_sample(&mut thread_rng())))
            .collect();
        let weights = sizes.iter().take(sizes.len() - 1).zip(sizes.iter().skip(1))
            .map(|(&x, &y)| Matrix::from_fn(y, x, |_, _| normal_dist.ind_sample(&mut thread_rng())))
            .collect();
        Network { sizes: sizes, biases: biases, weights: weights, cost: cost }
    }

    fn num_layers (&self) -> usize {
        self.sizes.len()
    }

    fn feedforward (&self, a: &Vector<f64>) -> Vector<f64> {
        let mut a_prime = a.clone();
        for (b, w) in self.biases.iter().zip(self.weights.iter()) {
            a_prime = (w * a_prime + b).apply(&sigmoid);
        }
        return a_prime;
    }

    fn sgd (&mut self, training_data: &mut Vec<DataPoint>, epochs: usize, mini_batch_size: usize, eta: f64, maybe_test_data : Option<&Vec<DataPoint>>) {
        let n = training_data.len();
        for j in 0..epochs {
            let slice = training_data.as_mut_slice();
            thread_rng().shuffle(slice);
            let mini_batches = Range::new(0, n, Some(mini_batch_size))
                .map(|k| slice[k..k+mini_batch_size].to_vec());
            for mini_batch in mini_batches {
                self.update_mini_batch(mini_batch, eta);
            }

            if let Some(test_data) = maybe_test_data {
                println!("Epoch {}: {} / {}", j, self.evaluate(test_data), test_data.len());
            } else {
                println!("Epoch {} complete", j);
            }
        }
    }

    fn new_biases_with_zeros (&self) -> Vec<Vector<f64>> {
        self.biases.iter()
            .map(|b| Vector::zeros(b.size()))
            .collect()
    }

    fn new_weights_with_zeros (&self) -> Vec<Matrix<f64>> {
        self.weights.iter()
            .map(|w| Matrix::zeros(w.rows(), w.cols()))
            .collect()
    }

    fn update_mini_batch (&mut self, mini_batch: Vec<DataPoint>, eta: f64) {
        let mut nabla_b = self.new_biases_with_zeros();
        let mut nabla_w = self.new_weights_with_zeros();
        let len_mini_batch = mini_batch.len() as f64;
        for (x, y) in mini_batch {
            let (delta_nabla_b, delta_nabla_w) = self.backprop(&x, &y);
            nabla_b = nabla_b.iter().zip(delta_nabla_b.iter())
                .map(|(nb, dnb)| nb + dnb)
                .collect();
            nabla_w = nabla_w.iter().zip(delta_nabla_w.iter())
                .map(|(nw, dnw)| nw + dnw)
                .collect();
        }
        self.weights = self.weights.iter().zip(nabla_w.iter())
            .map(|(w, nw)| w - nw * (eta / len_mini_batch))
            .collect();
        self.biases = self.biases.iter().zip(nabla_b.iter())
            .map(|(b, nb)| b - nb * (eta / len_mini_batch))
            .collect();
    }

    /*fn total_cost (&self, data: &Vec<DataPoint>, lambda: f64) -> f64 {

    }*/

    fn predict (&self, input: &Digit) -> usize {
        self.feedforward(input).argmax().0
    }

    fn evaluate(&self, test_data: &Vec<DataPoint>) -> usize {
        test_data.iter()
            .map(|&(ref x, ref y)| if self.predict(x) == y.argmax().0 { 1 } else { 0 })
            .fold(0, |sum, x| sum + x)
    }

    fn backprop (&self, x: &Digit, y: &Label) -> (Vec<Vector<f64>>, Vec<Matrix<f64>>) {
        let mut nabla_b = self.new_biases_with_zeros();
        let mut nabla_w = self.new_weights_with_zeros();
        // feedforward
        let initial_vals = (x.clone(), vec![], vec![x.clone()]); // (initial activation, zs, activations)
        let (_, zs, activations) = self.biases.iter().zip(self.weights.iter())
        .fold(initial_vals, |(activation, mut zs, mut activations), (b, w)| {
            let z = w * activation + b;
            zs.push(z.clone());
            let sigmoid_z = z.apply(&sigmoid);
            activations.push(sigmoid_z.clone());
            (sigmoid_z, zs, activations)
        });

        // lens to avoid borrowing issues
        let nabla_b_len = nabla_b.len();
        let nabla_w_len = nabla_w.len();
        let zs_len = zs.len();
        let weights_len = self.weights.len();
        let activations_len = activations.len();

        // backward pass
        let mut delta = self.cost.delta(
            &zs.last().unwrap().clone().apply(&sigmoid_prime),
            activations.last().unwrap(),
            &y
        );

        // BP3
        nabla_b[nabla_b_len - 1] = delta.clone();
        // BP4
        nabla_w[nabla_w_len - 1] = bp4(&delta, &activations[activations_len - 2]);

        for l in 2..self.num_layers() {
            let z = zs[zs_len - l].clone();
            let sp = z.apply(&sigmoid_prime);
            delta = (self.weights[weights_len - l + 1].transpose() * delta).elemul(&sp);
            nabla_b[nabla_b_len - l] = delta.clone();
            nabla_w[nabla_w_len - l] = bp4(&delta, &activations[activations_len - l - 1]);
        }

        (nabla_b, nabla_w)
    }
}

fn main () {
    let mut training_data = load_mnist_data();
    let sizes = vec![784, 30, 10];
    let mut network = Network::new(CrossEntropyCost, sizes);
    let test_data = training_data.split_off(50000);
    assert_eq!(training_data.len(), 50000);
    assert_eq!(test_data.len(), 10000);
    network.sgd(&mut training_data, 30, 10, 3.0, Some(&test_data));
}
