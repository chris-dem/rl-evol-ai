use errorfunctions::RealErrorFunctions;
use itertools::Itertools;
use num::rational::Ratio;
use std::sync::Arc;

const MIN_CLAMP: f32 = -5.;
const MAX_CLAMP: f32 = -5.;

pub trait Activate {
    fn activate(&self, x: f32) -> f32;
}

impl Activate for Activation {
    fn activate(&self, input: f32) -> f32 {
        match self {
            Activation::Abs => input.abs(),                      // |x|
            Activation::Cube => input * input * input,           // x^3
            Activation::Exp => input.exp(),                      // e^x
            Activation::Gauss => (-(input * input)).exp(),       // e^(-x^2)
            Activation::Hat => (1. - input.abs()).max(0.),       // max(1 - |x|, 0)
            Activation::Identity => input,                       // x
            Activation::Inv => 1. / (input * input + 1.).sqrt(), // 1 / sqrt(x^2 + 1)
            Activation::Log => input.abs().ln(),                 // ln(|x|)
            Activation::Relu => input.max(0.),                   // max(x, 0)
            Activation::Selu(alpha, lambda) => {
                if input > 0. {
                    // lambda * x
                    lambda * input
                } else {
                    lambda * alpha * (input.exp() - 1.) // lambda * alpha * (e^x - 1)
                }
            }
            Activation::Sigmoid => 1. / (1. + (-input).exp()), // 1 / (1 + e^(-x))
            Activation::Sin => input.sin(),                    // sin x
            Activation::Tanh => input.tanh(),                  // tanh x
            Activation::Softplus(beta) => beta.recip() * (1. + (input * beta).exp()).ln(), // (1 / beta) * ln(1. + exp(x * beta))
            Activation::Gelu => ((input as f64 / 2.0_f64.sqrt()).erf() as f32 + 1.) * 0.5 * input, // x/2 (1 + erf(x / sqrt(2)))
            Activation::Root => (input * input + 1.).sqrt(),
        }
    }
}

impl Activate for Clamp {
    fn activate(&self, input: f32) -> f32 {
        let input = if let Some(m) = self.max_limit {
            input.min(m)
        } else {
            input
        };
        let input = if let Some(m) = self.min_limit {
            input.max(m)
        } else {
            input
        };
        return input;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum Aggregation {
    Sum,
    Max,
    #[default]
    Mean,
    L1NormAvg,
    L2NormAvg,
}

impl Aggregation {
    pub fn apply(&self, a: impl Iterator<Item = f32>) -> f32 {
        match self {
            Aggregation::Sum => a.sum(),
            Aggregation::Max => a.reduce(|a, b| f32::max(a, b)).unwrap_or(0.),
            Aggregation::Mean => {
                let x = a.fold((0., 0), |(acc, cnt), x| (acc + x, cnt + 1));
                x.0 / x.1 as f32
            }
            Aggregation::L2NormAvg => {
                let v = a.collect_vec();
                let alpha = v
                    .iter()
                    .copied()
                    .reduce(|a, b| f32::max(a.abs(), b.abs()))
                    .expect("Should not contain NaN");
                v.iter()
                    .copied()
                    .fold(0., |acc, x| acc + (x / alpha) * (x / alpha))
                    .sqrt()
                    * alpha
                    / v.len() as f32
            }
            Aggregation::L1NormAvg => {
                let p = a.fold((0., 0), |(acc, cnt), x| (acc + x.abs(), cnt + 1));
                p.0 / p.1 as f32
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Clamp {
    pub min_limit: Option<f32>,
    pub max_limit: Option<f32>,
}

impl Default for Clamp {
    fn default() -> Self {
        Self {
            min_limit: Some(MIN_CLAMP),
            max_limit: Some(MAX_CLAMP),
        }
    }
}

impl Clamp {
    pub fn new(min_limit: Option<f32>, max_limit: Option<f32>) -> Option<Self> {
        match (min_limit, max_limit) {
            (Some(a), Some(b)) => {
                if a >= b {
                    None
                } else {
                    Some(Clamp {
                        min_limit,
                        max_limit,
                    })
                }
            }
            (a, b) => Some(Clamp {
                min_limit: a,
                max_limit: b,
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Activation {
    Abs,
    Cube,
    Exp,
    Gauss,
    Hat,
    Identity,
    Inv,
    Log,
    #[default]
    Relu,
    Selu(f32, f32),
    Sigmoid,
    Sin,
    Tanh,
    Softplus(f32),
    Gelu,
    Root,
}

#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub aggregation: Aggregation,
    pub clamp: Clamp,
    pub activation: Activation,
}

#[derive(Debug, Clone, Copy)]
pub struct Node {
    pub node_id: usize,
    pub config: Config,
    pub level: Ratio<usize>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            aggregation: Default::default(),
            clamp: Default::default(),
            activation: Default::default(),
        }
    }
}

impl Node {
    pub fn new(node_id: usize, level: Ratio<usize>, config: Option<Config>) -> Self {
        Self {
            node_id,
            level,
            config: config.unwrap_or_default(),
        }
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.level == other.level
    }
}

impl Eq for Node {}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.level.partial_cmp(&other.level)
    }
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.level.cmp(&other.level)
    }
}

#[derive(Debug, Clone)]
pub struct NodeList {
    pub input: Arc<[Node]>,
    pub output: Arc<[Node]>,
    pub hidden: Vec<Node>,
}
