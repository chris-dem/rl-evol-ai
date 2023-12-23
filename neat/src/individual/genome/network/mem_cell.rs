use errorfunctions::RealErrorFunctions;
use itertools::Itertools;

use crate::individual::genome::node_list::Node;

pub trait Activate {
    fn activate(&self, x: f32) -> f32;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Aggregation {
    Sum,
    Max,
    Mean,
    L1NormAvg,
    L2NormAvg,
}

impl Aggregation {
    fn apply(&self, a: impl Iterator<Item = f32>) -> f32 {
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
    min_limit: Option<f32>,
    max_limit: Option<f32>,
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    Abs,
    Cube,
    Exp,
    Gauss,
    Hat,
    Identity,
    Inv,
    Log,
    Relu,
    Selu(f32, f32),
    Sigmoid,
    Sin,
    Tanh,
    Softplus(f32),
    Gelu,
    Root,
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

#[derive(Debug)]
pub struct MemoryCell {
    node_id: Node,
    current: f32,
    prev: f32,
    bias: f32,
    current_data: Vec<f32>,
    aggregation: Aggregation,
    clamp: Clamp,
    activation: Activation,
    activated: bool,
    pub passed: bool,
}

impl MemoryCell {
    pub fn default(node: Node) -> Self {
        Self::new(
            node,
            0.,
            Aggregation::Mean,
            Clamp {
                min_limit: Some(-5.),
                max_limit: Some(5.),
            },
            Activation::Relu,
        )
    }

    pub fn new(
        node: Node,
        bias: f32,
        aggregation: Aggregation,
        clamp: Clamp,
        activation: Activation,
    ) -> Self {
        MemoryCell {
            node_id: node,
            current: 0.,
            prev: 0.,
            bias: bias,
            current_data: Vec::new(),
            aggregation,
            clamp,
            activation: activation,
            activated: false,
            passed: false,
        }
    }
    pub fn get_node(&self) -> Node {
        self.node_id
    }

    pub fn activate(&mut self, pass_flag: bool) {
        let agg_data = self.aggregation.apply(self.current_data.iter().copied()) + self.bias;
        let current = self.clamp.activate(self.activation.activate(agg_data));
        let pass = if pass_flag { 1 } else { 0 };
        self.prev = self.current;
        self.current = current;
        self.activated = pass_flag;
        self.current_data.clear();
    }

    pub fn get_current_output(&self, pass_flag: bool) -> Option<f32> {
        if self.activated == pass_flag {
            Some(self.current)
        } else {
            None
        }
    }

    pub fn get_previous_output(&self, pass_flag: bool) -> f32 {
        if self.activated == pass_flag {
            self.prev
        } else {
            self.current
        }
    }

    pub fn append_input(&mut self, input: f32) {
        self.current_data.push(input);
    }
}

#[derive(Debug)]
pub enum MemoryCellType {
    Input { node_id: Node, cell_value: f32 },
    Activation(MemoryCell),
}

impl PartialEq for MemoryCellType {
    fn eq(&self, other: &Self) -> bool {
        self.get_id() == other.get_id()
    }
}

impl Eq for MemoryCellType {}

impl PartialOrd for MemoryCellType {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.get_id().partial_cmp(&other.get_id())
    }
}

impl Ord for MemoryCellType {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.get_id().cmp(&other.get_id())
    }
}

impl MemoryCellType {
    pub fn get_id(&self) -> Node {
        match self {
            MemoryCellType::Input {
                node_id,
                cell_value,
            } => *node_id,
            MemoryCellType::Activation(MemoryCell { node_id, .. }) => *node_id,
        }
    }

    pub fn was_not_passed_set(&mut self, pass_flag: bool) -> bool {
        match self {
            MemoryCellType::Input {
                node_id,
                cell_value,
            } => true,
            MemoryCellType::Activation(MemoryCell { passed, .. }) => {
                let prev = *passed != pass_flag;
                *passed = pass_flag;
                prev
            }
        }
    }

    pub fn propagate_input(&mut self, input: f32) {
        match self {
            MemoryCellType::Input {
                node_id,
                cell_value,
            } => *cell_value = input,
            MemoryCellType::Activation(c) => c.append_input(input),
        }
    }

    pub fn activate(&mut self, pass_flag: bool) {
        match self {
            MemoryCellType::Activation(c) => c.activate(pass_flag),
            _ => (),
        }
    }

    pub fn get_previous_output(&self, pass_flag: bool) -> f32 {
        match self {
            MemoryCellType::Input { cell_value, .. } => *cell_value, // should never occur
            MemoryCellType::Activation(c) => c.get_previous_output(pass_flag),
        }
    }

    pub fn get_current_output(&self, pass_flag: bool) -> Option<f32> {
        match self {
            MemoryCellType::Input { cell_value, .. } => Some(*cell_value), // should never occur
            MemoryCellType::Activation(c) => c.get_current_output(pass_flag),
        }
    }
}