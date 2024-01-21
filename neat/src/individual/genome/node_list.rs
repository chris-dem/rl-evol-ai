use num::rational::Ratio;
use std::sync::Arc;

use super::{clamp::Clamp, aggregation::Aggregation, activation::Activation};

pub trait Activate {
    fn activate(&self, x: f32) -> f32;
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
    pub output: Vec<Node>, // Due to mutation, output cells also get mutated
    pub hidden: Vec<Node>,
}
