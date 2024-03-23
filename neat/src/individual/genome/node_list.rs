use num::rational::Ratio;
use std::sync::Arc;

use super::{activation::Activation, aggregation::Aggregation, clamp::Clamp};

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

    pub fn into_level(&self) -> LevelNode {
        LevelNode(*self)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LevelNode(pub Node);

impl PartialEq for LevelNode {
    fn eq(&self, other: &Self) -> bool {
        self.0.level == other.0.level
    }
}

impl Eq for LevelNode {}

impl PartialOrd for LevelNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.level.partial_cmp(&other.0.level)
    }
}

impl Ord for LevelNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other)
            .expect("Rational numbers are well ordered")
    }
}


impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.node_id == other.node_id
    }
}

impl Eq for Node {}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.node_id.partial_cmp(&other.node_id)
    }
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.node_id.cmp(&other.node_id)
    }
}

#[derive(Debug, Clone)]
pub struct NodeList {
    pub input: Arc<[Node]>,
    pub output: Vec<Node>, // Due to mutation, output cells also get mutated
    pub hidden: Vec<Node>,
}

impl NodeList {
    // Create node list assuming that hidden list is sorted
    pub fn new(input: Arc<[Node]>, output: Vec<Node>, hidden: Vec<Node>) -> Self {
        assert!(hidden.windows(2).all(|w| w[0].node_id < w[1].node_id));
        Self {
            input : input.clone(),
            output: output.clone(),
            hidden,
        }
    }
}
