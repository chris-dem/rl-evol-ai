use std::sync::Arc;

use num::rational::Ratio;

#[derive(Debug, Clone, Copy)]
pub struct Node {
    node_id: usize,
    level: Ratio<usize>,
}

impl Node {
    pub fn new(node_id: usize, level: Ratio<usize>) -> Self {
        Self { node_id, level }
    }
}

#[derive(Debug, Clone)]
pub struct NodeList {
    pub input: Arc<[Node]>,
    pub output: Arc<[Node]>,
    pub hidden: Vec<Node>,
}
