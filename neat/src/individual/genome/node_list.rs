use std::sync::Arc;
use num::rational::Ratio;

#[derive(Debug, Clone, Copy)]
pub struct Node {
    pub node_id: usize,
    pub level: Ratio<usize>,
}

impl Node {
    pub fn new(node_id: usize, level: Ratio<usize>) -> Self {
        Self { node_id, level }
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
