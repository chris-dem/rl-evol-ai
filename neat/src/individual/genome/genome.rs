use num::rational::Ratio;
use std::sync::Arc;

use super::node_list::{Node, NodeList};

const MIN_RATIO: usize = 1;
const MAX_RATIO: usize = 100;

// Consider using the following
// * (Arc/Rc)<_>
// * TinyVec::Vec<_>
// * Im::Vector<_>
// Original vector

/// Generate Node Lists
#[derive(Debug, Clone)]
pub struct GenomeFactory {
    input_list: Arc<[Node]>,
    output_list: Arc<[Node]>,
}

pub enum GenonomeError {
    ZeroIOVector,
}

impl GenomeFactory {
    pub fn init(input: usize, output: usize) -> Result<Self, GenonomeError> {
        if input == 0 || output == 0 {
            return Err(GenonomeError::ZeroIOVector);
        }
        let mut id_generator = 0..input + output;
        let input_list: Arc<_> = Arc::from_iter(
            (&mut id_generator)
                .take(input)
                .map(|id| Node::new(id, Ratio::from_integer(MIN_RATIO), None)),
        );
        let output_list: Arc<_> = Arc::from_iter(
            id_generator.map(|id| Node::new(id, Ratio::from_integer(MAX_RATIO), None)),
        );
        Ok(Self {
            input_list,
            output_list,
        })
    }
    pub fn generate_genome(&self) -> Genome {
        let node_list = NodeList {
            input: Arc::clone(&self.input_list),
            output: Arc::clone(&self.output_list),
            hidden: vec![],
        };
        Genome::new(node_list, vec![])
    }
}

pub struct Genome {
    pub node_list: NodeList,
    pub genome_list: Vec<GenomeEdge>,
}

#[derive(Debug, Clone, Copy)]
pub struct GenomeEdge {
    pub in_node: usize,
    pub out_node: usize,
    pub weight: f32,
    pub enabled: bool,
}

impl Genome {
    fn new(node_list: NodeList, genome_list: Vec<GenomeEdge>) -> Self {
        Self {
            node_list,
            genome_list,
        }
    }
}
