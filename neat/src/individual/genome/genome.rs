use itertools::Itertools;
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
    output_list: Vec<Node>,
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
        let output_list = Vec::from_iter(
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
            output: Vec::clone(&self.output_list),
            hidden: vec![],
        };
        Genome::new(node_list, vec![])
    }
}

pub struct Genome {
    pub node_list: NodeList,
    pub genome_list: OrderedGenomeList,
}

#[derive(Debug, Clone, Copy)]
pub struct GenomeEdge {
    pub innov_number: usize,
    pub in_node: usize,
    pub out_node: usize,
    pub weight: f32,
    pub enabled: bool,
}

impl PartialEq for GenomeEdge {
    fn eq(&self, other: &Self) -> bool {
        self.innov_number == other.innov_number
    }
}

impl Eq for GenomeEdge {}

impl PartialOrd for GenomeEdge {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.innov_number.partial_cmp(&other.innov_number)
    }
}

impl Ord for GenomeEdge {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.innov_number.cmp(&other.innov_number)
    }
}

pub struct OrderedGenomeList {
    pub edge_list: Vec<GenomeEdge>,
}

impl OrderedGenomeList {
    pub fn new(mut genome_list: Vec<GenomeEdge>) -> Self {
        genome_list.sort();
        Self {
            edge_list: genome_list,
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &GenomeEdge> {
        self.edge_list.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut GenomeEdge> {
        self.edge_list.iter_mut()
    }

    pub fn new_sorted(genome_list: impl Iterator<Item = GenomeEdge>) -> Self {
        let edge_list = genome_list.collect_vec();
        assert!(edge_list.windows(2).all(|a| a[0].cmp(&a[1]).is_le()));
        Self { edge_list }
    }
}

impl Genome {
    fn new(node_list: NodeList, genome_list: Vec<GenomeEdge>) -> Self {
        Self {
            node_list,
            genome_list: OrderedGenomeList::new(genome_list),
        }
    }
}
