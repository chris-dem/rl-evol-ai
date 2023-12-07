use itertools::Itertools;
use num::rational::Ratio;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use super::node_list::{Node, NodeList};

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

const MIN_RATIO: Ratio<usize> = Ratio::new(1, 1);
const MAX_RATIO: Ratio<usize> = Ratio::new(100, 1);

impl GenomeFactory {
    pub fn init(input: usize, output: usize) -> Result<Self, GenonomeError> {
        if input == 0 || output == 0 {
            return Err(GenonomeError::ZeroIOVector);
        }
        let mut id_generator = 0..input + output;
        let input_list: Arc<_> = Arc::from_iter(
            (&mut id_generator)
                .take(input)
                .map(|id| Node::new(id, MIN_RATIO)),
        );
        let output_list: Arc<_> = Arc::from_iter(id_generator.map(|id| Node::new(id, MAX_RATIO)));
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
    node_list: NodeList,
    genome_list: Vec<GenomeEdge>,
    memory: Vec<Memory>,
    pass: bool,
    edge_map: Vec<Vec<Edge>>,
}

#[derive(Debug, Clone, Copy)]
struct GenomeEdge {
    in_node: usize,
    out_node: usize,
    weight: f32,
    enabled: bool,
}

#[derive(Debug, Clone, Copy)]
struct Edge {
    dest: usize,
    weight: f32,
}

type PrevCurr = (f32, f32);

#[derive(Debug, Clone, Copy)]
struct Memory {
    cell_node: Node,
    mem_cell: PrevCurr,
    pass: bool,
}

impl Memory {
    fn new(cell_node: Node) -> Self {
        Memory {
            cell_node,
            mem_cell: (0., 0.),
            pass: false,
        }
    }
}

impl Genome {
    fn new(node_list: NodeList, genome_list: Vec<GenomeEdge>) -> Self {
        let memory = node_list
            .input
            .iter()
            .chain(node_list.output.iter())
            .chain(node_list.hidden.iter())
            .copied()
            .sorted()
            .map(Memory::new)
            .collect_vec();
        let mut edge_map = memory.iter().map(|_| Vec::new()).collect_vec();
        for GenomeEdge {
            in_node,
            out_node,
            weight,
            ..
        } in genome_list.iter().filter(|edge| edge.enabled).copied()
        {
            edge_map[in_node].push(Edge {
                dest: out_node,
                weight,
            })
        }
        Self {
            node_list,
            genome_list,
            memory,
            pass: false,
            edge_map,
        }
    }

    // Assumption of memory
    pub fn forward(&mut self, p: Vec<f32>) -> Vec<f32> {
        self.pass = !self.pass;
        // BFS to traverse the network
        let mut queue = VecDeque::from_iter(self.node_list.hidden.iter().copied());
        while let Some(head) = queue.pop_front() {
            let val = self.memory[get_mem_location(&self.memory, head)].mem_cell;
            for Edge { dest, weight } in self.edge_map[head].iter().copied() {
                let index = get_mem_location(&self.memory, dest);
                self.memory[index].mem_cell += weight * val;
                if self.memory[index].pass != self.pass {
                    self.memory[index].pass = self.pass;
                    queue.push_back(dest);
                }
            }
        }
        let len_input = self.node_list.input.len();
        let len_output = self.node_list.output.len();
        // Extract output memory cells
        self.memory[len_input..len_input + len_output]
            .iter()
            .copied()
            .map(|Memory { mem_cell, .. }| mem_cell)
            .collect_vec()
    }
}

#[inline]
fn get_mem_location(memory: &[Memory], item: usize) -> usize {
    memory
        .binary_search_by_key(&item, |Memory { cell_id, .. }| *cell_id)
        .expect("Id should be in list")
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;
    use itertools::Itertools;
    use num::rational::{self, Ratio};
    use proptest::{array, prelude::*};

    proptest! {
        #[test]
        fn test_no_hidden(x1 in any::<f32>().prop_filter("should not be too big", |x| x.abs() < 50.),x2 in any::<f32>().prop_filter("should not be too big", |x| x.abs() < 50.), weights in array::uniform8(any::<f32>().prop_filter("avoid overflow", |c| c.abs() < 10.))){
            let edges = vec![
                GenomeEdge { in_node: 0,out_node: 2,weight: weights[0], enabled : true},
                GenomeEdge { in_node: 1,out_node: 2,weight: weights[1], enabled : true},
                GenomeEdge { in_node: 0,out_node: 3,weight: weights[2], enabled : true},
                GenomeEdge { in_node: 1,out_node: 3,weight: weights[3], enabled : true},
                GenomeEdge { in_node: 0,out_node: 4,weight: weights[4], enabled : true},
                GenomeEdge { in_node: 1,out_node: 4,weight: weights[5], enabled : true},
                GenomeEdge { in_node: 0,out_node: 5,weight: weights[6], enabled : true},
                GenomeEdge { in_node: 1,out_node: 5,weight: weights[7], enabled : true},
            ];
            let node_list = NodeList {
                input : vec![0,1].into(),
                output : vec![2,3,4,5].into(),
                hidden : vec![]
            };
            let mut genome = Genome::new(node_list, edges);
            let outputs = vec![
                x1 * weights[0] + x2 * weights[1],
                x1 * weights[2] + x2 * weights[3],
                x1 * weights[4] + x2 * weights[5],
                x1 * weights[6] + x2 * weights[7],
            ];
            let output_genome = genome.forward(vec![x1,x2]);
            prop_assert!(dbg!(outputs)
                        .iter()
                        .copied()
                        .zip_eq(dbg!(output_genome).into_iter())
                        .all(|(a,b)| relative_eq!(a,b))
                    );
        }

    }
    #[test]
    fn rational_test() {
        let a = rational::Ratio::new(3usize, 2);
        let b = rational::Ratio::new(3usize, 2);
        let c = (a + b) / 2;
        assert_eq!(Ratio::new(3, 2), c);
    }
}
