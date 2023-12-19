use itertools::Itertools;
use num::rational::Ratio;
use std::sync::Arc;
use std::{cmp::Reverse, collections::BinaryHeap};

use super::{
    mem_cell::{MemoryCell, MemoryCellType},
    node_list::{Node, NodeList},
};

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

const MIN_RATIO: usize = 1;
const MAX_RATIO: usize = 100;

impl GenomeFactory {
    pub fn init(input: usize, output: usize) -> Result<Self, GenonomeError> {
        if input == 0 || output == 0 {
            return Err(GenonomeError::ZeroIOVector);
        }
        let mut id_generator = 0..input + output;
        let input_list: Arc<_> = Arc::from_iter(
            (&mut id_generator)
                .take(input)
                .map(|id| Node::new(id, Ratio::from_integer(MIN_RATIO))),
        );
        let output_list: Arc<_> =
            Arc::from_iter(id_generator.map(|id| Node::new(id, Ratio::from_integer(MAX_RATIO))));
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
    memory: Vec<MemoryCellType>,
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

impl Genome {
    fn new(node_list: NodeList, genome_list: Vec<GenomeEdge>) -> Self {
        let memory = node_list
            .input
            .iter()
            .map(|cell| MemoryCellType::Input {
                node_id: *cell,
                cell_value: 0.,
            })
            .chain(
                node_list
                    .output
                    .iter()
                    .chain(node_list.hidden.iter())
                    .map(|cell| MemoryCellType::Activation(MemoryCell::default(*cell))),
            )
            .sorted_by_key(|cell| cell.get_id().node_id)
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
    pub fn forward(&mut self, input_vector: &[f32]) -> Option<Vec<f32>> {
        self.pass = !self.pass;
        if input_vector.len() != self.node_list.input.len() {
            return None;
        }
        for (cell, val) in (0..self.node_list.input.len()).zip_eq(input_vector.iter().copied()) {
            self.memory[cell].propagate_input(val);
        }
        // BFS to traverse the network
        let mut queue = BinaryHeap::from_iter(self.node_list.input.iter().copied().map(Reverse));
        while let Some(Reverse(head)) = queue.pop() {
            let val_id = head;
            self.memory[val_id.node_id].activate(self.pass);
            for Edge { dest, weight } in self.edge_map[head.node_id].iter().copied() {
                let index = get_mem_location(&self.memory, dest);
                let target_cell = self.memory[index].get_id();
                let input = match val_id.cmp(&target_cell) {
                    std::cmp::Ordering::Less => self.memory[val_id.node_id]
                        .get_current_output(self.pass)
                        .expect(
                            "This must be a forward conneciton therefore we caluclated the output",
                        ), // forward
                    _ => self.memory[val_id.node_id].get_previous_output(self.pass),
                };
                self.memory[target_cell.node_id].propagate_input(input * weight);
                if self.memory[index].was_not_passed_set(self.pass) {
                    queue.push(Reverse(self.memory[index].get_id()));
                }
            }
        }
        let len_input = self.node_list.input.len();
        let len_output = self.node_list.output.len();
        // Extract output memory cells
        Some(
            self.memory[len_input..len_input + len_output]
                .iter()
                .map(|cell| cell.get_current_output(self.pass).unwrap_or(0.))
                .collect_vec(),
        )
    }
}

#[inline]
fn get_mem_location(memory: &[MemoryCellType], item: usize) -> usize {
    memory
        .binary_search_by_key(&item, |cell| cell.get_id().node_id)
        .expect("Id should be in list")
}

#[cfg(test)]
mod tests {
    use crate::genome::mem_cell::{Activate, Activation};

    use super::*;
    use approx::*;
    use itertools::Itertools;
    use num::rational::{self, Ratio};

    #[test]
    fn test_no_hidden() {
        let weights = [0.5; 8];
        let edges = vec![
            GenomeEdge {
                in_node: 0,
                out_node: 2,
                weight: weights[0],
                enabled: true,
            },
            GenomeEdge {
                in_node: 1,
                out_node: 2,
                weight: weights[1],
                enabled: true,
            },
            GenomeEdge {
                in_node: 0,
                out_node: 3,
                weight: weights[2],
                enabled: true,
            },
            GenomeEdge {
                in_node: 1,
                out_node: 3,
                weight: weights[3],
                enabled: true,
            },
            GenomeEdge {
                in_node: 0,
                out_node: 4,
                weight: weights[4],
                enabled: true,
            },
            GenomeEdge {
                in_node: 1,
                out_node: 4,
                weight: weights[5],
                enabled: true,
            },
            GenomeEdge {
                in_node: 0,
                out_node: 5,
                weight: weights[6],
                enabled: true,
            },
            GenomeEdge {
                in_node: 1,
                out_node: 5,
                weight: weights[7],
                enabled: true,
            },
        ];
        let node_list = NodeList {
            input: Arc::from_iter(
                [0, 1]
                    .map(|c| Node {
                        node_id: c,
                        level: Ratio::from_integer(1),
                    })
                    .into_iter(),
            ),
            output: Arc::from_iter(
                [2, 3, 4, 5]
                    .map(|c| Node {
                        node_id: c,
                        level: Ratio::from_integer(100),
                    })
                    .into_iter(),
            ),
            hidden: vec![],
        };
        let (x1, x2) = (0.1, 0.5);
        let mut genome = Genome::new(node_list, edges);
        let outputs = vec![
            Activation::Relu.activate((x1 * weights[0] + x2 * weights[1]) / 2.),
            Activation::Relu.activate((x1 * weights[2] + x2 * weights[3]) / 2.),
            Activation::Relu.activate((x1 * weights[4] + x2 * weights[5]) / 2.),
            Activation::Relu.activate((x1 * weights[6] + x2 * weights[7]) / 2.),
        ];
        let output_genome = genome.forward(&vec![x1, x2]);
        assert!(dbg!(outputs)
            .iter()
            .copied()
            .zip_eq(
                dbg!(output_genome)
                    .expect("Should be legal input")
                    .into_iter()
            )
            .all(|(a, b)| relative_eq!(a, b)));
    }

    mod hidden {
        use super::*;
        #[test]
        fn test_some_hidden_no_back() {
            let weights = [2.; 8];
            let edges = vec![
                GenomeEdge {
                    in_node: 0,
                    out_node: 2,
                    weight: weights[0],
                    enabled: true,
                },
                GenomeEdge {
                    in_node: 0,
                    out_node: 4,
                    weight: weights[1],
                    enabled: true,
                },
                GenomeEdge {
                    in_node: 1,
                    out_node: 3,
                    weight: weights[2],
                    enabled: true,
                },
                GenomeEdge {
                    in_node: 1,
                    out_node: 5,
                    weight: weights[3],
                    enabled: true,
                },
                GenomeEdge {
                    in_node: 4,
                    out_node: 2,
                    weight: weights[4],
                    enabled: true,
                },
                GenomeEdge {
                    in_node: 5,
                    out_node: 3,
                    weight: weights[5],
                    enabled: true,
                },
            ];
            let node_list = NodeList {
                input: Arc::from_iter(
                    [0, 1]
                        .map(|c| Node {
                            node_id: c,
                            level: Ratio::from_integer(1),
                        })
                        .into_iter(),
                ),
                output: Arc::from_iter(
                    [2, 3]
                        .map(|c| Node {
                            node_id: c,
                            level: Ratio::from_integer(100),
                        })
                        .into_iter(),
                ),
                hidden: [4, 5]
                    .map(|c| Node {
                        node_id: c,
                        level: Ratio::from_integer(50),
                    })
                    .into(),
            };
            let (x1, x2) = (0.1, 0.5);
            let mut genome = Genome::new(node_list, edges);
            let outputs = vec![0.3, 1.5];
            let output_genome = genome.forward(&vec![x1, x2]);
            assert!(dbg!(outputs)
                .iter()
                .copied()
                .zip_eq(
                    dbg!(output_genome)
                        .expect("Should be legal input")
                        .into_iter()
                )
                .all(|(a, b)| relative_eq!(a, b)));
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
