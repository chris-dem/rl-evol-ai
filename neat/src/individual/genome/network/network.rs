use super::mem_cell::MemoryCellType;
use crate::individual::genome::{
    genome::GenomeEdge, network::mem_cell::MemoryCell, node_list::NodeList,
};
use itertools::Itertools;
use std::{cmp::Reverse, collections::BinaryHeap};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Lengths {
    input: usize,
    output: usize,
    hidden: usize,
}

pub struct FFNetwork {
    memory: Vec<MemoryCellType>,
    pass: bool,
    edge_map: Vec2D<Edge>,
    back_map: Vec2D<Edge>,
    lengths: Lengths,
}

type Vec2D<T> = Vec<Vec<T>>;

#[derive(Debug, Clone, Copy)]
struct Edge {
    dest: usize,
    weight: f32,
}

#[inline]
fn get_mem_location(memory: &[MemoryCellType], item: usize) -> usize {
    memory
        .binary_search_by_key(&item, |cell| cell.get_node().node_id)
        .expect("Id should be in list")
}

impl FFNetwork {
    fn new(node_list: NodeList, genome_list: Vec<GenomeEdge>) -> Self {
        let memory = node_list
            .input
            .iter()
            .map(|cell| MemoryCellType::Input {
                node: *cell,
                cell_value: 0.,
            })
            .chain(
                node_list
                    .output
                    .iter()
                    .chain(node_list.hidden.iter())
                    .map(|cell| MemoryCellType::Activation(MemoryCell::default(*cell))),
            )
            .sorted_by_key(|cell| cell.get_node().node_id)
            .collect_vec();
        let mut edge_map = memory.iter().map(|_| Vec::new()).collect_vec();
        let mut back_map = vec![vec![]; node_list.hidden.len()];
        for GenomeEdge {
            in_node,
            out_node,
            weight,
            ..
        } in genome_list.iter().filter(|edge| edge.enabled).copied()
        {
            let in_index = get_mem_location(&memory, in_node);
            let out_index = get_mem_location(&memory, out_node);
            let in_node_el = memory[in_index].get_node();
            let out_node_el = memory[out_index].get_node();
            if in_node_el.level >= out_node_el.level {
                back_map[out_index - (node_list.input.len() + node_list.output.len())].push(Edge {
                    dest: in_node,
                    weight,
                });
            } else {
                edge_map[in_index].push(Edge {
                    dest: out_node,
                    weight,
                });
            }
        }

        Self {
            memory,
            pass: false,
            edge_map,
            back_map,
            lengths: Lengths {
                input: node_list.input.len(),
                output: node_list.output.len(),
                hidden: node_list.hidden.len(),
            },
        }
    }

    #[inline]
    fn is_hidden(&self, node_id: usize) -> bool {
        self.lengths.input + self.lengths.output <= node_id
    }

    #[inline]
    fn translate_hidden(&self, node_id: usize) -> usize {
        let index = get_mem_location(&self.memory, node_id);
        index - (self.lengths.input + self.lengths.output)
    }

    // Assumption of memory
    pub fn forward(&mut self, input_vector: &[f32]) -> Option<Vec<f32>> {
        self.pass = !self.pass;
        if input_vector.len() != self.lengths.input {
            return None;
        }
        for (cell, val) in (0..self.lengths.input).zip_eq(input_vector.iter().copied()) {
            self.memory[cell].propagate_input(val);
        }
        // BFS to traverse the network
        let mut queue = BinaryHeap::from_iter(
            self.memory[0..self.lengths.input]
                .iter()
                .map(MemoryCellType::get_node)
                .map(Reverse),
        );
        while let Some(Reverse(head)) = queue.pop() {
            let head_id = head;
            let head_idx = get_mem_location(&self.memory, head_id.node_id);
            if self.is_hidden(head_id.node_id) {
                for v in self.back_map[self.translate_hidden(head_idx)]
                    .iter()
                    .copied()
                {
                    let index = get_mem_location(&self.memory, v.dest);
                    let inp = self.memory[index].get_previous_output(self.pass);
                    self.memory[head_idx].propagate_input(inp * v.weight);
                }
            }

            self.memory[head_idx].activate(self.pass);
            for Edge { dest, weight } in self.edge_map[head_idx].iter().copied() {
                let index = get_mem_location(&self.memory, dest);
                let target_cell = self.memory[index].get_node();
                let input = self.memory[head_idx]
                    .get_current_output(self.pass)
                    .expect("This must be a forward conneciton therefore we caluclated the output");
                self.memory[index].propagate_input(input * weight);
                if self.memory[index].was_not_passed_set(self.pass) {
                    queue.push(Reverse(self.memory[index].get_node()));
                }
            }
        }
        // Extract output memory cells
        Some(
            self.memory[self.lengths.input..self.lengths.input + self.lengths.output]
                .iter()
                .map(|cell| cell.get_current_output(self.pass).unwrap_or(0.))
                .collect_vec(),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::individual::genome::node_list::{Activate, Activation, Node};
    use std::sync::Arc;

    use super::*;
    use approx::*;
    use itertools::Itertools;
    use num::rational::{self, Ratio};

    #[test]
    fn test_no_hidden() {
        let weights = [0.5; 8];
        let edges = vec![
            GenomeEdge {
                innov_number: 0,
                in_node: 0,
                out_node: 2,
                weight: weights[0],
                enabled: true,
            },
            GenomeEdge {
                innov_number: 1,
                in_node: 1,
                out_node: 2,
                weight: weights[1],
                enabled: true,
            },
            GenomeEdge {
                innov_number: 2,
                in_node: 0,
                out_node: 3,
                weight: weights[2],
                enabled: true,
            },
            GenomeEdge {
                innov_number: 0,
                in_node: 1,
                out_node: 3,
                weight: weights[3],
                enabled: true,
            },
            GenomeEdge {
                innov_number: 0,
                in_node: 0,
                out_node: 4,
                weight: weights[4],
                enabled: true,
            },
            GenomeEdge {
                innov_number: 0,
                in_node: 1,
                out_node: 4,
                weight: weights[5],
                enabled: true,
            },
            GenomeEdge {
                innov_number: 0,
                in_node: 0,
                out_node: 5,
                weight: weights[6],
                enabled: true,
            },
            GenomeEdge {
                innov_number: 0,
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
                        config: Default::default(),
                    })
                    .into_iter(),
            ),
            output: Arc::from_iter(
                [2, 3, 4, 5]
                    .map(|c| Node {
                        node_id: c,
                        level: Ratio::from_integer(100),
                        config: Default::default(),
                    })
                    .into_iter(),
            ),
            hidden: vec![],
        };
        let (x1, x2) = (0.1, 0.5);
        let mut genome = FFNetwork::new(node_list, edges);
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
                    innov_number: 0,
                },
                GenomeEdge {
                    innov_number: 1,
                    in_node: 1,
                    out_node: 3,
                    weight: weights[2],
                    enabled: true,
                },
                GenomeEdge {
                    innov_number: 2,
                    in_node: 1,
                    out_node: 5,
                    weight: weights[3],
                    enabled: true,
                },
                GenomeEdge {
                    innov_number: 3,
                    in_node: 4,
                    out_node: 2,
                    weight: weights[4],
                    enabled: true,
                },
                GenomeEdge {
                    innov_number: 4,
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
                            config: Default::default(),
                        })
                        .into_iter(),
                ),
                output: Arc::from_iter(
                    [2, 3]
                        .map(|c| Node {
                            node_id: c,
                            level: Ratio::from_integer(100),
                            config: Default::default(),
                        })
                        .into_iter(),
                ),
                hidden: [4, 5]
                    .map(|c| Node {
                        node_id: c,
                        level: Ratio::from_integer(50),
                        config: Default::default(),
                    })
                    .into(),
            };
            let (x1, x2) = (0.1, 0.5);
            let mut genome = FFNetwork::new(node_list, edges);
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

        #[test]
        fn test_some_hidden_back() {
            // 2 in - 1 hid - 2 out
            // FFN plus on backward from second exit
            // weights: forward 3, backward 0.5
            // Expected run 1 : 1,1 -> 6,6
            // Expected run 2 : 0.5,0.5 -> 6,6
            let weights = [2., 2., 2., 2., -0.5];
            let edges = vec![
                GenomeEdge {
                    innov_number: 0,
                    in_node: 0,
                    out_node: 4,
                    weight: weights[0],
                    enabled: true,
                },
                GenomeEdge {
                    innov_number: 0,
                    in_node: 1,
                    out_node: 4,
                    weight: weights[1],
                    enabled: true,
                },
                GenomeEdge {
                    innov_number: 0,
                    in_node: 4,
                    out_node: 2,
                    weight: weights[2],
                    enabled: true,
                },
                GenomeEdge {
                    innov_number: 0,
                    in_node: 4,
                    out_node: 3,
                    weight: weights[3],
                    enabled: true,
                },
                GenomeEdge {
                    innov_number: 0,
                    in_node: 3,
                    out_node: 4,
                    weight: weights[4],
                    enabled: true,
                },
            ];
            let node_list = NodeList {
                input: Arc::from_iter(
                    [0, 1]
                        .map(|c| Node {
                            node_id: c,
                            level: Ratio::from_integer(1),
                            config: Default::default(),
                        })
                        .into_iter(),
                ),
                output: Arc::from_iter(
                    [2, 3]
                        .map(|c| Node {
                            node_id: c,
                            level: Ratio::from_integer(100),
                            config: Default::default(),
                        })
                        .into_iter(),
                ),
                hidden: [4]
                    .map(|c| Node {
                        node_id: c,
                        level: Ratio::from_integer(50),
                        config: Default::default(),
                    })
                    .into(),
            };
            let mut genome = FFNetwork::new(node_list, edges);
            let (x1, x2) = (0.3, 0.3);
            let outputs = vec![0.8, 0.8];
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

            let (x1, x2) = (0.1, 0.1);
            let outputs = vec![0., 0.];
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

        #[test]
        fn test_some_advanced_hidden_back() {
            // 2 in - 1 hid - 2 out
            // FFN plus on backward from second exit
            // weights: forward 3, backward 0.5
            // Expected run 1 : 1,1 -> 6,6
            // Expected run 2 : 0.5,0.5 -> 6,6
            let weights = [0.5; 7];
            let edges = vec![
                GenomeEdge {
                    innov_number: 0,
                    in_node: 0,
                    out_node: 3,
                    weight: weights[0],
                    enabled: true,
                },
                GenomeEdge {
                    innov_number: 0,
                    in_node: 1,
                    out_node: 4,
                    weight: weights[1],
                    enabled: true,
                },
                GenomeEdge {
                    innov_number: 0,
                    in_node: 4,
                    out_node: 5,
                    weight: weights[2],
                    enabled: true,
                },
                GenomeEdge {
                    innov_number: 0,
                    in_node: 3,
                    out_node: 5,
                    weight: weights[3],
                    enabled: true,
                },
                GenomeEdge {
                    innov_number: 0,
                    in_node: 5,
                    out_node: 4,
                    weight: weights[4],
                    enabled: true,
                },
                GenomeEdge {
                    innov_number: 0,
                    in_node: 3,
                    out_node: 2,
                    weight: weights[5],
                    enabled: true,
                },
                GenomeEdge {
                    innov_number: 0,
                    in_node: 5,
                    out_node: 2,
                    weight: weights[6],
                    enabled: true,
                },
            ];
            let node_list = NodeList {
                input: Arc::from_iter(
                    [0, 1]
                        .map(|c| Node {
                            node_id: c,
                            level: Ratio::from_integer(1),
                            config: Default::default(),
                        })
                        .into_iter(),
                ),
                output: Arc::from_iter(
                    [2].map(|c| Node {
                        node_id: c,
                        level: Ratio::from_integer(100),
                        config: Default::default(),
                    })
                    .into_iter(),
                ),
                hidden: [
                    Node {
                        node_id: 3,
                        level: Ratio::from_integer(25),
                        config: Default::default(),
                    },
                    Node {
                        node_id: 4,
                        level: Ratio::from_integer(25),
                        config: Default::default(),
                    },
                    Node {
                        node_id: 5,
                        level: Ratio::from_integer(50),
                        config: Default::default(),
                    },
                ]
                .into(),
            };
            let mut genome = FFNetwork::new(node_list, edges);
            let (x1, x2) = (1., 1.);
            let outputs = vec![0.171875];
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

            let (x1, x2) = (2., 2.);
            let outputs = vec![0.3466796875];
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

        #[test]
        fn test_some_hidden_no_back_no_order() {
            let weights = [2.; 8];
            let edges = vec![
                GenomeEdge {
                    innov_number: 0,
                    in_node: 0,
                    out_node: 2,
                    weight: weights[0],
                    enabled: true,
                },
                GenomeEdge {
                    innov_number: 0,
                    in_node: 0,
                    out_node: 6,
                    weight: weights[1],
                    enabled: true,
                },
                GenomeEdge {
                    innov_number: 0,
                    in_node: 1,
                    out_node: 3,
                    weight: weights[2],
                    enabled: true,
                },
                GenomeEdge {
                    innov_number: 0,
                    in_node: 1,
                    out_node: 7,
                    weight: weights[3],
                    enabled: true,
                },
                GenomeEdge {
                    innov_number: 0,
                    in_node: 6,
                    out_node: 2,
                    weight: weights[4],
                    enabled: true,
                },
                GenomeEdge {
                    innov_number: 0,
                    in_node: 7,
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
                            config: Default::default(),
                        })
                        .into_iter(),
                ),
                output: Arc::from_iter(
                    [2, 3]
                        .map(|c| Node {
                            node_id: c,
                            level: Ratio::from_integer(100),
                            config: Default::default(),
                        })
                        .into_iter(),
                ),
                hidden: [6, 7]
                    .map(|c| Node {
                        node_id: c,
                        level: Ratio::from_integer(50),
                        config: Default::default(),
                    })
                    .into(),
            };
            let (x1, x2) = (0.1, 0.5);
            let mut genome = FFNetwork::new(node_list, edges);
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
