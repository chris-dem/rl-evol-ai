use std::collections::BTreeSet as TreeSet;

use itertools::Itertools;
use rand::prelude::*;

use crate::individual::genome::{genome::{Genome, GenomeEdge}, node_list::{Node, Config}, clamp::Clamp, aggregation::Aggregation, activation::Activation};

use super::innovation_number::InnovNumber;

pub trait MutationMethod {
    fn mutate(&self, rng: &mut dyn RngCore, child: &mut Genome , innov_number: &mut InnovNumber);
}

#[derive(Clone, Debug, Copy)]
pub struct ProbabilityMatrixNode {
    prob_clamp : f64,
    prob_activation : f64,
    prob_aggregation : f64
}

#[derive(Clone, Debug, Copy)]
pub struct ProbabilityMatrixEdge {
    prob_enabled : f64,
    prob_weight : f64,
    prob_new_node : f64,
    prob_new_edge : f64,
}

#[derive(Clone, Debug, Copy)]
pub struct ProbabilityMatrix {
    node_probs: ProbabilityMatrixNode,
    prob_edge : ProbabilityMatrixEdge,
}

#[derive(Clone, Debug, Copy)]
pub struct GaussianMutation {
    /// Probability of a changing gene
    pub prob: ProbabilityMatrix,
    /// Coefficient for the mutation
    pub coeff : f32,
    /// Iteration loopa
    pub max_iteration : usize
}

impl Default for GaussianMutation {
    fn default() -> Self {
        let a= 0.5;
        Self {
            prob:  ProbabilityMatrix {
                node_probs: ProbabilityMatrixNode{
                    prob_clamp: 0.5,
                    prob_activation: 0.5,
                    prob_aggregation: 0.5,
                },
                prob_edge: ProbabilityMatrixEdge {
                    prob_weight: 0.5,
                    prob_enabled: 0.5,
                    prob_new_node: 0.5,
                    prob_new_edge: 0.5,
                }
            },
            coeff: 1.,
            max_iteration: 10,
        }
    }
}

impl GaussianMutation {
    pub fn new(prob: ProbabilityMatrix, coeff : f32, max_iteration : usize) -> Self {
        Self { prob, coeff, max_iteration }
    }
}

fn weight_mutation(rng: &mut dyn RngCore, coeff: f32) -> f32 {
    (rng.gen::<f32>() * 2. - 1.) * coeff
}

pub trait Mutation {
    fn mutate(&mut self, rng: &mut dyn RngCore);
}

impl Mutation for Clamp {
    fn mutate(&mut self, rng: &mut dyn RngCore) {
        self.min_limit = self.min_limit.map(|x| x + weight_mutation(rng, 1.));
        self.max_limit = self.max_limit.map(|x| x + weight_mutation(rng, 1.));
    }
}

impl Mutation for Aggregation {
    fn mutate(&mut self, rng: &mut dyn RngCore) {
        *self = rng.gen::<Aggregation>();
    }
}

impl Mutation for Activation {
    fn mutate(&mut self, rng: &mut dyn RngCore) {
        *self = match rng.gen::<Activation>() {
            Activation::Softplus(_) => Activation::Softplus(weight_mutation(rng, 1.)),
            Activation::Selu(_,_) => Activation::Selu(weight_mutation(rng, 1.),weight_mutation(rng, 1.)),
            v => v
        }
    }
}

impl MutationMethod for GaussianMutation {
    fn mutate(&self, rng: &mut dyn RngCore, Genome {genome_list, node_list}: &mut Genome, innov_number : &mut InnovNumber) {
        let prob_node = self.prob.node_probs;
        for Node {config, ..} in node_list.hidden.iter_mut().chain(node_list.output.iter_mut()) {
            // Mutate 
            if rng.gen_bool(prob_node.prob_clamp) {
                config.clamp.mutate(rng)
            }
            if rng.gen_bool(prob_node.prob_aggregation) {
                config.aggregation.mutate(rng);
            }
            if rng.gen_bool(prob_node.prob_activation) {
                config.activation.mutate(rng);
            }
        }
        let prob_edge = self.prob.prob_edge;
        // Weight mutation
        for v in genome_list.iter_mut() {
            if rng.gen_bool(prob_edge.prob_enabled) {
                v.enabled = !v.enabled;
            }

            if rng.gen_bool(prob_edge.prob_weight) {
                v.weight += weight_mutation(rng, self.coeff);
            }
        }
        let concated_list = [node_list.input.iter(),node_list.output.iter(), node_list.hidden.iter()].into_iter().flatten().collect_vec();
        // Topological mutations
        if rng.gen_bool(prob_edge.prob_new_node) {
            let edge = genome_list
                        .iter_mut()
                        .choose(rng)
                        .unwrap();
            let node_start = concated_list[concated_list.binary_search_by(|a| a.node_id.cmp(&edge.in_node)).unwrap()];
            let node_end = concated_list[concated_list.binary_search_by(|a| a.node_id.cmp(&edge.out_node)).unwrap()];
            edge.enabled = false;
            let number = innov_number.next();
            let new_node = Node { 
                node_id: number,
                level: (node_start.level + node_end.level) / 2,
                config: Config {
                    aggregation: rng.gen(),
                    clamp: Clamp::default(),
                    activation: rng.gen(),
                },
            };
            let number = innov_number.next();
            let edge1 = GenomeEdge {
                in_node: node_start.node_id,
                out_node: new_node.node_id,
                innov_number: number,
                weight: 2. * rng.gen::<f32>() - 1.,
                enabled: true,
            };
            let number = innov_number.next();
            let edge2 = GenomeEdge {
                in_node: new_node.node_id,
                out_node: node_end.node_id,
                innov_number: number,
                weight: 2. * rng.gen::<f32>() - 1.,
                enabled: true,
            }; 
            genome_list.edge_list.push(edge1);
            genome_list.edge_list.push(edge2);
            node_list.hidden.push(new_node);
        }
        if rng.gen_bool(prob_edge.prob_new_edge) {
            let n = node_list.input.len();
            let p = node_list.hidden.len() + node_list.output.len();
            let total = n * p + p * (p - 1);
            if genome_list.edge_list.len() != total {
                let ratio = genome_list.edge_list.len() as f64 / total as f64;
                let attempt = (0.01f64.log(ratio).ceil().min(100.) as usize + 2).min(self.max_iteration);
                let map = TreeSet::from_iter(genome_list.iter().map(|el| (el.in_node,el.out_node)));
                for _ in 0..attempt {
                    let start = [
                        node_list.input.iter(),
                        node_list.hidden.iter(),
                        node_list.output.iter(),
                    ].into_iter().flatten().choose(rng).unwrap();
                    let end = [
                        node_list.hidden.iter(),
                        node_list.output.iter(),
                    ].into_iter().flatten().choose(rng).unwrap();
                    if !map.contains(&(start.node_id,end.node_id)) {
                        genome_list.edge_list.push(GenomeEdge {
                            innov_number: innov_number.next(),
                            in_node: start.node_id,
                            out_node: end.node_id,
                            weight: 2. * rng.gen::<f32>() - 1.,
                            enabled: rng.gen_bool(0.9),
                        });
                        break
                    }
                }
            }
        }
    }
}