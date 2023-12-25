use rand::Rng;

use crate::individual::genome::node_list::{
    Activate, Activation, Aggregation, Clamp, Config, Node,
};

use super::crossover::Crossover;

impl Crossover for Node {
    fn crossover(&self, rng: &mut impl Rng, fit: f32, other: &Self, other_fit: f32) -> Self {
        assert_eq!(
            self.node_id, other.node_id,
            "Should cross over the same node"
        );
        assert_eq!(self.level, other.level, "Level should be the same");
        Self {
            node_id: self.node_id,
            level: self.level,
            config: Config {
                aggregation: self.config.aggregation.crossover(
                    rng,
                    fit,
                    &other.config.aggregation,
                    other_fit,
                ),
                clamp: self
                    .config
                    .clamp
                    .crossover(rng, fit, &other.config.clamp, other_fit),
                activation: self.config.activation.crossover(
                    rng,
                    fit,
                    &other.config.activation,
                    other_fit,
                ),
            },
        }
    }
}

impl Crossover for Aggregation {
    fn crossover(&self, rng: &mut impl Rng, fit: f32, other: &Self, other_fit: f32) -> Self {
        todo!()
    }
}
impl Crossover for Clamp {
    fn crossover(&self, rng: &mut impl Rng, fit: f32, other: &Self, other_fit: f32) -> Self {
        todo!()
    }
}

impl Crossover for Activation {
    fn crossover(&self, rng: &mut impl Rng, fit: f32, other: &Self, other_fit: f32) -> Self {
        todo!()
    }
}
