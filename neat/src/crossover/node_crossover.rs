use itertools::Itertools;
use rand::{Rng, RngCore};

use crate::individual::genome::{
    genome::GenomeEdge,
    node_list::{Config, Node}, clamp::Clamp, activation::Activation, aggregation::Aggregation,
};

use super::{crossover::Crossover, misc_crossover::CrossoverMisc};

impl Crossover for Node {
    fn crossover(&self, rng: &mut dyn RngCore, fit: f32, other: &Self, other_fit: f32) -> Self {
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

pub trait FloatList {
    type SelfItem;
    type Item;

    fn to_floats(&self) -> Vec<Self::Item>;
    fn from_floats_inner(
        &self,
        chromes: impl Iterator<Item = Self::Item>,
    ) -> Option<Self::SelfItem>;

    fn from_floats(&self, mut chromes: impl Iterator<Item = Self::Item>) -> Option<Self::SelfItem> {
        let res = self.from_floats_inner(&mut chromes);
        if chromes.next().is_some() {
            None
        } else {
            res
        }
    }
}

impl FloatList for Clamp {
    type Item = Option<f32>;
    type SelfItem = Self;

    fn to_floats(&self) -> Vec<Option<f32>> {
        vec![self.min_limit, self.max_limit]
    }

    fn from_floats_inner(&self, mut chromes: impl Iterator<Item = Option<f32>>) -> Option<Self> {
        Some(Self {
            min_limit: chromes.next()?,
            max_limit: chromes.next()?,
        })
    }
}

impl FloatList for Activation {
    type Item = f32;
    type SelfItem = Self;

    fn to_floats(&self) -> Vec<f32> {
        match self {
            Activation::Selu(a, b) => vec![*a, *b],
            Activation::Softplus(a) => vec![*a],
            Activation::Periodic(a) => vec![*a],
            _ => vec![],
        }
    }

    fn from_floats_inner(
        &self,
        mut chromes: impl Iterator<Item = Self::Item>,
    ) -> Option<Self::SelfItem> {
        Some(match self {
            Activation::Selu(_, _) => Activation::Selu(chromes.next()?, chromes.next()?),
            Activation::Softplus(_) => Activation::Softplus(chromes.next()?),
            Activation::Periodic(_) => Activation::Periodic(chromes.next()?.abs()),
            r => *r,
        })
    }
}

impl Crossover for Aggregation {
    fn crossover(&self, rng: &mut dyn RngCore, fit: f32, other: &Self, other_fit: f32) -> Self {
        if self != other {
            CrossoverMisc::default()
                .bernoulli_crossover::<Aggregation>(rng, *self, fit, *other, other_fit)
        } else {
            *self
        }
    }
}

impl Crossover for Clamp {
    fn crossover(&self, rng: &mut dyn RngCore, fit: f32, other: &Self, other_fit: f32) -> Self {
        self.from_floats(
            self.to_floats()
                .into_iter()
                .zip_eq(other.to_floats().into_iter())
                .map(|(a, b)| match (a, b) {
                    (None, None) => None,
                    (Some(a), None) | (None, Some(a)) => Some(a),
                    (Some(a), Some(b)) => {
                        Some(CrossoverMisc::default().f32_crossover(rng, a, fit, b, other_fit))
                    }
                }),
        )
        .expect("Weights should match")
    }
}

impl Crossover for Activation {
    fn crossover(&self, rng: &mut dyn RngCore, fit: f32, other: &Self, other_fit: f32) -> Self {
        self.from_floats(
            self.to_floats()
                .into_iter()
                .zip_eq(other.to_floats().into_iter())
                .map(|(a, b)| CrossoverMisc::default().f32_crossover(rng, a, fit, b, other_fit)),
        )
        .expect("Weights should match")
    }
}

impl Crossover for GenomeEdge {
    fn crossover(&self, rng: &mut dyn RngCore, fit: f32, other: &Self, other_fit: f32) -> Self {
        assert_eq!(self.innov_number, other.innov_number);
        GenomeEdge {
            innov_number: self.innov_number,
            in_node: self.in_node,
            out_node: self.out_node,
            weight: CrossoverMisc::default().f32_crossover(
                rng,
                self.weight,
                fit,
                other.weight,
                other_fit,
            ),
            enabled: CrossoverMisc::default().bernoulli_crossover(
                rng,
                self.enabled,
                fit,
                other.enabled,
                other_fit,
            ),
        }
    }
}
