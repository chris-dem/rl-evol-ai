use itertools::Itertools;
use rand::RngCore;

use crate::individual::genome::{
    activation::Activation,
    aggregation::Aggregation,
    clamp::Clamp,
    genome::GenomeEdge,
    node_list::{Config, Node},
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
                    (Some(a), None) => CrossoverMisc::default().bernoulli_crossover(
                        rng,
                        Some(a),
                        fit,
                        None,
                        other_fit,
                    ),
                    (None, Some(b)) => CrossoverMisc::default().bernoulli_crossover(
                        rng,
                        None,
                        fit,
                        Some(b),
                        other_fit,
                    ),
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

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use proptest::prelude::*;

    use super::*;

    mod clamp_crossover {

        use approx::{Relative, RelativeEq};

        use super::*;

        fn compare_relative(l : f32, h: f32, prod : f32) -> (f32,f32) {
            let diff_1 = (prod - l).abs();
            let diff_2 = (prod - h).abs();
            (diff_1, diff_2)
        }

        proptest! {
            #[test]
            fn test_node_clamp(
                lims_1 in any::<(f32,f32)>().prop_filter("First limits", |a| a.0 < a.1),
                lims_2 in any::<(f32,f32)>().prop_filter("Second limits", |a| a.0 < a.1),
                perf_1 in any::<f32>(), perf_2 in any::<f32>(),
            ) {
                let mut rng = ChaCha8Rng::seed_from_u64(32);
                let clamp_1 = Clamp::new(Some(lims_1.0), Some(lims_1.1)).unwrap();
                let clamp_2 = Clamp::new(Some(lims_2.0), Some(lims_2.1)).unwrap();
                let mut count_min = 0;
                let mut count_max = 0;
                for _ in 0..1_000 {
                    let res = clamp_1.crossover(&mut rng, perf_1, &clamp_2, perf_2);
                    let res_min = compare_relative(clamp_1.min_limit.unwrap(), clamp_2.min_limit.unwrap(), res.min_limit.unwrap());
                    let res_max = compare_relative(clamp_1.max_limit.unwrap(), clamp_2.max_limit.unwrap(), res.max_limit.unwrap());
                    count_min += {
                        if Relative::default().eq(&perf_1,  &perf_2) { // if they are equal
                            (lims_1.0.min(lims_2.0) <= res.min_limit.unwrap() && lims_1.0.max(lims_2.0) >= res.min_limit.unwrap()) as u8
                        } else {
                            ((perf_1 <= perf_2) as u8 ^ (res_min.0 > res_min.1) as u8) | (res_min.0 == res_min.1) as u8
                        }
                    } as usize;
                    count_max += {
                        if Relative::default().eq(&perf_1,  &perf_2) { // if they are equal
                            (lims_1.1.min(lims_2.1) <= res.max_limit.unwrap() && lims_1.1.max(lims_2.1) >= res.max_limit.unwrap()) as u8
                        } else {
                            ((perf_1 <= perf_2) as u8 ^ (res_max.0 > res_max.1) as u8) | (res_max.0 == res_max.1) as u8
                        }
                    } as usize;
                }
                assert!(dbg!(count_min) as f64 / 1_000f64 > 0.5_f64);
                assert!(dbg!(count_max) as f64 / 1_000f64 > 0.5_f64);
            }
        }

        // #[test]
        // fn test_node_clamp() {
            
        // }
    }
}
