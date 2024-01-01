use rand::{prelude::Rng, RngCore};

use crate::individual::genome::node_list::{Activate, Activation};

const DEFAULT_RANGE: f32 = 5.;

pub struct CrossoverMisc {
    range_float: f32,
}

impl Default for CrossoverMisc {
    fn default() -> Self {
        Self {
            range_float: DEFAULT_RANGE,
        }
    }
}

impl CrossoverMisc {
    pub fn f32_crossover(
        &self,
        rng: &mut dyn RngCore,
        fst: f32,
        weight_fst: f32,
        snd: f32,
        weight_snd: f32,
    ) -> f32 {
        // Current implementation, use linear interpolation and sigmoid to interpolate between points
        let expo = Activation::Sigmoid.activate(weight_fst - weight_snd)
            * (self.range_float - self.range_float.recip())
            + self.range_float.recip();
        let t = rng.gen::<f32>();
        let t = t.powf(expo);
        fst * (1. - t) + t * snd
    }

    pub fn bernoulli_crossover<T>(
        &self,
        rng: &mut dyn RngCore,
        item_fst: T,
        weight_fst: f32,
        item_snd: T,
        weight_snd: f32,
    ) -> T {
        let expo = Activation::Sigmoid.activate(weight_fst - weight_snd)
            * (self.range_float - self.range_float.recip())
            + self.range_float.recip();
        let t = rng.gen::<f32>();
        if t.powf(expo) < 0.5 {
            item_fst
        } else {
            item_snd
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;

    mod floats {
        use super::*;

        #[test]
        fn on_average_favours_point_first() {
            let a = 10.;
            let f1 = 2.;
            let b = 5.;
            let f2 = -2.;
            let st = CrossoverMisc::default();
            let mut cnt = 0;
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            for _ in 0..10_000 {
                let f = st.f32_crossover(&mut rng, a, f1, b, f2);
                cnt += if (a - f).abs() < (b - f).abs() { 1 } else { -1 };
            }
            let avg = cnt as f32 / 10_000.0;
            assert!(dbg!(avg) > 0.)
        }

        #[test]
        fn on_average_favours_point_second() {
            let a = 10.;
            let f1 = 2.;
            let b = 5.;
            let f2 = 10.;
            let st = CrossoverMisc::default();
            let mut cnt = 0;
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            for _ in 0..10_000 {
                let f = st.f32_crossover(&mut rng, a, f1, b, f2);
                cnt += if (a - f).abs() < (b - f).abs() { 1 } else { -1 };
            }
            let avg = cnt as f32 / 10_000.0;
            assert!(dbg!(avg) < 0.)
        }
    }

    mod items {
        use std::collections::{BTreeMap, HashMap};

        use super::*;

        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
        pub enum Colour {
            R,
            G,
            B,
        }

        #[test]
        fn test_discrete_weighted_first() {
            let a = Colour::R;
            let f1 = 2.;
            let b = Colour::B;
            let f2 = -2.;
            let st = CrossoverMisc::default();
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let mut map: BTreeMap<Colour, usize> = BTreeMap::new();
            for _ in 0..10_000 {
                let f = st.bernoulli_crossover::<Colour>(&mut rng, a, f1, b, f2);
                map.entry(f).and_modify(|e| *e += 1).or_insert(1);
            }
            assert_eq!(map.get(&Colour::G), None);
            assert!(*map.get(&Colour::R).unwrap() > *map.get(&Colour::B).unwrap());
        }

        #[test]
        fn test_discrete_weighted_second() {
            let a = Colour::R;
            let f1 = 2.;
            let b = Colour::B;
            let f2 = 10.;
            let st = CrossoverMisc::default();
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let mut map: BTreeMap<Colour, usize> = BTreeMap::new();
            for _ in 0..10_000 {
                let f = st.bernoulli_crossover::<Colour>(&mut rng, a, f1, b, f2);
                map.entry(f).and_modify(|e| *e += 1).or_insert(1);
            }
            assert_eq!(map.get(&Colour::G), None);
            assert!(*map.get(&Colour::R).unwrap() < *map.get(&Colour::B).unwrap());
        }
    }
}
