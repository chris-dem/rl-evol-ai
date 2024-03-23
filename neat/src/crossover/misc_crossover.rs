use std::f32::consts::E;

use rand::{prelude::Rng, RngCore};

use crate::individual::genome::{activation::Activation, node_list::Activate};

#[derive(Debug, Clone, Copy)]
pub struct CrossoverMisc {
  pub range: f32
}

impl CrossoverMisc {
  pub fn new(range: f32) -> Self {
    let range = range.abs();
    Self { range }
  }
}

const DEFAULT_RANGE : f32 = 1000.;

impl Default for CrossoverMisc {
    fn default() -> Self {
        Self { range: DEFAULT_RANGE }
    }
}

#[inline]
fn generate_weight(max_w: f32, w1: f32, w2: f32) -> f32 {
    let w1 = w1.clamp(-max_w, max_w);
    let w2 = w2.clamp(-max_w, max_w);
    let diff = w1- w2;
    let factor = (diff.powi(2) + E).ln();
        // Current implementation, use linear interpolation and sigmoid to interpolate between points
    Activation::Sigmoid.activate(diff)
            * (factor - factor.recip())
            + factor.recip()
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
        let t = rng.gen::<f32>();
        let t = t.powf(generate_weight(self.range, weight_fst, weight_snd));
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
        let t = rng.gen::<f32>();
        if t.powf(generate_weight(self.range, weight_fst, weight_snd)) < 0.5 {
            item_fst
        } else {
            item_snd
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use super::*;

    proptest! {
      #[test]
      fn on_average_favours_point_first(
        a in any::<f32>(), wa in any::<f32>().prop_filter("Filter range", |e| e.abs() < DEFAULT_RANGE),
        b in any::<f32>(), wb in any::<f32>().prop_filter("Filter range", |e| e.abs() < DEFAULT_RANGE),
      ) {
          let st = CrossoverMisc::default();
          let mut cnt = 0;
          let mut rng = ChaCha8Rng::from_seed(Default::default());
          let n = 10_000;
          for _ in 0..n {
              let f = st.f32_crossover(&mut rng, a, wa, b, wb);
              let dist = (
                (f - a).abs(),
                (f - b).abs(),
              );
              cnt += if (wa - wb).abs() < 1. {
                true
              } else if wa - wb < -1.{
                dist.0 >= dist.1
              } else {
                  dist.0 <= dist.1 
              } as usize
          }
          let avg = cnt as f64 / n as f64;
          assert!(avg >= 0.49, "avg {}", avg); 
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

      proptest!{
          #[test]
          fn test_discrete_weighted(
            wa in any::<f32>().prop_filter("Filter range", |e| e.abs() < DEFAULT_RANGE),
            wb in any::<f32>().prop_filter("Filter range", |e| e.abs() < DEFAULT_RANGE),
          ) {
              let a = Colour::R;
              let b = Colour::B;
              let st = CrossoverMisc::default();
              let mut rng = ChaCha8Rng::from_seed(Default::default());
              let mut map: BTreeMap<Colour, usize> = BTreeMap::new();
              for _ in 0..50_000 {
                  let f = st.bernoulli_crossover::<Colour>(&mut rng, a, wa, b, wb);
                  map.entry(f).and_modify(|e| *e += 1).or_insert(1);
              }
              assert_eq!(map.get(&Colour::G), None);
              if (wa - wb).abs() < 0.5 {
                assert!(*map.get(&Colour::R).unwrap() as f64 / 50_000f64 <= 0.51, "{}",*map.get(&Colour::R).unwrap() as f64); 
              } else if (wa - wb) < -0.5 {
                assert!(*map.get(&Colour::R).unwrap_or(&0) < *map.get(&Colour::B).unwrap());
              } else {
                assert!(*map.get(&Colour::R).unwrap() > *map.get(&Colour::B).unwrap_or(&0));
              }
          }
      }
    }
  }