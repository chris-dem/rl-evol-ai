use itertools::Itertools;
use rand_derive2::RandGen;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, RandGen)]
pub enum Aggregation {
    Sum,
    Max,
    #[default]
    Mean,
    L1NormAvg,
    L2NormAvg,
}

impl Aggregation {
    pub fn apply(&self, a: impl Iterator<Item = f32>) -> f32 {
        match self {
            Aggregation::Sum => a.sum(),
            Aggregation::Max => a.reduce(|a, b| f32::max(a, b)).unwrap_or(0.),
            Aggregation::Mean => {
                let x = a.fold((0., 0), |(acc, cnt), x| (acc + x, cnt + 1));
                x.0 / x.1 as f32
            }
            Aggregation::L2NormAvg => {
                let v = a.collect_vec();
                let alpha = v
                    .iter()
                    .copied()
                    .reduce(|a, b| f32::max(a.abs(), b.abs()))
                    .expect("Should not contain NaN");
                v.iter()
                    .copied()
                    .fold(0., |acc, x| acc + (x / alpha) * (x / alpha))
                    .sqrt()
                    * alpha
                    / v.len() as f32
            }
            Aggregation::L1NormAvg => {
                let p = a.fold((0., 0), |(acc, cnt), x| (acc + x.abs(), cnt + 1));
                p.0 / p.1 as f32
            }
        }
    }
}