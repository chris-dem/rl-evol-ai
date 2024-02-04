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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    
    #[test]
    fn test_aggregations() {
        const DATA : [f32; 5] = [1., 2., 3., 4., 5.];
        assert_relative_eq!(Aggregation::Sum.apply(DATA.iter().copied()), 15.);
        assert_relative_eq!(Aggregation::Max.apply(DATA.iter().copied()), 5.);
        assert_relative_eq!(Aggregation::Mean.apply(DATA.iter().copied()), 3.);
        assert_relative_eq!(Aggregation::L2NormAvg.apply(DATA.iter().copied()), DATA.iter().map(|x| x * x).sum::<f32>().sqrt() / DATA.len() as f32);
        assert_relative_eq!(Aggregation::L1NormAvg.apply(DATA.iter().copied()), DATA.iter().map(|x| x.abs()).sum::<f32>() / DATA.len() as f32);
    }
}