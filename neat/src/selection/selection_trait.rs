use itertools::Itertools;
use rand::{seq::SliceRandom, RngCore};

use crate::individual::individual::Individual;

pub trait SelectionMethod {
    fn select<'a, 'b , I>(&self, rng: &mut dyn RngCore, population: &'a [&'b I]) -> &'b I
    where
        I: Individual;
}

#[derive(Default)]
pub struct RoulleteSelection;

impl RoulleteSelection {
    pub fn new() -> Self {
        Self
    }
}

impl SelectionMethod for RoulleteSelection {
    fn select<'a, 'b , I>(&self, rng: &mut dyn RngCore, population: &'a [&'b I]) -> &'b I
    where
        I: Individual,
    {
        let weights = population.iter().map(|s| s.fitness()).collect_vec();
        let total_weight = weights.iter().sum::<f32>();
        population
            .choose_weighted(rng, |el| el.fitness() / total_weight)
            .expect("should not surpass")
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;

    #[cfg(test)]
    #[derive(Clone, Debug)]
    pub struct TestIndividual {
        fitness: f32,
    }

    #[cfg(test)]
    impl TestIndividual {
        pub fn new(fitness: f32) -> Self {
            Self { fitness }
        }
    }

    #[cfg(test)]
    impl Individual for TestIndividual {
        fn fitness(&self) -> f32 {
            self.fitness
        }
    }

    #[test]
    fn rand_tests() {
        let method = RoulleteSelection::new();
        let mut rng = ChaCha8Rng::from_seed(Default::default());

        let population = vec![
            TestIndividual::new(2.0),
            TestIndividual::new(1.0),
            TestIndividual::new(4.0),
            TestIndividual::new(3.0),
        ];

        let mut actual_histogram = BTreeMap::new();

        //               there is nothing special about this thousand;
        //          v--v a number as low as fifty might do the trick, too
        for _ in 0..10000 {
            let fitness = method.select(&mut rng, &population.iter().collect_vec()).fitness() as i32;

            *actual_histogram.entry(fitness).or_insert(0) += 1;
        }

        let els = actual_histogram.iter().sorted_by(|(_,a2),(_,b2)| a2.cmp(b2)).map(|(x,_)| (*x)).collect_vec();
        assert_eq!(els, vec![1,2,3,4]);
    }
}
