use rand::RngCore;

use crate::individual::genome::{
    genome::{Genome, OrderedGenomeList},
    node_list::NodeList,
};

use super::misc_crossover::CrossoverMisc;

/// Helper struct to encapsulate the fitness and the genome.
pub struct Item {
    pub item: Genome,
    pub fitness: f32,
}

/// Crossover trait. Will be used mainly by the crossover method to crossover
/// between two genomes.
pub trait Crossover {
    fn crossover(&self, rng: &mut dyn RngCore, fit: f32, other: &Self, other_fit: f32) -> Self;
}

/// Helper trait to define how parent a and parent b be will cross over with each other.
/// Main difference is that this is not attached to the item itself itself.
pub trait CrossoverMethod {
    fn crossover_method(&self, rng: &mut dyn RngCore, parent_a: &Item, parent_b: &Item) -> Genome;
}

/// Helper function two merge two sequences of genomes. This assumes
/// That the two sequencesa are sorted.
/// If the two parents are equal to each other, apply the crossover method.
fn merge<'a, T: Crossover + Ord + 'a + Clone>(
    fst: impl Iterator<Item = &'a T>,
    snd: impl Iterator<Item = &'a T>,
    rng: &mut dyn RngCore,
    fit_fst: f32,
    fit_snd: f32,
) -> Vec<T> {
    let mut ret = Vec::new();
    let mut fst_peek = fst.peekable();
    let mut snd_peek = snd.peekable();
    loop {
        let fst_c = fst_peek.peek();
        let snd_c = snd_peek.peek();
        match (fst_c, snd_c) {
            (Some(a), Some(b)) => match a.cmp(b) {
                std::cmp::Ordering::Less => {
                    ret.push(fst_peek.next().expect("Was peeked").clone());
                }
                std::cmp::Ordering::Greater => {
                    ret.push(snd_peek.next().expect("Was peeked").clone());
                }
                std::cmp::Ordering::Equal => {
                    let fst_el = fst_peek.next().expect("Was peeked");
                    let snd_el = snd_peek.next().expect("Was peeked");
                    ret.push(fst_el.crossover(rng, fit_fst, snd_el, fit_snd));
                }
            },
            _ => break,
        }
    }
    ret.append(&mut (fst_peek.cloned().collect()));
    ret.append(&mut (snd_peek.cloned().collect()));
    ret
}

impl Crossover for NodeList {
    /// Cross over method for lists. The two lists of genomes will be merged fully.
    /// Hidden lists are by definition sorted.
    fn crossover(&self, rng: &mut dyn RngCore, fit: f32, other: &Self, other_fit: f32) -> Self {
        Self::new(
            self.input.clone(),
            self.output.clone(),
            merge(self.hidden.iter(), other.hidden.iter(), rng, fit, other_fit),
        )
    }
}

/// Ordered genome list. Neat crossover algorithm is applied.
impl Crossover for OrderedGenomeList {
    fn crossover(&self, rng: &mut dyn RngCore, fit: f32, other: &Self, other_fit: f32) -> Self {
        Self::new_sorted(merge(self.iter(), other.iter(), rng, fit, other_fit).into_iter())
    }
}

/// Trait to implement the crossover method
#[derive(Clone, Copy)]
pub struct NeatCrossover {
    /// Crossover method for misc calculations (f32, bernoulli).
    pub crossover_misc: CrossoverMisc,
}

impl NeatCrossover {
    pub fn new(crossover_misc: CrossoverMisc) -> Self {
        Self { crossover_misc }
    }
}

impl Default for NeatCrossover {
    fn default() -> Self {
        Self {
            crossover_misc: CrossoverMisc::default(),
        }
    }
}

/// Crossover implementation for neat. Given two genomes
/// crossover node list and genome list. Create node list from the result.
impl CrossoverMethod for NeatCrossover {
    fn crossover_method(
        &self,
        rng: &mut dyn RngCore,
        Item {
            item: item_a,
            fitness: fit_a,
        }: &Item,
        Item {
            item: item_b,
            fitness: fit_b,
        }: &Item,
    ) -> Genome {
        let fit_a = *fit_a;
        let fit_b = *fit_b;
        let new_list = item_a
            .node_list
            .crossover(rng, fit_a, &item_b.node_list, fit_b);
        let new_genome_list = item_a
            .genome_list
            .crossover(rng, fit_a, &item_b.genome_list, fit_b);
        Genome {
            node_list: new_list,
            genome_list: new_genome_list,
        }
    }
}

#[cfg(test)]
mod crossover_tests {
    use super::*;
    use itertools::Itertools;
    use proptest::{array::*, prelude::*};

    #[derive(Debug, Clone, Copy, Hash)]
    struct TestCrossover(pub i32, pub i32);

    impl PartialEq for TestCrossover {
        fn eq(&self, other: &Self) -> bool {
            self.0 == other.0
        }
    }

    impl Eq for TestCrossover {}

    impl PartialOrd for TestCrossover {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.0.cmp(&other.0))
        }
    }

    impl Ord for TestCrossover {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.0.cmp(&other.0)
        }
    }

    impl Crossover for TestCrossover {
        fn crossover(
            &self,
            _rng: &mut dyn RngCore,
            fit: f32,
            other: &Self,
            other_fit: f32,
        ) -> Self {
            if fit > other_fit {
                *self
            } else if fit < other_fit {
                *other
            } else {
                TestCrossover(self.0, self.1.max(other.1))
            }
        }
    }

    proptest! {
      #[test]
      fn test_merge_no_conflict(
        a in any::<f32>(),
        b in any::<f32>(),
        items in (uniform32(any::<(i32, i32)>()), uniform32(any::<(i32, i32)>()))
          .prop_filter("should have unique elements",
          |(el1,el2)| el1.iter().chain(el2.iter()).map(|(a,b)| TestCrossover(*a,*b)).all_unique())
          .prop_map(|(el1,el2)| (el1.into_iter().map(|(a,b)| TestCrossover(a,b)).sorted().collect::<Vec<_>>(), el2.into_iter().map(|(a,b)| TestCrossover(a,b)).sorted().collect::<Vec<_>>()))
      ) {
          let mut rng = rand::thread_rng();
          let (fst, snd) = items;
          let m = merge(fst.iter(), snd.iter(), &mut rng, a, b);
          let v1 = fst.iter().chain(snd.iter()).sorted().copied().collect_vec();

          assert!(m.iter().copied().zip(v1.iter().copied())
            .all(|(a,b)| a == b), "Assertion: {m:?} {v1:?}");
      }

      #[test]
      fn test_merge_no_conflict_unequal_size(
        a in any::<f32>(),
        b in any::<f32>(),
        items in (uniform16(any::<(i32, i32)>()), uniform32(any::<(i32, i32)>()))
          .prop_filter("should have unique elements",
          |(el1,el2)| el1.iter().chain(el2.iter()).map(|(a,b)| TestCrossover(*a,*b)).all_unique())
          .prop_map(|(el1,el2)| (el1.into_iter().map(|(a,b)| TestCrossover(a,b)).sorted().collect::<Vec<_>>(), el2.into_iter().map(|(a,b)| TestCrossover(a,b)).sorted().collect::<Vec<_>>()))
      ) {
          let mut rng = rand::thread_rng();
          let (fst, snd) = items;
          let m = merge(fst.iter(), snd.iter(), &mut rng, a, b);
          let v1 = fst.iter().chain(snd.iter()).sorted().copied().collect_vec();

          assert!(m.iter().copied().zip(v1.iter().copied())
            .all(|(a,b)| a == b), "Assertion: {m:?} {v1:?}");
      }

      #[test]
      fn test_merge_all_conflict(
        a in any::<f32>(),
        b in any::<f32>(),
        items in (uniform4(any::<i32>()), uniform4(any::<i32>()))
          .prop_map(|(el1,el2)| (
            el1.into_iter().enumerate().map(|(ind,a)| TestCrossover(ind as i32,a)).sorted().collect::<Vec<_>>(),
            el2.into_iter().enumerate().map(|(ind,a)| TestCrossover(ind as i32,a)).sorted().collect::<Vec<_>>())
      )) {
          let mut rng = rand::thread_rng();
          let (fst, snd) = items;
          let m = merge(fst.iter(), snd.iter(), &mut rng, a, b);
          let expected = match a.partial_cmp(&b).unwrap() {
            std::cmp::Ordering::Less => snd.clone(),
            std::cmp::Ordering::Greater => fst.clone(),
            std::cmp::Ordering::Equal => fst.iter().zip_eq(snd.iter()).map(|(f,s)| TestCrossover(f.0, f.1.max(s.1))).collect(),
          };
          m.iter().zip_eq(expected.iter()).for_each(|(a,b)| assert_eq!(a.1,b.1));
      }
    }
}
