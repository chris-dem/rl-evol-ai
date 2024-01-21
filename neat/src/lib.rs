use crossover::crossover::CrossoverMethod;
use individual::{genome::genome::Genome, individual::Individual};
use rand::RngCore;
use selection::selection_trait::SelectionMethod;
use speciation::speciation::{Comparable, SpeciationMethod};

use crate::crossover::crossover::Item;

mod crossover;
mod individual;
mod selection;
mod speciation;
mod mutation;

pub struct GeneticAlgortihm<Spe, Sel> {
    speciation: Spe,
    selection: Sel,
    crossover: Box<dyn CrossoverMethod>,
}

impl<Spe, Sel> GeneticAlgortihm<Spe, Sel>
where
    Spe: SpeciationMethod,
    Sel: SelectionMethod,
{
    pub fn new(spec_method: Spe, sel_method: Sel, cross_method: Box<dyn CrossoverMethod>) -> Self {
        Self {
            speciation: spec_method,
            selection: sel_method,
            crossover: cross_method,
        }
    }

    fn evolve<I>(&self, rng: &mut dyn RngCore, population: &[I]) -> Vec<Genome>
    where
        I: Individual + Comparable,
    {
        assert!(!population.is_empty());
        let s = self.speciation.speciate(population.iter());
        let mut ret = Vec::with_capacity(population.len());
        for sub_pop in s {
            for _ in 0..sub_pop.len() {
                let parent_a = self.selection.select(rng, &sub_pop);
                let parent_b = self.selection.select(rng, &sub_pop);
                let child = self.crossover.crossover_method(
                    rng,
                    &Item {
                        item: parent_a.to_genome(),
                        fitness: parent_a.fitness(),
                    },
                    &Item {
                        item: parent_b.to_genome(),
                        fitness: parent_a.fitness(),
                    },
                );
                todo!("Mutation");
                ret.push(child);
            }
        }
        ret
    }
}
