use individual::individual::Individual;
use rand::RngCore;
use selection::selection_trait::SelectionMethod;
use speciation::speciation::{SpeciationMethod, Comparable};

mod selection;
mod speciation;
mod individual;
mod crossover;


pub struct GeneticAlgortihm<Spe,Sel> {
    speciation : Spe,
    selection : Sel,
}


impl<Spe,Sel> GeneticAlgortihm<Spe, Sel>
    where Spe : SpeciationMethod, Sel : SelectionMethod
{
    pub fn new(spec_method : Spe, sel_method : Sel) -> Self {
        Self {
            speciation : spec_method,
            selection : sel_method
        }
    }

    fn evolve<I>(&self, rng : &mut dyn RngCore, population : &[I]) -> Vec<I> 
        where I: Individual + Comparable
    {
        assert!(!population.is_empty());
        let s = self.speciation.speciate(population.iter());
        let mut ret = Vec::new();
        for sub_pop in s { 
            for el in 0..sub_pop.len() {
                let parent_a = self.selection.select(rng, &sub_pop);
                let parent_b = self.selection.select(rng, &sub_pop);
                todo!("Crossover");
                todo!("Mutation");
            }
        }
        ret
    }
}