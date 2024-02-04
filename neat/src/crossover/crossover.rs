use rand::RngCore;

use crate::individual::genome::{
    genome::{Genome, OrderedGenomeList},
    node_list::{Node, NodeList},
};

pub struct Item {
    pub item: Genome,
    pub fitness: f32,
}

pub trait Crossover {
    fn crossover(&self, rng: &mut dyn RngCore, fit: f32, other: &Self, other_fit: f32) -> Self;
}

pub trait CrossoverMethod {
    fn crossover_method(&self, rng: &mut dyn RngCore, parent_a: &Item, parent_b: &Item) -> Genome;
}

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
                std::cmp::Ordering::Equal => {
                    ret.push(snd_peek.next().expect("Was peeked").clone());
                }
                std::cmp::Ordering::Greater => {
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
    fn crossover(&self, rng: &mut dyn RngCore, fit: f32, other: &Self, other_fit: f32) -> Self {
        Self {
            input: self.input.clone(),
            output: self.output.clone(),
            hidden: merge(self.hidden.iter(), other.hidden.iter(), rng, fit, other_fit),
        }
    }
}

impl Crossover for OrderedGenomeList {
    fn crossover(&self, rng: &mut dyn RngCore, fit: f32, other: &Self, other_fit: f32) -> Self {
        Self::new_sorted(merge(self.iter(), other.iter(), rng, fit, other_fit).into_iter())
    }
}

pub struct NeatCrossover;

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
mod tests {
    use super::*;
    use rand_chacha::ChaCha8Core;

    #[test]
    fn crossover_genomes() {}
}
