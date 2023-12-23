pub trait Comparable {
    fn compare(&self, other: &Self) -> f32;
}

pub trait SpeciationMethod {
    fn speciate<'a, C : Comparable>(&self, population: impl Iterator<Item = &'a C>) -> Vec<Vec<&'a C>>;
}

pub struct SpeciationThreshold {
    threshold : f32
}

impl SpeciationThreshold {
    fn new(t : f32) -> Self {
        Self { threshold : t }
    }
}

impl SpeciationMethod for SpeciationThreshold {
    fn speciate<'a, C : Comparable>(&self,mut population: impl Iterator<Item = &'a C>) -> Vec<Vec<&'a C>> {
        let mut ret = Vec::new();
        let mut curr_vec = vec![population.next().expect("At speciate, first element should exist")];
        for el in population {
            if curr_vec[0].compare(el) >= self.threshold {
                curr_vec.push(el);
            } else {
                ret.push(curr_vec);
                curr_vec = vec![el];
            }
        }
        ret.push(curr_vec);
        ret
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_2_PI as HALF_PI;
    
    #[derive(Debug, PartialEq)]
    struct TestIndividual((f32,f32));

    impl Comparable for TestIndividual{
        // cosine comparison
        fn compare(&self, other: &Self) -> f32 {
            let x = self.0;
            let y = other.0;
            let d1 = (x.0 * x.0 + x.1 * x.1).sqrt();
            let d2 = (y.0 * y.0 + y.1 * y.1).sqrt();
            ((x.0 * y.0 + x.1 * y.1) / (d1 * d2)).abs()
        }
    }

    fn generate_from_angle(theta : f32) -> (f32,f32) {
        theta.sin_cos()
    }

    #[test]
    fn test_simple_speciation() {
        let population = vec![
            TestIndividual(generate_from_angle(0.)),
            TestIndividual(generate_from_angle(f32::EPSILON)),
            TestIndividual(generate_from_angle(-f32::EPSILON)),
            TestIndividual(generate_from_angle(HALF_PI)),
            TestIndividual(generate_from_angle(HALF_PI + f32::EPSILON)),
            TestIndividual(generate_from_angle(HALF_PI - f32::EPSILON)),
        ];

        let spec = SpeciationThreshold::new(0.99);
        let v = dbg!(spec.speciate(population.iter()));
        assert_eq!(v.len(), 2);
        assert_eq!(*v[0][0], population[0]);
        assert_eq!(*v[0][1], population[1]);
        assert_eq!(*v[0][2], population[2]);
        assert_eq!(*v[1][0], population[3]);
        assert_eq!(*v[1][1], population[4]);
    }
}