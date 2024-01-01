use super::genome::genome::Genome;

pub trait Individual {
    fn fitness(&self) -> f32;
    fn to_genome(&self) -> Genome;
}
