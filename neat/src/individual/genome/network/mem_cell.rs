use crate::individual::genome::node_list::{Activate, Node};

#[derive(Debug)]
pub struct MemoryCell {
    node: Node,
    current: f32,
    prev: f32,
    bias: f32,
    current_data: Vec<f32>,
    activated: bool,
    pub passed: bool,
}

impl MemoryCell {
    pub fn default(node: Node) -> Self {
        Self::new(node, 0.)
    }

    pub fn new(node: Node, bias: f32) -> Self {
        MemoryCell {
            node,
            current: 0.,
            prev: 0.,
            bias: bias,
            current_data: Vec::new(),
            activated: false,
            passed: false,
        }
    }
    pub fn get_node(&self) -> Node {
        self.node
    }

    pub fn activate(&mut self, pass_flag: bool) {
        if self.activated == pass_flag {
            return; // If already activated, do not activate again
        }
        let config = self.node.config;
        let agg_data = config.aggregation.apply(self.current_data.iter().copied()) + self.bias;
        let current = config.clamp.activate(config.activation.activate(agg_data));
        self.prev = self.current;
        self.current = current;
        self.activated = pass_flag;
        self.current_data.clear();
    }

    pub fn get_current_output(&self, pass_flag: bool) -> Option<f32> {
        if self.activated == pass_flag {
            Some(self.current)
        } else {
            None
        }
    }

    pub fn get_previous_output(&self, pass_flag: bool) -> f32 {
        if self.activated == pass_flag {
            self.prev
        } else {
            self.current
        }
    }

    pub fn append_input(&mut self, input: f32) {
        self.current_data.push(input);
    }
}

#[derive(Debug)]
pub enum MemoryCellType {
    Input { node: Node, cell_value: f32 },
    Activation(MemoryCell),
}

impl PartialEq for MemoryCellType {
    fn eq(&self, other: &Self) -> bool {
        self.get_node().into_level() == other.get_node().into_level()
    }
}

impl Eq for MemoryCellType {}

impl PartialOrd for MemoryCellType {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.get_node()
            .into_level()
            .partial_cmp(&other.get_node().into_level())
    }
}

impl Ord for MemoryCellType {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.get_node()
            .into_level()
            .cmp(&other.get_node().into_level())
    }
}

impl MemoryCellType {
    pub fn get_node(&self) -> Node {
        match self {
            MemoryCellType::Input { node, .. } => *node,
            MemoryCellType::Activation(MemoryCell { node, .. }) => *node,
        }
    }

    pub fn was_not_passed_set(&mut self, pass_flag: bool) -> bool {
        match self {
            MemoryCellType::Input { .. } => true,
            MemoryCellType::Activation(MemoryCell { passed, .. }) => {
                let prev = *passed != pass_flag;
                *passed = pass_flag;
                prev
            }
        }
    }

    pub fn propagate_input(&mut self, input: f32) {
        match self {
            MemoryCellType::Input { cell_value, .. } => *cell_value = input,
            MemoryCellType::Activation(c) => c.append_input(input),
        }
    }

    pub fn activate(&mut self, pass_flag: bool) {
        match self {
            MemoryCellType::Activation(c) => c.activate(pass_flag),
            _ => (),
        }
    }

    pub fn get_previous_output(&self, pass_flag: bool) -> f32 {
        match self {
            MemoryCellType::Input { cell_value, .. } => *cell_value, // should never occur
            MemoryCellType::Activation(c) => c.get_previous_output(pass_flag),
        }
    }

    pub fn get_current_output(&self, pass_flag: bool) -> Option<f32> {
        match self {
            MemoryCellType::Input { cell_value, .. } => Some(*cell_value), // should never occur
            MemoryCellType::Activation(c) => c.get_current_output(pass_flag),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num::rational::Ratio;

    mod mem_cell_tests {
        use crate::individual::genome::{
            activation::Activation, aggregation::Aggregation, clamp::Clamp, node_list::Config,
        };
        use approx::assert_relative_eq;
        use proptest::prelude::*;

        use super::*;

        fn generate_default_node() -> Node {
            Node {
                node_id: 0,
                config: Config {
                    activation: Activation::Gauss,
                    aggregation: crate::individual::genome::aggregation::Aggregation::L2NormAvg,
                    clamp: Clamp {
                        min_limit: Some(-10.),
                        max_limit: Some(10.),
                    },
                },
                level: Ratio::new(0, 1),
            }
        }

        proptest! {
            #[test]
            fn test_activation(a in proptest::array::uniform32(-1.0f32..1.0)) {
                let mut mem_cell = MemoryCell::new(generate_default_node(), 1.);
                for el in a.iter().copied() {
                    mem_cell.append_input(el);
                }
                mem_cell.activate(true);
                let curr_val = mem_cell.get_current_output(true).unwrap();
                let exp = Activation::Gauss.activate(Aggregation::L2NormAvg.apply(a.iter().copied()) + 1.).clamp(-10., 10.);
                assert_relative_eq!(curr_val, exp);
            }

            #[test]
            fn test_two_activation(a in proptest::array::uniform32(-1.0f32..1.0), b in proptest::array::uniform32(-1.0f32..1.0)) {
                let pass = true;
                let mut mem_cell = MemoryCell::new(generate_default_node(), 1.);
                for el in a.iter().copied() {
                    mem_cell.append_input(el);
                }
                mem_cell.activate(pass);
                let curr_val = mem_cell.get_current_output(pass).unwrap();
                let exp_a = Activation::Gauss.activate(Aggregation::L2NormAvg.apply(a.iter().copied()) + 1.).clamp(-10., 10.);
                assert_relative_eq!(curr_val, exp_a);
                let pass = !pass;
                for el in b.iter().copied() {
                    mem_cell.append_input(el);
                }
                mem_cell.activate(pass);
                let curr_val = mem_cell.get_current_output(pass).unwrap();
                let prev_val = mem_cell.get_previous_output(pass);
                let exp = Activation::Gauss.activate(Aggregation::L2NormAvg.apply(b.iter().copied()) + 1.).clamp(-10., 10.);
                assert_relative_eq!(curr_val, exp);
                assert_relative_eq!(prev_val, exp_a);
            }
        }
    }
}
