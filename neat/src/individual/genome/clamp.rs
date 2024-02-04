use super::node_list::Activate;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Clamp {
    pub min_limit: Option<f32>,
    pub max_limit: Option<f32>,
}

const MIN_CLAMP: f32 = -5.;
const MAX_CLAMP: f32 = 5.;

impl Default for Clamp {
    fn default() -> Self {
        Self {
            min_limit: Some(MIN_CLAMP),
            max_limit: Some(MAX_CLAMP),
        }
    }
}

impl Clamp {
    pub fn new(min_limit: Option<f32>, max_limit: Option<f32>) -> Option<Self> {
        match (min_limit, max_limit) {
            (Some(a), Some(b)) => {
                if a >= b {
                    None
                } else {
                    Some(Clamp {
                        min_limit,
                        max_limit,
                    })
                }
            }
            (a, b) => Some(Clamp {
                min_limit: a,
                max_limit: b,
            }),
        }
    }
}

impl Activate for Clamp {
    fn activate(&self, input: f32) -> f32 {
        let input = if let Some(m) = self.max_limit {
            input.min(m)
        } else {
            input
        };
        let input = if let Some(m) = self.min_limit {
            input.max(m)
        } else {
            input
        };
        return input;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
      #[test]
      fn rand_number(a in any::<f32>()) {
        let clamp = Clamp::default();
        let res = clamp.activate(a);
        prop_assert!(res >= -5. && res <= 5.);
      }

      #[test]
      fn lower_rand_number(a in any::<f32>()) {
        let clamp = Clamp::new(None, Some(2.)).unwrap();
        let res = clamp.activate(a);
        prop_assert!(res <= 2.);
      }

      #[test]
      fn higher_rand_number(a in any::<f32>()) {
        let clamp = Clamp::new(Some(-2.), None).unwrap();
        let res = clamp.activate(a);
        prop_assert!(res >= -2.);
      }

      #[test]
      fn check_clamp(a in -10.0f32..10.0f32, b in -10.0f32..10.0f32) {
        let clamp = Clamp::new(Some(a), Some(b));
        match a.partial_cmp(&b).unwrap() {
          std::cmp::Ordering::Less => prop_assert!(clamp.is_some()),
          _ => prop_assert!(clamp.is_none())
        }
      }
    }
}
