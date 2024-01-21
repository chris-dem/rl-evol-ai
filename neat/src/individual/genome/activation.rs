use errorfunctions::RealErrorFunctions;
use rand_derive2::RandGen;

use super::node_list::Activate;

#[derive(Debug, Clone, Copy, PartialEq, Default, RandGen)]
pub enum Activation {
    Abs,
    Cube,
    Exp,
    Gauss,
    Hat,
    Identity,
    Inv,
    Log,
    #[default]
    Relu,
    Selu(f32, f32),
    Sigmoid,
    Sin,
    Tanh,
    Softplus(f32),
    Gelu,
    Root,
}

impl Activate for Activation {
    fn activate(&self, input: f32) -> f32 {
        match self {
            Activation::Abs => input.abs(),                      // |x|
            Activation::Cube => input * input * input,           // x^3
            Activation::Exp => input.exp(),                      // e^x
            Activation::Gauss => (-(input * input)).exp(),       // e^(-x^2)
            Activation::Hat => (1. - input.abs()).max(0.),       // max(1 - |x|, 0)
            Activation::Identity => input,                       // x
            Activation::Inv => 1. / (input * input + 1.).sqrt(), // 1 / sqrt(x^2 + 1)
            Activation::Log => input.abs().ln(),                 // ln(|x|)
            Activation::Relu => input.max(0.),                   // max(x, 0)
            Activation::Selu(alpha, lambda) => {
                if input > 0. {
                    // lambda * x
                    lambda * input
                } else {
                    lambda * alpha * (input.exp() - 1.) // lambda * alpha * (e^x - 1)
                }
            }
            Activation::Sigmoid => 1. / (1. + (-input).exp()), // 1 / (1 + e^(-x))
            Activation::Sin => input.sin(),                    // sin x
            Activation::Tanh => input.tanh(),                  // tanh x
            Activation::Softplus(beta) => beta.recip() * (1. + (input * beta).exp()).ln(), // (1 / beta) * ln(1. + exp(x * beta))
            Activation::Gelu => ((input as f64 / 2.0_f64.sqrt()).erf() as f32 + 1.) * 0.5 * input, // x/2 (1 + erf(x / sqrt(2)))
            Activation::Root => (input * input + 1.).sqrt(),
        }
    }
}

