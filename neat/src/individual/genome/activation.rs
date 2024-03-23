use errorfunctions::RealErrorFunctions;
use num::{traits::real::Real, Float};
use rand_derive2::RandGen;

use super::node_list::Activate;

#[derive(Debug, Clone, Copy, PartialEq, Default, RandGen)]
pub enum Activation {
    Abs,
    Exp,
    Gauss,
    Hat,
    Identity,
    Inv,
    Log,
    #[default]
    Relu,
    Selu,
    Sigmoid,
    Sin,
    Cos,
    Tanh,
    Softplus(f32),
    Gelu,
    Root,
    Periodic(f32),
}

impl Activate for Activation {
    fn activate(&self, input: f32) -> f32 {
        match self {
            Activation::Abs => input.abs(),                      // |x|
            Activation::Exp => input.min(5.).exp(),              // e^x // Avoid exploding
            Activation::Gauss => (-(input * input)).exp(),       // e^(-x^2)
            Activation::Hat => (1. - input.abs()).max(0.),       // max(1 - |x|, 0)
            Activation::Identity => input,                       // x
            Activation::Inv => 1. / (input * input + 1.).sqrt(), // 1 / sqrt(x^2 + 1)
            Activation::Log => input.abs().ln_1p(),                 // ln(|x| + 1)
            Activation::Relu => input.max(0.),                   // max(x, 0)
            Activation::Selu => {
                let lambda = 1.0507009873554804934193349852946;
                let alpha =  1.6732632423543772848170429916717;
                if input >= 0. {
                    // lambda * x
                    lambda * input
                } else {
                    lambda * alpha * (input.exp() - 1.) // lambda * alpha * (e^x - 1)
                }
            },
            Activation::Sigmoid => (1. + (-input).exp()).recip(), // 1 / (1 + e^(-x))
            Activation::Sin => input.sin(),                    // sin x
            Activation::Cos => input.cos(),                    // cos x
            Activation::Tanh => input.tanh(),                  // tanh x
            Activation::Softplus(beta) => beta.recip() * (-(beta * input).abs()).exp().ln_1p(), // (1 / beta) * ln(1. + exp(x * beta)) (Stable)
            Activation::Gelu => ((input as f64 / 2.0_f64.sqrt()).erf() as f32 + 1.) * 0.5 * input, // x/2 (1 + erf(x / sqrt(2)))
            Activation::Root => (input * input + 1.).sqrt(), // sqrt(x^2 + 1)
            Activation::Periodic(p) => (input - p * (input / (p + f32::EPSILON)).floor()) - p / 2. // x - p * floor (x/(p + c)) - p/2
        }
    }
}

