use dfdx::prelude::{
    modules::{self},
    Bias2D, Conv2D, DropoutOneIn, Flatten2D, GeLU, Linear, MaxPool2D, SplitInto,
};

use crate::agent::consts::*;

pub type BiasedConv<
    const IN_CHAN: usize,
    const OUT_CHAN: usize,
    const KERNEL_SIZE: usize = 4,
    const STRIDE: usize = 2,
    const PADDING: usize = 3,
> = (
    Conv2D<IN_CHAN, OUT_CHAN, KERNEL_SIZE, STRIDE, PADDING>,
    Bias2D<OUT_CHAN>,
);
pub type BiasedConvModel<
    const IN_CHAN: usize,
    const OUT_CHAN: usize,
    E,
    D,
    const KERNEL_SIZE: usize = 4,
    const STRIDE: usize = 2,
    const PADDING: usize = 3,
> = (
    modules::Conv2D<IN_CHAN, OUT_CHAN, KERNEL_SIZE, STRIDE, PADDING, E, D>,
    modules::Bias2D<OUT_CHAN, E, D>,
);
pub type LayerA<const IN_CHAN: usize, const OUT_CHAN: usize> = (
    (BiasedConv<IN_CHAN, OUT_CHAN>, GeLU),
    (BiasedConv<OUT_CHAN, OUT_CHAN>, GeLU),
    MaxPool2D<2, 2>,
);
pub type LayerAModel<const IN_CHAN: usize, const OUT_CHAN: usize, E, D> = (
    (BiasedConvModel<IN_CHAN, OUT_CHAN, E, D>, GeLU),
    (BiasedConvModel<OUT_CHAN, OUT_CHAN, E, D>, GeLU),
    MaxPool2D<2, 2>,
);
pub type LayerB<const IN_CHAN: usize, const OUT_CHAN: usize> = (
    (BiasedConv<IN_CHAN, OUT_CHAN>, GeLU),
    (BiasedConv<OUT_CHAN, OUT_CHAN>, GeLU),
    (BiasedConv<OUT_CHAN, OUT_CHAN>, GeLU),
    MaxPool2D<2, 2>,
);
pub type LayerBModel<const IN_CHAN: usize, const OUT_CHAN: usize, E, D> = (
    (BiasedConvModel<IN_CHAN, OUT_CHAN, E, D>, GeLU),
    (BiasedConvModel<OUT_CHAN, OUT_CHAN, E, D>, GeLU),
    (BiasedConvModel<OUT_CHAN, OUT_CHAN, E, D>, GeLU),
    MaxPool2D<2, 2>,
);

pub type VggModel<E, D> = (
    (LayerAModel<1, L1, E, D>, LayerAModel<L1, L2, E, D>),
    (LayerBModel<L2, L3, E, D>, LayerBModel<L3, L3, E, D>),
    // LayerBModel<L3, L4, E, D>,
    // Added Output block
    (
        Flatten2D,
        (modules::Linear<AFTER_FLAT, VGG_FC, E, D>, GeLU),
        (modules::Linear<VGG_FC, VGG_FC, E, D>, GeLU),
    ),
);

pub type Vgg = (
    (LayerA<1, L1>, LayerA<L1, L2>),
    (LayerB<L2, L3>, LayerB<L3, L3>),
    // LayerB<L3, L4>,
    // Added Output block
    (
        Flatten2D,
        (Linear<AFTER_FLAT, VGG_FC>, GeLU),
        (Linear<VGG_FC, VGG_FC>, GeLU),
    ),
);

pub type ActorLinear = (DropoutOneIn<4>, Linear<VGG_FC, ACTION>);
pub type ActorLinearModel<E, D> = (
    modules::DropoutOneIn<4>,
    modules::Linear<VGG_FC, ACTION, E, D>,
);
pub type Actor = (Vgg, ActorLinear);
pub type ActorModel<E, D> = (VggModel<E, D>, ActorLinearModel<E, D>);

pub type CriticLinear = (DropoutOneIn<4>, Linear<VGG_FC, 1>);
pub type CriticLinearModel<E, D> = (modules::DropoutOneIn<4>, modules::Linear<VGG_FC, 1, E, D>);

pub type Critic = (Vgg, CriticLinear);
pub type CriticModel<E, D> = (VggModel<E, D>, CriticLinearModel<E, D>);

pub type Network = (
    DropoutOneIn<5>,
    (Vgg, SplitInto<(ActorLinear, CriticLinear)>),
);
pub type NetworkModel<E, D> = (
    modules::DropoutOneIn<5>,
    (
        VggModel<E, D>,
        SplitInto<(ActorLinearModel<E, D>, CriticLinearModel<E, D>)>,
    ),
);

pub type DQN = (DropoutOneIn<5>, (Vgg, ActorLinear));
pub type DQNModel<E, D> = (
    modules::DropoutOneIn<5>,
    (VggModel<E, D>, ActorLinearModel<E, D>),
);
