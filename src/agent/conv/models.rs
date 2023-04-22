use dfdx::prelude::{
    modules::{self},
    BatchNorm2D, Bias2D, Conv2D, DropoutOneIn, Flatten2D, GeLU, Linear, MaxPool2D, SplitInto,
};

use crate::agent::consts::*;

pub type BiasedConv<const IN_CHAN: usize, const OUT_CHAN: usize> = (
    Conv2D<IN_CHAN, OUT_CHAN, KERNEL_SIZE, STRIDE, PADDING>,
    Bias2D<OUT_CHAN>,
);
pub type BiasedConvModel<const IN_CHAN: usize, const OUT_CHAN: usize, E, D> = (
    modules::Conv2D<IN_CHAN, OUT_CHAN, KERNEL_SIZE, STRIDE, PADDING, E, D>,
    modules::Bias2D<OUT_CHAN, E, D>,
);
pub type LayerA<const IN_CHAN: usize, const OUT_CHAN: usize> = (
    (BiasedConv<IN_CHAN, OUT_CHAN>, GeLU),
    MaxPool2D<2, 2>,
    (BiasedConv<OUT_CHAN, OUT_CHAN>, GeLU),
    MaxPool2D<2, 2>,
);
pub type LayerAModel<const IN_CHAN: usize, const OUT_CHAN: usize, E, D> = (
    (BiasedConvModel<IN_CHAN, OUT_CHAN, E, D>, GeLU),
    MaxPool2D<2, 2>,
    (BiasedConvModel<OUT_CHAN, OUT_CHAN, E, D>, GeLU),
    MaxPool2D<2, 2>,
);
pub type LayerB<const IN_CHAN: usize, const OUT_CHAN: usize> = (
    (BiasedConv<IN_CHAN, OUT_CHAN>, GeLU),
    MaxPool2D<2, 2>,
    (BiasedConv<OUT_CHAN, OUT_CHAN>, GeLU),
    MaxPool2D<2, 2>,
    (BiasedConv<OUT_CHAN, OUT_CHAN>, GeLU),
    MaxPool2D<2, 2>,
);
pub type LayerBModel<const IN_CHAN: usize, const OUT_CHAN: usize, E, D> = (
    (BiasedConvModel<IN_CHAN, OUT_CHAN, E, D>, GeLU),
    MaxPool2D<2, 2>,
    (BiasedConvModel<OUT_CHAN, OUT_CHAN, E, D>, GeLU),
    MaxPool2D<2, 2>,
    (BiasedConvModel<OUT_CHAN, OUT_CHAN, E, D>, GeLU),
    MaxPool2D<2, 2>,
);
// WITH BATCH NORM LAYER

pub type LayerABn<const IN_CHAN: usize, const OUT_CHAN: usize> = (
    (BiasedConv<IN_CHAN, OUT_CHAN>, BatchNorm2D<OUT_CHAN>, GeLU),
    MaxPool2D<2, 2>,
);

pub type LayerABnModel<const IN_CHAN: usize, const OUT_CHAN: usize, E, D> = (
    (
        BiasedConvModel<IN_CHAN, OUT_CHAN, E, D>,
        modules::BatchNorm2D<OUT_CHAN, E, D>,
        GeLU,
    ),
    MaxPool2D<2, 2>,
);

pub type LayerBBn<const IN_CHAN: usize, const OUT_CHAN: usize> = (
    (BiasedConv<IN_CHAN, OUT_CHAN>, BatchNorm2D<OUT_CHAN>, GeLU),
    (BiasedConv<OUT_CHAN, OUT_CHAN>, BatchNorm2D<OUT_CHAN>, GeLU),
    MaxPool2D<2, 2>,
);
pub type LayerBBnModel<const IN_CHAN: usize, const OUT_CHAN: usize, E, D> = (
    (
        BiasedConvModel<IN_CHAN, OUT_CHAN, E, D>,
        modules::BatchNorm2D<OUT_CHAN, E, D>,
        GeLU,
    ),
    (
        BiasedConvModel<IN_CHAN, OUT_CHAN, E, D>,
        modules::BatchNorm2D<OUT_CHAN, E, D>,
        GeLU,
    ),
    MaxPool2D<2, 2>,
);

pub type VggModel<E, D> = (
    (LayerABnModel<1, L1, E, D>, DropoutOneIn<5>),
    (LayerABnModel<L1, L2, E, D>, DropoutOneIn<5>),
    (LayerABnModel<L2, L3, E, D>, DropoutOneIn<5>),
    (LayerABnModel<L3, L4, E, D>, DropoutOneIn<5>),
    // Added Output block
    (
        Flatten2D,
        (modules::Linear<AFTER_FLAT, VGG_FC, E, D>, GeLU),
        (modules::Linear<VGG_FC, VGG_FC, E, D>, GeLU),
    ),
);

pub type Vgg = (
    (LayerABn<1, L1>, DropoutOneIn<5>),
    (LayerABn<L1, L2>, DropoutOneIn<5>),
    (LayerABn<L2, L3>, DropoutOneIn<5>),
    (LayerABn<L3, L4>, DropoutOneIn<5>),
    // LayerB<L3, L4>,
    // Added Output block
    (
        Flatten2D,
        (Linear<AFTER_FLAT, VGG_FC>, GeLU),
        (Linear<VGG_FC, VGG_FC>, GeLU),
    ),
);

pub type ActorLinear = (DropoutOneIn<5>, Linear<VGG_FC, ACTION>);
pub type ActorLinearModel<E, D> = (
    modules::DropoutOneIn<5>,
    modules::Linear<VGG_FC, ACTION, E, D>,
);
pub type Actor = (Vgg, ActorLinear);
pub type ActorModel<E, D> = (VggModel<E, D>, ActorLinearModel<E, D>);

pub type CriticLinear = (DropoutOneIn<5>, Linear<VGG_FC, 1>);
pub type CriticLinearModel<E, D> = (modules::DropoutOneIn<5>, modules::Linear<VGG_FC, 1, E, D>);

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

pub type DQN = (Vgg, ActorLinear);
pub type DQNModel<E, D> = (VggModel<E, D>, ActorLinearModel<E, D>);
