#![feature(generic_const_exprs)]

use binance::{agent::{conv::models::*, consts::{IMAGE_WIDTH, IMAGE_HEIGHT}}, utils::to_array};
use dfdx::prelude::*;

const IMAGE_VEC_LEN: usize = IMAGE_WIDTH * IMAGE_HEIGHT * 1;
// #[cfg(feature = "nightly")]
fn main() {
    let dev: Cpu = Cpu::default();
    let img = image::open("data/stock/0.png").unwrap();
    let vec: Vec<f32> = img
        .to_luma8()
        .to_vec()
        .iter()
        .map(|x| (*x as f32) / 255.0)
        .collect();
    let expected = IMAGE_VEC_LEN;
    println!("{:?} == {} ?", vec.len(), expected);
    let t: [f32; IMAGE_VEC_LEN] = to_array(vec);
    let t: Tensor<(Const<1>, Const<1>, Const<IMAGE_WIDTH>, Const<IMAGE_HEIGHT>), f32, Cpu> =
        dev.tensor(t).reshape();

    let actor: ActorModel<f32, Cpu> = dev.build_module::<Actor, f32>();
    let x = actor.forward(t.clone());
    println!("{:#?}", x.array());
    let critic: CriticModel<f32, Cpu> = dev.build_module::<Critic, f32>();
    let x = critic.forward(t);
    println!("{:#?}", x.array());
}
// #[cfg(not(feature = "nightly"))]
// fn main() {
//     panic!("Run with +nightly");
// }

// pub mod builder {
//     pub struct Vgg11;
// }
// impl<E, D> BuildOnDevice<D, E> for builder::Vgg11
// where
//     E: Dtype,
//     D: Device<E>,
//     Vgg11<D, E>: BuildModule<D, E>,
// {
//     type Built = Vgg11<D, E>;
//     fn build_on_device(device: &D) -> Self::Built {
//         Self::try_build_on_device(device).unwrap()
//     }

//     fn try_build_on_device(device: &D) -> Result<Self::Built, <D>::Err> {
//         Self::Built::try_build(device)
//     }
// }
// const IN_CHAN: usize = 64;
// const OUT_CHAN: usize = 512;
// const KERNEL: usize = 2;
// const STRIDE: usize = 1;
// const PADDING: usize = 1;
// struct Vgg11<E, D>
// where
//     D: DeviceStorage + Device<E>,
//     E: Dtype,
// {
//     conv1: (
//         (Conv2D<64, 64, KERNEL, STRIDE, PADDING, E, D>, ReLU),
//         MaxPool2D<2, 2, 0>,
//     ),
//     conv2: (
//         (Conv2D<64, 128, KERNEL, STRIDE, PADDING, E, D>, ReLU),
//         MaxPool2D<2, 2, 0>,
//     ),
//     conv3: (
//         (Conv2D<128, 512, KERNEL, STRIDE, PADDING, E, D>, ReLU),
//         MaxPool2D<2, 2, 0>,
//     ),
//     conv4: (
//         (Conv2D<512, 512, KERNEL, STRIDE, PADDING, E, D>, ReLU),
//         MaxPool2D<2, 2, 0>,
//     ),
//     conv5: (
//         (Conv2D<512, 512, KERNEL, STRIDE, PADDING, E, D>, ReLU),
//         MaxPool2D<2, 2, 0>,
//     ),
//     conv6: (
//         (Conv2D<512, 512, KERNEL, STRIDE, PADDING, E, D>, ReLU),
//         MaxPool2D<2, 2, 0>,
//     ),
//     conv7: (
//         (Conv2D<512, 512, KERNEL, STRIDE, PADDING, E, D>, ReLU),
//         MaxPool2D<2, 2, 0>,
//     ),
//     conv8: (
//         (Conv2D<512, 512, KERNEL, STRIDE, PADDING, E, D>, ReLU),
//         MaxPool2D<2, 2, 0>,
//     ),
//     fc1: (Linear<4096, 4096, E, D>, ReLU),
//     fc2: (Linear<4096, 4096, E, D>, ReLU),
//     out: Linear<4096, 3, E, D>,
// }
// type ImageInput<const B: usize, const W: usize, E, D> =
//     Tensor<(Const<B>, Const<W>, Const<W>), E, D>;

// //ImageInput<W, H, E, D>
// #[cfg(feature = "nightly")]
// impl<const B: usize, const W: usize, E, D, Img> Module<Img> for Vgg11<E, D>
// where
//     D: DeviceStorage + Device<E> + HasErr,
//     E: Dtype,
//     Img: TryConv2D<ImageInput<B, W, E, D>> + Sized,
//     Vgg11<E, D>: Sized,
// {
//     type Output = Linear<4096, 3, E, D>;

//     type Error = <D as HasErr>::Err;

//     fn try_forward(&self, input: Img) -> Result<Self::Output, Self::Error> {
//         let x = self.conv1.forward(input);
//         let x = self.conv2.forward(x);
//         let x = self.conv3.forward(x);
//         let x = self.conv4.forward(x);
//         let x = self.conv5.forward(x);
//         let x = self.conv6.forward(x);
//         let x = self.conv7.forward(x);
//         let x = self.conv8.forward(x);
//         let x = x.try_reshape()?;
//         let x = self.fc1.forward(x);
//         let x = self.fc2.forward(x);
//         self.out.try_forward(x)
//     }

//     fn forward(&self, input: Img) -> Self::Output {
//         self.try_forward(input).unwrap()
//     }
// }
