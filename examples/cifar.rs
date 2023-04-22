#![feature(generic_const_exprs)]

use dfdx::optim::{Adam, AdamConfig, WeightDecay};
#[allow(incomplete_features)]
// use binance::agent::conv::models::LayerA;
use dfdx::{
    prelude::*,
    tensor::{Cuda, Tensor, TensorFrom},
};
use rayon::prelude::*;
#[allow(unused)]
use std::{error::Error, fs::File, io::Read, path::Path, time::Instant};
use yata::{methods::SMA, prelude::Method};
const WIDTH: usize = 32;
const HEIGHT: usize = 32;
const CH: usize = 3;
const BUF_LEN: usize = WIDTH * HEIGHT * CH;
const BATCH_SIZE: usize = 256;
const EPISODES: usize = 200;
const SMA_PERIOD: u64 = 5;
const LEARNING_RATE: f32 = 1.9e-3;
const LEARNING_RATE_DECAY: f32 = 0.995;

struct Cifar10Image {
    label: Label,
    data: Vec<u8>,
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
#[repr(u8)]
#[allow(unused, non_camel_case_types)]
enum Label {
    airplane = 0,
    automobile,
    bird,
    cat,
    deer,
    dog,
    frog,
    horse,
    ship,
    truck,
}
fn main() -> Result<(), Box<dyn Error>> {
    let mut dataset = Vec::new();
    for i in 1..=5 {
        let filepath = format!("data/cifar-10-batches-bin/data_batch_{}.bin", i);
        let mut images = read_data_file(filepath)?;
        dataset.append(&mut images);
    }
    // let cifar_image = dataset.get(10).unwrap();
    // image_from_buf(cifar_image, WIDTH as u32, HEIGHT as u32)?;
    let dev = Cuda::default();
    let mut model = dev.build_module::<Cifar10Quick, f32>();
    let mut grads = model.alloc_grads();
    let mut opt: Adam<_, f32, Cuda> = Adam::new(
        &model,
        AdamConfig {
            lr: LEARNING_RATE,
            betas: [0.9, 0.999], //[0.5, 0.25],
            eps: 1e-6,
            weight_decay: Some(WeightDecay::Decoupled(1e-2)),
        },
    );
    let dataset_len = dataset.len();
    let mut sma = SMA::new(SMA_PERIOD, &0.0)?;
    for ep in 1..=EPISODES {
        let idx_rande: Vec<usize> = [(); BATCH_SIZE]
            .par_iter()
            .map(|_| fastrand::usize(0..dataset_len))
            .collect();
        // let time = Instant::now();
        let mut tensor_data = Vec::new();
        let mut label_data = Vec::new();
        for i in idx_rande {
            if let Some(cifar_image) = dataset.get(i) {
                let mut data = cifar_image.data.clone();
                tensor_data.append(&mut data);
                let mut embed = [0f32; 10].to_vec();
                if let Some(el) = embed.get_mut(cifar_image.label as usize) {
                    *el = 1.0;
                }
                label_data.append(&mut embed);
            }
        }
        let x: Tensor<Rank4<BATCH_SIZE, 3, 32, 32>, f32, Cuda> = dev.tensor(
            tensor_data
                .par_iter()
                .map(|x| *x as f32 / 255.0)
                .collect::<Vec<f32>>(),
        );
        let y: Tensor<Rank2<BATCH_SIZE, 10>, f32, Cuda> = dev.tensor(label_data);
        let logits = model.forward_mut(x.trace(grads));
        let loss = cross_entropy_with_logits_loss(logits, y);
        let sma_err = sma.next(&(loss.array() as f64));
        if ep as u64 % SMA_PERIOD == 0 {
            println!(
                "#{: >-2} loss: {: >-3.4e} lr: {:.3e}",
                ep, sma_err, opt.cfg.lr
            );
        }
        grads = loss.backward();
        opt.update(&mut model, &grads).unwrap();
        model.zero_grads(&mut grads);
        if ep % 10 == 0 {
            let lr = opt.cfg.lr * LEARNING_RATE_DECAY;

            opt.cfg.lr = lr.max(5e-5);
        }
        // if ep < 30 {

        // }else if ep < 40 {
        //     opt.cfg.lr = 5e-3;
        // }else {
        //     opt.cfg.lr = 5e-4;
        // }
    }
    model.save("data/cifar10.npz")?;
    Ok(())
}
#[allow(unused)]
fn read_data_file2<P: AsRef<Path>>(filepath: P) -> Result<Vec<Cifar10Image>, Box<dyn Error>> {
    let mut file = File::open(filepath)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let mut images: Vec<Cifar10Image> = Vec::new();
    let image_size = 3 * 32 * 32;

    for chunk in buffer.chunks_exact(1 + image_size) {
        let label = unsafe { ::std::mem::transmute(chunk[0]) };
        let data = chunk[1..].to_vec();
        images.push(Cifar10Image { label, data });
    }

    Ok(images)
}
fn read_data_file<P: AsRef<Path>>(filepath: P) -> Result<Vec<Cifar10Image>, Box<dyn Error>> {
    let mut file: File = File::open(filepath)?;
    let mut images = Vec::new();
    loop {
        let mut buf = Vec::with_capacity(BUF_LEN + 1);
        let r = file
            .by_ref()
            .take((BUF_LEN + 1) as u64)
            .read_to_end(&mut buf)?;
        if r > 0 {
            let img = Cifar10Image {
                label: unsafe { ::std::mem::transmute(buf[0]) },
                // data: convert_cifar_to_rgb(&buf[1..]),
                data: buf[1..].to_vec(),
            };
            images.push(img);
        } else {
            break;
        }
    }
    Ok(images)
}
#[allow(unused)]
fn image_from_buf(
    cifar_image: &Cifar10Image,
    width: u32,
    height: u32,
) -> Result<(), Box<dyn Error>> {
    if let Some(img_buf) = image::RgbImage::from_raw(width, height, cifar_image.data.clone()) {
        let filename_rgb = format!("data/ficar10/{:?}_rgb.png", cifar_image.label);
        img_buf.save(filename_rgb)?;
    }
    Ok(())
}
#[allow(unused)]
fn convert_cifar_to_rgb(vec: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    for r in 0..1024 {
        let g = r + 1024;
        let b = r + 1024 + 1024;
        out.push(*vec.get(r).unwrap());
        out.push(*vec.get(g).unwrap());
        out.push(*vec.get(b).unwrap());
    }
    out
}

type Cifar10Quick = (
    (LayerABn<3, 32>, DropoutOneIn<5>),
    // (LayerABn<32, 64>, DropoutOneIn<5>),
    (LayerBBn<32, 64>, DropoutOneIn<5>),
    Flatten2D,
    (Linear<1024, 512>, DropoutOneIn<5>),
    Linear<512, 10>,
);
pub type LayerA<const IN_CHAN: usize, const OUT_CHAN: usize> =
    ((BiasedConv<IN_CHAN, OUT_CHAN>, ReLU), MaxPool2D<2, 2, 0>);
pub type LayerABn<const IN_CHAN: usize, const OUT_CHAN: usize> = (
    (BiasedConv<IN_CHAN, OUT_CHAN>, BatchNorm2D<OUT_CHAN>),
    (ReLU, MaxPool2D<2, 2, 0>),
);
pub type LayerB<const IN_CHAN: usize, const OUT_CHAN: usize> = (
    (BiasedConv<IN_CHAN, OUT_CHAN>, ReLU),
    (BiasedConv<OUT_CHAN, OUT_CHAN>, ReLU),
    MaxPool2D<2, 2>,
);
pub type LayerBBn<const IN_CHAN: usize, const OUT_CHAN: usize> = (
    (BiasedConv<IN_CHAN, OUT_CHAN>, BatchNorm2D<OUT_CHAN>, ReLU),
    (BiasedConv<OUT_CHAN, OUT_CHAN>, BatchNorm2D<OUT_CHAN>, ReLU),
    MaxPool2D<2, 2, 0>,
);

pub type LayerC<const IN_CHAN: usize, const OUT_CHAN: usize> = (
    (BiasedConv<IN_CHAN, OUT_CHAN>, ReLU),
    (BiasedConv<OUT_CHAN, OUT_CHAN>, ReLU),
    (BiasedConv<OUT_CHAN, OUT_CHAN>, ReLU),
    MaxPool2D<2, 2>,
);
pub type BiasedConv<
    const IN_CHAN: usize,
    const OUT_CHAN: usize,
    const KERNEL_SIZE: usize = 4,
    const STRIDE: usize = 1,
    const PADDING: usize = 0,
> = (
    Conv2D<IN_CHAN, OUT_CHAN, KERNEL_SIZE, STRIDE, PADDING>,
    Bias2D<OUT_CHAN>,
);
