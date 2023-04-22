#[allow(unused)]
use dfdx::{
    shapes::{Const, Rank1, Rank2, Rank3, Rank4},
    tensor::{Cpu, Cuda, Tensor},
};
use plotters::backend::{PixelFormat, RGBPixel};

pub const DATA_LEN: usize = WINDOW_SIZE + 30;
pub const INIT_BALANCE: f32 = 100.0;
pub const TARGET_BALANCE: f32 = INIT_BALANCE * 4.0;
pub const SIZE_OF_TRADE: f32 = 0.001;
pub const EPOSIDOES: usize = 150; //20000;
pub const MEMORY_SIZE: usize = 700; //1_000;
pub const LEARNING_RATE: f64 = 1e-5;
pub const LEARNING_RATE_MIN: f32 = 1e-4;
pub const LEARNING_RATE_DECAY: f32 = 0.995;
pub const HIDDEN_NUM: usize = STATE; // STATE*2
pub const BATCH: usize = 256;
pub const BATCH_TRAIN_TIMES: usize = 5;
pub const TRAIN_EVERY_STEP: usize = 12;
pub const TRAIN_MAX_STEPS: usize = 24 * 30;

pub const STATE: usize = NUM_FEATURES * WINDOW_SIZE + 3;
pub const ACTION: usize = 3;
pub const WINDOW_SIZE: usize = 24 * 3;
pub const CANDLE_STICK_DATA_LEN: usize = 4;
pub const NUM_FEATURES: usize = CANDLE_STICK_DATA_LEN; // it is required to recalculate each time when you add new feature
pub const RANDOM_ACTION_EPSILON: f32 = 1.0;
pub const RANDOM_ACTION_EPSILON_MIN: f32 = 1e-3;
pub const RANDOM_ACTION_EPSILON_DECAY: f32 = 0.995;
//
//
//
pub const TRANSACTION_FEE: f32 = 0.001;
pub const ONE_PLUS_TRANSACTION_FEE: f32 = 1.0 + TRANSACTION_FEE;
pub const ONE_MINUS_TRANSACTION_FEE: f32 = 1.0 - TRANSACTION_FEE;
//
//
//
pub const SAVE_STEP_IMAGE: bool = false;
pub const SAVE_ENV_IMAGE_EVERY_STEP: usize = 5;

pub const IMAGE_SIZE: usize = WINDOW_SIZE * 3; // 3px is every candle
pub const IMAGE_WIDTH: usize = WINDOW_SIZE * 3; // 3px is every candle
pub const IMAGE_HEIGHT: usize = WINDOW_SIZE * 2; // 3px is every candle
pub const IMAGE_CH_IN: usize = 1; // We use grayscale image, so it is only 1 IN_CH
pub const AFTER_FLAT: usize = 1920;
pub const L1: usize = 4;
pub const L2: usize = L1 * 2;
pub const L3: usize = L2 * 2;
pub const L4: usize = L3 * 2;
pub const VGG_FC: usize = 512;
pub const KERNEL_SIZE: usize = 4;
pub const STRIDE: usize = 1;
pub const PADDING: usize = 0;
// const L1:usize = 64;
// const L2:usize = 128;
// const L3:usize = 256;
// const L4:usize = 512;
pub const BUFFER_SIZE: usize = RGBPixel::PIXEL_SIZE * (IMAGE_WIDTH * IMAGE_HEIGHT) as usize;

// pub const STATE_SHAPE: [i64; 2] = [BATCH as i64, NUM_FEATURES as i64];
pub const STATE_SHAPE: Rank2<BATCH, STATE> = (Const::<BATCH>, Const::<STATE>);
pub const ACTION_SHAPE: Rank1<BATCH> = (Const::<BATCH>,);
pub const REWARD_SHAPE: Rank1<BATCH> = ACTION_SHAPE;
pub const DONE_SHAPE: Rank1<BATCH> = ACTION_SHAPE;

pub type StateTensor = Vec<f32>;
pub type StateTensorCuda = Tensor<(Const<STATE>,), f32, Cuda>;
pub type BatchStateTensor = Tensor<(Const<BATCH>, Const<STATE>), f32, Device>;
pub type BatchTensor<T> = Tensor<(Const<BATCH>,), T, Device>;
pub type BatchActionTensor = BatchTensor<usize>;
pub type BatchRewardTensor = Tensor<(Const<BATCH>, Const<1>), f32, Device>;
pub type BatchDoneTensor = BatchRewardTensor;

pub type Device = Cuda;
pub type ImageTensor = Tensor<Rank3<1, IMAGE_WIDTH, IMAGE_HEIGHT>, f32, Device>;
pub type BatchImageTensor<const B: usize> =
    Tensor<Rank4<B, 1, IMAGE_WIDTH, IMAGE_HEIGHT>, f32, Device>;
