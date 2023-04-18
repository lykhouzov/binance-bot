// use dfdx::{
//     optim::{Adam, AdamConfig, WeightDecay},
//     prelude::*,
//     tensor::{Tensor, TensorFrom},
//     tensor_ops::Backward,
// };
// use rand::{thread_rng, Rng};

// use crate::{
//     agent::consts::{
//         BatchImageTensor, BatchTensor, Device, ImageTensor, ACTION, BATCH, HIDDEN_NUM, IMAGE_SIZE,
//         LEARNING_RATE,
//     },
//     utils::argmax,
// };

// #[derive(Debug, Default)]
// pub struct AgentStats {
//     random_actions: usize,
// }
// #[allow(unused)]
// pub struct A2CAgent {
//     epsilon: f32,
//     stats: AgentStats,
//     dev: Device,
//     conv: ConvModel,
//     actor: ActorModel,
//     critic: CriticModel,
//     grad_conv: Gradients<f32, Cpu>,
//     grad_actor: Gradients<f32, Cpu>,
//     grad_critic: Gradients<f32, Cpu>,
//     optimizer: Adam<ActorModel, f32, Cuda>,
//     gamma: f32,
// }

// impl A2CAgent {
//     pub fn new() -> Self {
//         let dev = Device::default();
//         let conv: ConvModel = dev.build_module::<ConvNet, f32>();
//         let actor: ActorModel = dev.build_module::<ActorNet, f32>();
//         let critic: CriticModel = dev.build_module::<CriticNet, f32>();
//         let grad_conv = conv.alloc_grads();
//         let grad_actor = actor.alloc_grads();
//         let grad_critic = critic.alloc_grads();
//         let optimizer: Adam<ActorModel, f32, Cuda> = Adam::new(
//             &actor,
//             AdamConfig {
//                 lr: LEARNING_RATE as f32,
//                 betas: [0.5, 0.25],
//                 eps: 1e-6,
//                 weight_decay: Some(WeightDecay::Decoupled(1e-2)),
//             },
//         );

//         Self {
//             epsilon: 0.001,
//             dev,
//             conv,
//             grad_conv,
//             actor,
//             grad_actor,
//             critic,
//             grad_critic,
//             optimizer,
//             gamma: 0.99,
//             stats: Default::default(),
//         }
//     }
//     pub fn forward(
//         &mut self,
//         x: ImageTensor,
//     ) -> (
//         Tensor<(Const<ACTION>,), f32, Device>,
//         Tensor<(Const<1>,), f32, Device>,
//     ) {
//         let features: Tensor<(Const<16>, Const<IMAGE_SIZE>, Const<IMAGE_SIZE>), f32, Device> =
//             self.conv.forward(x);
//         let features: Tensor<(Const<FLATTEN_2D>,), f32, Device> = features.reshape();
//         let action_probs: Tensor<(Const<ACTION>,), f32, Device> =
//             self.actor.forward(features.clone());
//         let value: Tensor<(Const<1>,), f32, Device> = self.critic.forward(features);
//         (action_probs, value)
//     }
//     pub fn choose_action(&mut self, state: &[f32]) -> usize {
//         if rand::random::<f32>() < self.epsilon {
//             self.stats.random_actions += 1;
//             let mut rng = thread_rng();
//             rng.gen_range(0..ACTION)
//         } else {
//             let x: ImageTensor = self.dev.tensor(state.to_vec());

//             //forward
//             let action_probs = {
//                 let features: Tensor<
//                     (Const<16>, Const<IMAGE_SIZE>, Const<IMAGE_SIZE>),
//                     f32,
//                     Device,
//                 > = self.conv.forward(x);
//                 let features: Tensor<(Const<FLATTEN_2D>,), f32, Device> = features.reshape();
//                 let action_probs: Tensor<(Const<ACTION>,), f32, Device> =
//                     self.actor.forward(features.clone());
//                 action_probs
//             };
//             // let (action_probs, _value) = self.forward(x);
//             argmax(&action_probs.as_vec()[..]) as usize
//         }
//     }
//     pub fn step(&mut self) {}
//     pub fn learn(
//         &mut self,
//         state: &[f32],
//         action: &[usize],
//         reward: &[f32],
//         next_state: &[f32],
//         done: &[f32],
//     ) {
//         // self.optimizer.zero_grad()
//         let states: BatchImageTensor<BATCH> = self.dev.tensor(state.to_vec());
//         let next_states: BatchImageTensor<BATCH> = self.dev.tensor(next_state.to_vec());
//         let actions: BatchTensor<usize> = self.dev.tensor(action.to_vec());
//         let rewards: BatchTensor<f32> = self.dev.tensor(reward.to_vec());
//         let dones: BatchTensor<f32> = self.dev.tensor(done.to_vec());
//         let grads = self.grad_actor.clone();
//         let (action_probs, value) = {
//             let features: Tensor<
//                 (
//                     Const<BATCH>,
//                     Const<16>,
//                     Const<IMAGE_SIZE>,
//                     Const<IMAGE_SIZE>,
//                 ),
//                 f32,
//                 Device,
//             > = self.conv.forward(states.clone());
//             // Flatten2D
//             let features: Tensor<(Const<BATCH>, Const<FLATTEN_2D>), f32, Device> =
//                 features.reshape();
//             let action_probs = self.actor.forward(features.trace(grads));
//             let value = self.critic.forward(features);
//             (action_probs, value.reshape())
//         };
//         let next_value = {
//             let features: Tensor<
//                 (
//                     Const<BATCH>,
//                     Const<16>,
//                     Const<IMAGE_SIZE>,
//                     Const<IMAGE_SIZE>,
//                 ),
//                 f32,
//                 Device,
//             > = self.conv.forward(states);
//             // Flatten2D
//             let features: Tensor<(Const<BATCH>, Const<FLATTEN_2D>), f32, Device> =
//                 features.reshape();
//             let value = self.critic.forward(features);
//             value.reshape()
//         };
//         let action_log_prob = action_probs.select(actions).ln();
//         let advantage: BatchTensor<f32> = (rewards
//             + (self.dev.tensor(1.0f32).broadcast() - dones)
//                 * self.dev.tensor(self.gamma).broadcast()
//                 * next_value
//             - value)
//             .reshape();

//         let actor_loss = (-action_log_prob * advantage.clone()).sum();
//         let critic_loss = advantage.powi(2).sum();
//         let loss = actor_loss + critic_loss;
//         self.grad_actor = loss.backward();
//         // self.optimizer
//         // .update(&mut self.actor, &self.grad_actor)
//         // .expect("Unused params");
//         self.actor.zero_grads(&mut self.grad_actor);
//     }
// }

// type ActorModel = (
//     (modules::Linear<FLATTEN_2D, HIDDEN_NUM, f32, Device>, ReLU),
//     (modules::Linear<HIDDEN_NUM, ACTION, f32, Device>, ReLU),
// );
// type ActorNet = (
//     (Linear<FLATTEN_2D, HIDDEN_NUM>, ReLU),
//     (Linear<HIDDEN_NUM, ACTION>, ReLU),
// );
// type CriticModel = (
//     (modules::Linear<FLATTEN_2D, HIDDEN_NUM, f32, Device>, ReLU),
//     (modules::Linear<HIDDEN_NUM, 1, f32, Device>, ReLU),
// );
// type CriticNet = (
//     (Linear<FLATTEN_2D, HIDDEN_NUM>, ReLU),
//     (Linear<HIDDEN_NUM, 1>, ReLU),
// );
// type ConvModel = (
//     (modules::Conv2D<1, 4, 3, 1, 1, f32, Device>, ReLU),
//     (modules::Conv2D<4, 8, 3, 1, 1, f32, Device>, ReLU),
//     (modules::Conv2D<8, 16, 3, 1, 1, f32, Device>, ReLU),
//     // Flatten2D,
//     // modules::Linear<7744, 7744, f32, Device>,
// );
// type ConvNet = (
//     (Conv2D<1, 4, 3, 1, 1>, ReLU),
//     (Conv2D<4, 8, 3, 1, 1>, ReLU),
//     (Conv2D<8, 16, 3, 1, 1>, ReLU),
//     // Flatten2D,
//     // Linear<7744, ACTION>,
// );
// const FLATTEN_2D: usize = IMAGE_SIZE * IMAGE_SIZE * 16;

// // type Vgg11Model = (
// //     (modules::Conv2D<1, 64, 64, 1, 1, f32, Device>, ReLU),
// //     (modules::Conv2D<128, 128, 3, 1, 1, f32, Device>, ReLU),
// //     (modules::Conv2D<128, 256, 3, 1, 1, f32, Device>, ReLU),
// //     (modules::Conv2D<256, 512, 3, 1, 1, f32, Device>, ReLU),
// //     (modules::Conv2D<512, 512, 3, 1, 1, f32, Device>, ReLU),
// //     // Flatten2D,
// //     // modules::Linear<7744, 7744, f32, Device>,
// // );
// // type Vgg11Net = (
// //     (Conv2D<1, 64, 64, 1, 1>, ReLU),
// //     (Conv2D<128, 128, 3, 1, 1>, ReLU),
// //     (Conv2D<128, 256, 3, 1, 1>, ReLU),
// //     (Conv2D<256, 512, 3, 1, 1>, ReLU),
// //     (Conv2D<512, 512, 3, 1, 1>, ReLU),
// //     // Flatten2D,
// //     // Linear<7744, ACTION>,
// // );
