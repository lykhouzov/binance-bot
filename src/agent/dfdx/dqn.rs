use dfdx::{
    nn::{LoadFromNpz, SaveToNpz},
    tensor::TensorFrom,
};
#[allow(unused_imports)]
use dfdx::{
    optim::{Adam, AdamConfig, Momentum, Sgd, SgdConfig, WeightDecay},
    prelude::{
        huber_loss, modules, mse_loss, DeviceBuildExt, Linear, Module, Optimizer, ReLU, ZeroGrads,
    },
    shapes::{Const, Rank1},
    tensor::{AsArray, Cuda, Gradients, Tensor, TensorFromVec, Trace},
    tensor_ops::{Backward, MaxTo, SelectTo, TryStack},
};
// use rand::{thread_rng, Rng};

use crate::{
    agent::{consts::*, memory::Memory, Agent},
    utils::{argmax, to_array},
};

// type Memory = (Vec<f32>, i64, f32, Vec<f32>, f32);
#[allow(unused)]
pub struct DQNAgent {
    // input_dim: i64,
    // action_size: i64,
    memory_size: usize,
    // batch_size: i64,
    // gamma: f32,
    // learning_rate: f32,
    memory: Memory<StateTensor, usize, f32, StateTensor, f32>,
    q_net: QNetworkModel,
    target_q_net: QNetworkModel,
    grads: Gradients<f32, Cuda>,
    opt: Adam<QNetworkModel, f32, Cuda>,
    // opt: Sgd<QNetworkModel, f32, Cuda>,
    dev: Cuda,
    pub total_loss: f32,
    pub avg_loss: f32,
    pub eposide_loss: f32,
    pub epsilon: f32,
    pub min_epsilon: f32,
    epsilon_decay: f32,
    random_actions: usize,
}
impl Agent for DQNAgent {
    fn choose_random_action(&mut self) -> usize {
        // self.random_actions += 1;
        // let mut rng = thread_rng();
        // rng.gen_range(0..ACTION)
        fastrand::usize(0..ACTION)
    }
    fn choose_action(&mut self, state: &StateTensor, train: bool) -> usize {
        if train && fastrand::f32() < self.epsilon {
            self.random_actions += 1;
           fastrand::usize(0..ACTION)
        } else {
            let state = self.dev.tensor(to_array::<STATE, f32>(state.to_vec()));
            let q_values = self.q_net.forward(state.trace(self.grads.clone()));
            argmax(&q_values.as_vec()[..]) as usize
        }
    }
    fn step_finished(&mut self, step: usize) {
        if step % 10 == 0 {
            for _ in 0..BATCH_TRAIN_TIMES {
                self.learn();
            }
        }
    }
    fn update(
        &mut self,
        state: StateTensor,
        action: usize,
        reward: f32,
        next_state: StateTensor,
        done: f32,
    ) {
        self.avg_loss += 1.0;
        self.remember(state, action, reward, next_state, done);

    }
    fn episode_finished(&mut self, ep: usize) {
        self.target_q_net.clone_from(&self.q_net);
        self.total_loss += self.eposide_loss;
        self.epsilon = self.min_epsilon.max(self.epsilon * self.epsilon_decay);
        self.grads = self.q_net.alloc_grads();
        let lr = self.opt.cfg.lr;
        if ep % 50 == 0 {
            self.opt.cfg.lr = (1e-5_f32).max(lr * 0.95);
        }
    }
    fn episode_started(&mut self, _ep: usize) {
        self.eposide_loss = 0.0;
        self.avg_loss = 1.0;
        self.random_actions = 0;
    }

    fn info(&mut self) -> String {
        format!(
            "DQNAgent: episode loss: {:.5}; avg loss: {:.5};\nmem len {};LR={:.5}; random actions: {}",
            self.eposide_loss,
            // self.total_loss,
            self.eposide_loss / self.avg_loss.max(f32::EPSILON),
            self.memory.len(),
            self.opt.cfg.lr,
            self.random_actions,
        )
    }

    fn save<P: AsRef<std::path::Path>>(&self, path: P) {
        self.target_q_net.save(path).unwrap();
    }

    fn load<P: AsRef<std::path::Path>>(&mut self, path: P) {
        self.q_net.load(&path).unwrap();
        self.target_q_net.load(&path).unwrap();
    }
}
impl DQNAgent {
    pub fn new(// input_dim: i64,
        // action_size: i64,
        // memory_size: usize,
        // batch_size: i64,
        // gamma: f32,
        // learning_rate: f32,
    ) -> Self {
        let memory: Memory<StateTensor, usize, f32, StateTensor, f32> =
            Memory::new::<MEMORY_SIZE>();

        let dev: Cuda = Default::default();
        // initialize model
        let q_net = dev.build_module::<QNetwork, f32>();
        let target_q_net = q_net.clone();

        let grads = q_net.alloc_grads();

        // let opt: Sgd<QNetworkModel, f32, Cuda> = Sgd::new(
        //     &q_net,
        //     SgdConfig {
        //         lr: LEARNING_RATE,
        //         momentum: Some(Momentum::Nesterov(0.9)),
        //         weight_decay: None,
        //     },
        // );
        let opt: Adam<_, f32, Cuda> = Adam::new(
            &q_net,
            AdamConfig {
                lr: LEARNING_RATE as f32,
                betas: [0.5, 0.25],
                eps: 1e-6,
                weight_decay: Some(WeightDecay::Decoupled(1e-2)),
            },
        );
        DQNAgent {
            memory_size: MEMORY_SIZE,
            memory,
            q_net,
            target_q_net,
            grads,
            opt,
            dev,
            total_loss: 0.0,
            eposide_loss: 0.0,
            epsilon: RANDOM_ACTION_EPSILON,
            min_epsilon: RANDOM_ACTION_EPSILON_MIN,
            epsilon_decay: RANDOM_ACTION_EPSILON_DECAY,
            avg_loss: 0.0,
            random_actions: 0,
        }
    }
    fn remember(
        &mut self,
        state: StateTensor,
        action: usize,
        reward: f32,
        next_state: StateTensor,
        done: f32,
    ) {
        self.memory
            .remember(state, action, reward, next_state, done);
        // if self.memory.len() > 0 && self.memory.len() >= self.memory_size {
        //     let _ = self.memory.pop_front();
        // }
        // self.memory
        //     .push_back((state, action, reward, next_state, done));
    }

    fn learn(&mut self) {
        if self.memory.len() < BATCH {
            return ();
        }
        let mut grads = self.grads.clone();
        let (states, actions, rewards, next_states, dones) = self.memory.sample::<BATCH>();

        let (states, actions, rewards, next_states, dones): (
            BatchStateTensor,
            BatchTensor<usize>,
            BatchTensor<f32>,
            BatchStateTensor,
            BatchTensor<f32>,
        ) = (
            self.dev
                .tensor(states.iter().flatten().map(|x| *x).collect::<Vec<f32>>()),
            self.dev.tensor(actions),
            self.dev.tensor(rewards),
            self.dev.tensor(
                next_states
                    .iter()
                    .flatten()
                    .map(|x| *x)
                    .collect::<Vec<f32>>(),
            ),
            self.dev.tensor(dones),
        );
        // let states:BatchStateTensor =states.reshape();
        // let grads =

        let q_values = self.q_net.forward(states.trace(grads));
        // No batch is concidered
        // It is requried to fix it.
        let action_qs = q_values.select(actions);

        // targ_q = R + discount * max(Q(S'))
        // curr_q = Q(S)[A]
        // loss = huber(curr_q, targ_q, 1)
        let next_q_values = self.target_q_net.forward(next_states);
        let max_next_q = next_q_values.max::<Rank1<BATCH>, _>();
        let target_q = (max_next_q * (-dones + 1.0)) * 0.99 + rewards;
        // let loss = huber_loss(action_qs, target_q, 1.0);
        let loss = mse_loss(action_qs, target_q);
        self.eposide_loss += loss.array();
        // println!("loss = {:?}", loss.array());

        // run backprop
        grads = loss.backward();

        // update weights with optimizer
        self.opt
            .update(&mut self.q_net, &grads)
            .expect("Unused params");
        self.q_net.zero_grads(&mut grads);
        self.grads = grads;
    }
}
type QNetworkModel = (
    // (modules::DropoutOneIn<2>,),
    (modules::Linear<STATE, HIDDEN_NUM, f32, Cuda>, ReLU),
    (modules::Linear<HIDDEN_NUM, HIDDEN_NUM, f32, Cuda>, ReLU),
    (modules::Linear<HIDDEN_NUM, HIDDEN_NUM, f32, Cuda>, ReLU),
    (modules::Linear<HIDDEN_NUM, HIDDEN_NUM, f32, Cuda>, ReLU),
    (modules::Linear<HIDDEN_NUM, HIDDEN_NUM, f32, Cuda>, ReLU),
    modules::Linear<HIDDEN_NUM, ACTION, f32, Cuda>,
);
type QNetwork = (
    // (DropoutOneIn<2>,),
    (Linear<STATE, HIDDEN_NUM>, ReLU),
    (Linear<HIDDEN_NUM, HIDDEN_NUM>, ReLU),
    (Linear<HIDDEN_NUM, HIDDEN_NUM>, ReLU),
    (Linear<HIDDEN_NUM, HIDDEN_NUM>, ReLU),
    (Linear<HIDDEN_NUM, HIDDEN_NUM>, ReLU),
    Linear<HIDDEN_NUM, ACTION>,
);
