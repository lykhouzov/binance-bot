pub mod memory;

use crate::{
    agent::{
        consts::{
            BatchImageTensor, BatchTensor, Device, ImageTensor, StateTensor, ACTION, BATCH,
            BATCH_TRAIN_TIMES, IMAGE_HEIGHT, IMAGE_WIDTH, LEARNING_RATE, MEMORY_SIZE,
            RANDOM_ACTION_EPSILON, RANDOM_ACTION_EPSILON_DECAY, RANDOM_ACTION_EPSILON_MIN,
            TRAIN_EVERY_STEP,
        },
        conv::models::{Network, NetworkModel},
        Agent,
    },
    utils::{argmax, to_array},
};
use dfdx::{
    optim::{Adam, AdamConfig, WeightDecay},
    prelude::*,
    tensor::{Tensor, TensorFrom},
    tensor_ops::Backward,
};
use memory::*;

const IMAGE_STATE: usize = IMAGE_WIDTH * IMAGE_HEIGHT;
#[derive(Debug, Clone, Copy)]
pub struct ImageState<const N: usize>(pub [f32; N]);

impl<const N: usize> Default for ImageState<N> {
    fn default() -> Self {
        Self([0.0f32; N])
    }
}
impl<const N: usize> IntoIterator for ImageState<N> {
    type Item = f32;

    type IntoIter = std::array::IntoIter<f32, N>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIterator::into_iter(self.0)
    }
}

#[derive(Debug, Default)]
pub struct AgentStats {
    pub random_actions: usize,
    pub critic_loss: f32,
    pub actor_loss: f32,
    pub total_loss: f32,
    pub total_steps: usize,
}
impl AgentStats {
    pub fn info(&self) -> String {
        let actor_loss = self.actor_loss / self.total_steps as f32;
        let critic_loss = self.critic_loss / self.total_steps as f32;
        let total_loss = self.total_loss / self.total_steps as f32;
        format!(
            "actor loss {:>-3.2e} | critic loss {:>-3.2e} | ppo loss {:>-3.2e} | lrn times {} | rand acts: {}",
            actor_loss, critic_loss, total_loss, self.total_steps, self.random_actions
        )
    }
}

#[allow(unused)]
pub struct PPOAgent {
    epsilon: f32,
    min_epsilon: f32,
    epsilon_decay: f32,
    stats: AgentStats,
    pub dev: Device,
    pub network: NetworkModel<f32, Device>,
    pub target_network: NetworkModel<f32, Device>,
    grads: Gradients<f32, Device>,
    optimizer: Adam<NetworkModel<f32, Device>, f32, Device>,
    gamma: f32,
    memory: Memory<ImageState<IMAGE_STATE>, usize, f32, ImageState<IMAGE_STATE>, f32>,
}

impl Agent for PPOAgent {
    fn choose_random_action(&mut self) -> usize {
        self.stats.random_actions += 1;
        fastrand::usize(0..ACTION)
    }
    fn choose_action(&mut self, state: &StateTensor, train: bool) -> usize {
        if train && fastrand::f32() < self.epsilon {
            self.choose_random_action()
        } else {
            let x: ImageTensor = self.dev.tensor(state.to_vec());

            let (action_probs, _value): (Tensor<(Const<ACTION>,), f32, Device>, _) =
                self.network.forward(x);
            let action = argmax(&action_probs.as_vec()) as usize;
            action
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
        let done = (1.0 - done) * self.gamma;
        self.memory.remember(
            ImageState(to_array(state)),
            action,
            reward,
            ImageState(to_array(next_state)),
            done,
            // value,
        );
    }

    fn step_finished(&mut self, step: usize) {
        if self.memory.len() >= BATCH && step % TRAIN_EVERY_STEP == 0 {
            for _ in 0..BATCH_TRAIN_TIMES {
                self.learn();
            }
            self.target_network.clone_from(&self.network);
        }
    }

    fn info(&mut self) -> String {
        self.stats.info()
    }

    fn save<P: AsRef<std::path::Path>>(&self, path: P) {
        self.target_network.save(path).unwrap();
    }

    fn load<P: AsRef<std::path::Path>>(&mut self, path: P) {
        self.network.save(path).unwrap();
        self.target_network = self.network.clone();
    }
    fn episode_started(&mut self, _ep: usize) {
        self.stats = Default::default();
    }
    fn episode_finished(&mut self, _ep: usize) {
        self.learn();
        self.epsilon = self.min_epsilon.max(self.epsilon * self.epsilon_decay);
        self.target_network.clone_from(&self.network);
    }
}

impl PPOAgent {
    pub fn new() -> Self {
        let dev = Device::default();
        let network: NetworkModel<f32, Device> = dev.build_module::<Network, f32>();
        let target_network: NetworkModel<f32, Device> = network.clone();
        let grads = network.alloc_grads();

        let optimizer: Adam<NetworkModel<_, _>, f32, Device> = Adam::new(
            &network,
            AdamConfig {
                lr: LEARNING_RATE as f32,
                betas: [0.5, 0.25],
                eps: 1e-6,
                weight_decay: Some(WeightDecay::Decoupled(1e-2)),
            },
        );

        let memory = Memory::new::<MEMORY_SIZE>();
        Self {
            epsilon: RANDOM_ACTION_EPSILON,
            min_epsilon: RANDOM_ACTION_EPSILON_MIN,
            epsilon_decay: RANDOM_ACTION_EPSILON_DECAY,
            dev,
            network,
            target_network,
            grads,
            optimizer,
            gamma: 0.99,
            memory,
            stats: Default::default(),
        }
    }

    pub fn choose_action2(&mut self, state: &[f32]) -> (Vec<f32>, f32) {
        let x: ImageTensor = self.dev.tensor(state.to_vec());

        let (action_probs, value): (Tensor<(Const<ACTION>,), f32, Device>, _) =
            self.network.forward(x);

        (action_probs.as_vec(), value.array()[0])
    }

    pub fn learn(&mut self) {
        if self.memory.len() < BATCH {
            return ();
        }
        self.stats.total_steps += 1;
        
        //
        // Take a sample from memory
        //
        let (states, actions, rewards, next_states, dones) = self.memory.sample::<BATCH>();
        //
        // Prepare tensors
        //
        let states: Vec<f32> = states.into_iter().flatten().collect();
        let states: BatchImageTensor<BATCH> = self.dev.tensor(states);
        let next_states: Vec<f32> = next_states.into_iter().flatten().collect();
        let next_states: BatchImageTensor<BATCH> = self.dev.tensor(next_states);
        let actions: BatchTensor<usize> = self.dev.tensor(actions);
        let rewards: BatchTensor<f32> = self.dev.tensor(rewards);
        // Normalize rewards
        // let rewards = normalize(rewards);
        let dones: BatchTensor<f32> = self.dev.tensor(dones);
        // let values: BatchTensor<f32> = self.dev.tensor(values);
        let mut grads = self.grads.clone();
        //
        // DO MAGIC AFTER HERE
        //
        //This is calculated on remember step;
        // let c =
        //     (self.dev.tensor(1.0f32).broadcast() - dones) * self.dev.tensor(self.gamma).broadcast();
        let masks = dones;

        // # Compute the values, action probabilities and log probabilities
        // action_probs, values = self.policy(states)
        // action_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze()
        // log_probs = torch.log(action_probs)

        // log_prob_a = log(P(action | state, pi_net))
        let (action_probs, values) = self.target_network.forward(states.clone());
        let old_log_prob_a = action_probs
            .log_softmax::<Axis<1>>()
            .select(actions.clone());

        // // old_log_prob_a = log(P(action | state, target_pi_net))
        let (_, next_values) = self.target_network.forward(next_states.clone());
        // let log_prob_a = action_probs
        //     .log_softmax::<Axis<1>>()
        //     .select(actions.clone());
        let returns: BatchTensor<f32> = rewards + next_values.reshape() * masks;
        // // # Compute the advantages and cumulative returns
        // // next_values = self.policy.critic(next_states)
        // // returns = rewards + (1 - dones) * self.gamma * next_values
        // // advantages = returns - values
        let advantages = -values.reshape() + returns.clone();
        // Normalize advantages
        // let advantages = normalize(advantages);
        // for _ in 0..5 {
        let (action_probs, values) = self.network.forward_mut(states.trace(grads.clone()));
   
        // Caclulate entroby
        let entropy = {
            let logits = action_probs
                .with_empty_tape()
                .clamp(f32::MIN, 1.0 - f32::EPSILON);
            let probs = action_probs
                .with_empty_tape()
                .clamp(f32::EPSILON, 1.0 - f32::EPSILON)
                .softmax::<Axis<1>>();
            (logits * probs).sum::<Rank1<BATCH>, _>().mean()
        };
        let log_prob_a = action_probs
            .log_softmax::<Axis<1>>()
            .select(actions.clone());
        // ratio = P(action | state, pi_net) / P(action | state, target_pi_net)
        // but compute in log space and then do .exp() to bring it back out of log space
        let ratio = (log_prob_a - old_log_prob_a.clone()).exp();

        // because we need to re-use `ratio` a 2nd time, we need to do some tape manipulation here.
        let surr1 = ratio.with_empty_tape() * advantages.clone();
        let surr2 = ratio.clamp(0.8, 1.2) * advantages.clone();

        let actor_loss = -(surr2.minimum(surr1).mean());
        self.stats.actor_loss += actor_loss.array();
        let critic_loss = (-values.reshape() + returns.clone()).powi(2).mean();
        self.stats.critic_loss += critic_loss.array();
        let ppo_loss = critic_loss * self.dev.tensor(0.5).broadcast()
            - entropy * self.dev.tensor(0.1).broadcast()
            + actor_loss;
        let ppo_loss = ppo_loss.clamp(-1.0, 1.0);
        // total_loss += ppo_loss.array();
        // log::debug!("ppo_loss = {:?}", ppo_loss.array());
        self.stats.total_loss += ppo_loss.array();

        // run backprop
        grads = ppo_loss.backward();

        // update weights with optimizer
        self.optimizer
            .update(&mut self.network, &grads)
            .expect("Unused params");
        self.grads = grads.clone();
        self.network.zero_grads(&mut self.grads);
        // }
    }

    pub fn reset(&mut self) {
        self.stats = Default::default();
    }
    pub fn episode_started(&mut self, _ep: usize) {
        self.stats = Default::default();
    }
}
