use crate::agent::consts::{RANDOM_ACTION_EPSILON_DECAY, RANDOM_ACTION_EPSILON_MIN, RANDOM_ACTION_EPSILON};
use crate::agent::conv::models::*;
use crate::agent::memory::Memory;
use crate::{
    agent::consts::{
        BatchImageTensor, BatchTensor, Device, ImageTensor, ACTION, BATCH,
        IMAGE_SIZE, LEARNING_RATE, MEMORY_SIZE,
    },
    utils::{argmax, to_array},
};
use dfdx::{
    optim::{Adam, AdamConfig, WeightDecay},
    prelude::*,
    tensor::{SplitTape, Tensor, TensorFrom},
    tensor_ops::Backward,
};
// use rand::{thread_rng, Rng};
const IMAGE_STATE: usize = IMAGE_SIZE * IMAGE_SIZE;
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
    pub total_steps: usize,
}
impl AgentStats {
    pub fn info(&self) -> String {
        let actor_loss = self.actor_loss / self.total_steps as f32;
        let critic_loss = self.critic_loss / self.total_steps as f32;
        format!(
            "actor loss {:.5} | critic loss {:.5} | total_steps {}",
            actor_loss, critic_loss, self.total_steps
        )
    }
}

#[allow(unused)]
pub struct A2CAgent {
    epsilon: f32,
    min_epsilon: f32,
    epsilon_decay: f32,
    stats: AgentStats,
    dev: Device,
    actor: ActorModel<f32, Device>,
    critic: CriticModel<f32, Device>,
    grad_actor: Gradients<f32, Device>,
    grad_critic: Gradients<f32, Device>,
    optimizer: Adam<ActorModel<f32, Device>, f32, Device>,
    gamma: f32,
    memory: Memory<ImageState<IMAGE_STATE>, usize, f32, ImageState<IMAGE_STATE>, f32>,
    optimizer_c: Adam<CriticModel<f32, Device>, f32, Device>,
}

impl A2CAgent {
    pub fn new() -> Self {
        let dev = Device::default();
        let actor: ActorModel<f32, Device> = dev.build_module::<Actor, f32>();
        let critic: CriticModel<f32, Device> = dev.build_module::<Critic, f32>();
        let grad_actor = actor.alloc_grads();
        let grad_critic = critic.alloc_grads();
        let optimizer: Adam<ActorModel<_, _>, f32, Device> = Adam::new(
            &actor,
            AdamConfig {
                lr: LEARNING_RATE as f32,
                betas: [0.5, 0.25],
                eps: 1e-6,
                weight_decay: Some(WeightDecay::Decoupled(1e-2)),
            },
        );
        let optimizer_c: Adam<CriticModel<_, _>, f32, Device> = Adam::new(
            &critic,
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
            actor,
            grad_actor,
            critic,
            grad_critic,
            optimizer,
            optimizer_c,
            gamma: 0.99,
            memory,
            stats: Default::default(),
        }
    }

    pub fn choose_action(&mut self, state: &[f32]) -> usize {
        if fastrand::f32() < self.epsilon {
            log::trace!("Random action");
            self.stats.random_actions += 1;
            // let mut rng = thread_rng();
            fastrand::usize(0..ACTION)
        } else {
            log::trace!("Conscious action");
            let x: ImageTensor = self.dev.tensor(state.to_vec());

            let action_probs: Tensor<(Const<ACTION>,), f32, Device> = self.actor.forward(x);
            argmax(&action_probs.as_vec()[..]) as usize
        }
    }
    pub fn update(
        &mut self,
        state: Vec<f32>,
        action: usize,
        reward: f32,
        new_state: Vec<f32>,
        done: f32,
    ) {
        log::debug!("Update memory");
        self.memory.remember(
            ImageState(to_array(state)),
            action,
            reward,
            ImageState(to_array(new_state)),
            done,
        );
        if self.memory.len() >= BATCH {
            self.learn();
        }
    }
    pub fn learn(&mut self) {
        self.stats.total_steps += 1;
        log::debug!("Let's learn something");
        //
        // Take a sample from memory
        //
        log::trace!("mem len = {}", self.memory.len());
        let (states, actions, rewards, next_states, dones) = self.memory.sample::<BATCH>();
        log::trace!("Samples are prepared");
        //
        // Prepare tensors
        //
        let states: Vec<f32> = states.into_iter().flatten().collect();
        let states: BatchImageTensor<BATCH> = self.dev.tensor(states);
        let next_states: Vec<f32> = next_states.into_iter().flatten().collect();
        let next_states: BatchImageTensor<BATCH> = self.dev.tensor(next_states);
        let actions: BatchTensor<usize> = self.dev.tensor(actions);
        let rewards: BatchTensor<f32> = self.dev.tensor(rewards);
        let dones: BatchTensor<f32> = self.dev.tensor(dones);
        log::trace!("Tensors are prepared");
        //
        // Take a gradients
        //
        log::trace!("Take actor grads");
        let grads_actor = self.grad_actor.clone();
        log::trace!("Take critic grads");
        let grads_critic = self.grad_critic.clone();
        log::trace!("Calculate value_loss");
        let (value_loss, advantages) = {
            // # Compute the value and policy loss
            let (values, tape) = self
                .critic
                .forward_mut(states.trace(grads_critic))
                .reshape()
                .split_tape();
            let (next_values, tape) = self
                .critic
                .forward_mut(next_states.put_tape(tape))
                .reshape()
                .split_tape();
            // let advantages = rewards + (1 - dones) * next_values - values;
            let c = (self.dev.tensor(1.0f32).broadcast() - dones)
                * self.dev.tensor(self.gamma).broadcast();
            let advantages = rewards + c * next_values - values;
            let value_loss = advantages.clone().put_tape(tape).powi(2).mean();
            (value_loss, advantages)
        };
        let critic_loss = value_loss.array();
        self.stats.critic_loss += critic_loss;
        log::debug!("Calculated value_loss ={}", critic_loss);

        let policy_loss = {
            let (action_probs, tape) = self.actor.forward_mut(states.trace(grads_actor)).split_tape();
            let action_log_probs = action_probs.select(actions);
            // torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze());
            let policy_loss = -(advantages.put_tape(tape) * action_log_probs).mean();
            policy_loss
        };
        let actor_loss = policy_loss.array();
        self.stats.actor_loss += actor_loss;
        log::debug!("Calculated policy_loss ={}", actor_loss);

        self.grad_critic = value_loss.backward();
        log::trace!("Optimize Critic");
        self.optimizer_c
            .update(&mut self.critic, &self.grad_critic)
            .expect("Unused params");

        self.grad_actor = policy_loss.backward();
        log::trace!("Optimize Actor");
        self.optimizer
            .update(&mut self.actor, &self.grad_actor)
            .expect("Unused params");

        self.actor.zero_grads(&mut self.grad_actor);
        self.critic.zero_grads(&mut self.grad_critic);
    }
    pub fn info(&self) -> String {
        self.stats.info()
    }
    pub fn reset(&mut self) {
        self.stats = Default::default();
    }

    pub fn episode_finished(&mut self) {
        self.epsilon = self.min_epsilon.max(self.epsilon * self.epsilon_decay);
    }
}
