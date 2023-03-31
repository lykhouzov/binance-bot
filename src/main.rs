use binance::{
    candlestick::CandleStickData,
    utils::{argmax, load_data},
};
use dfdx::{
    losses::{huber_loss, mse_loss},
    optim::{Momentum, Sgd, SgdConfig},
    prelude::*,
};
use rand::{thread_rng, Rng};
use std::collections::VecDeque;

const INIT_BALANCE: Reward = 100.0;
const SIZE_OF_TRADE: f32 = 0.0001;
const EPOSIDOES: usize = 5;
const MEMORY_SIZE: usize = 2000;
const LEARNING_RATE: f32 = 1e-2;
const NUM_FEATURES: usize = 21;
#[tokio::main]
async fn main() {
    let mut epsilon = 1.0;
    let min_epsilon = 0.01f32;
    let epsilon_decay = 0.995f32;
    let data = load_data("BTCEUR");
    let mut env = TradingEnvironment::new(data);
    let mut agent = DQNAgent::new();
    for ep in 1..=EPOSIDOES {
        let mut state: Vec<f32> = env.reset();
        agent.total_loss = 0.0;
        for _step in 0..10_000 {
            let action = agent.choose_action(&state, epsilon);
            let (next_state, reward, done) = env.step(action);
            agent.update(state.clone(), action, reward, next_state.clone(), done);
            state = next_state;
            if done == 1.0 {
                break;
            }
        }
        agent.finish_episode();
        epsilon = min_epsilon.max(epsilon * epsilon_decay);
        println!("Episode {}/{}", ep, EPOSIDOES);
        println!("total loss: {}", &agent.total_loss);
        println!(
            "Balance: {}; profit: {}%",
            &env.balance,
            (&env.balance - INIT_BALANCE) / INIT_BALANCE * 100.0
        );
    }
}

pub type Reward = f32;
pub type Action = usize;
pub type Done = f32;
#[derive(Debug)]
pub struct State {
    pub prices: Vec<f32>,
    pub balance: f32,
}
impl State {
    pub fn to_vec(&self) -> Vec<f32> {
        let (min, max) = self
            .prices
            .iter()
            .fold((f32::MAX, f32::MIN), |acc, x| (x.min(acc.0), x.max(acc.1)));
        let mut v: Vec<f32> = self
            .prices
            .iter()
            .map(|x| ((x - min) / (max - min).max(f32::EPSILON)))
            .collect();
        v.push((self.balance - INIT_BALANCE) / INIT_BALANCE);
        v
    }
}
#[derive(Debug)]
pub struct TradingEnvironment {
    data: Vec<CandleStickData>,
    current_step: usize,
    done: bool,
    balance: Reward,
    shares: usize,
    price_bought: f32,
    hold_step: f32,
}
impl TradingEnvironment {
    pub fn new(data: Vec<CandleStickData>) -> Self {
        Self {
            data,
            current_step: 0,
            done: false,
            balance: INIT_BALANCE,
            shares: 0,
            price_bought: 0.0,
            hold_step: 0.0,
        }
    }
    pub fn reset(&mut self) -> Vec<f32> {
        self.current_step = 0;
        self.done = false;
        self.balance = INIT_BALANCE;
        self.shares = 0;
        self.price_bought = 0.0;
        self.hold_step = 0.0;
        self.get_state()
    }
    fn get_state(&mut self) -> Vec<f32> {
        let mut v: Vec<f32> = self.data[self.current_step..self.current_step + 5]
            .iter()
            .map(|x| Into::<Vec<f32>>::into(x))
            .flatten()
            .collect();
        // let mut v: State = self.data.get(self.current_step).unwrap().into();
        v.push(0.0);
        v
    }

    pub fn step(&mut self, action: usize) -> (Vec<f32>, Reward, Done) {
        self.current_step += 1;
        let done = if self.data.len() - 1 <= self.current_step {
            1.0
        } else {
            0.0
        };
        let reward = self.calculate_reward(action);
        let next_state: Vec<f32> = self.get_state();

        (next_state, reward, done)
    }
    fn calculate_reward(&mut self, action: usize) -> Reward {
        if let Some(price) = self.data.get(self.current_step) {
            let price = price.close * SIZE_OF_TRADE;
            let mut balance_before_action = self.balance;
            let mut reward = 0.0;
            match action {
                //hold
                0 => {
                    if self.shares > 0 {
                        reward += (price - self.price_bought) * (self.shares as f32);
                        self.hold_step += 1.0;
                    }
                }
                //buy
                1 => {
                    //check if I can buy
                    if self.balance >= price && self.shares == 0 {
                        self.shares += 1;
                        self.balance -= price;
                        self.price_bought = price;
                    }
                }
                //sell
                2 => {
                    //check if I have something to sell
                    if self.shares > 0 {
                        self.balance += price * (self.shares as f32);
                        self.shares = 0;
                    }
                    if price >= self.price_bought * 2.0 {
                        reward += (price - self.price_bought) * 0.5;
                    }
                    self.hold_step = 0.0;
                }
                _ => panic!("We should not be here"),
            }
            reward -= self.hold_step * 5e-4;//more you hold, nore you lose
            reward += (self.balance - balance_before_action) / balance_before_action;
            reward
        } else {
            0.0
        }
    }
}

type Memory = (Vec<f32>, Action, Reward, Vec<f32>, Done);

#[allow(unused)]
pub struct DQNAgent {
    // input_dim: i64,
    // action_size: i64,
    memory_size: usize,
    // batch_size: i64,
    // gamma: f32,
    // learning_rate: f32,
    memory: VecDeque<Memory>,
    q_net: QNetworkModel,
    target_q_net: QNetworkModel,
    grads: Gradients<f32, Cpu>,
    sgd: Sgd<QNetworkModel, f32, Cpu>,
    dev: Cpu,
    pub total_loss: f32,
}
impl DQNAgent {
    pub fn new(// input_dim: i64,
        // action_size: i64,
        // memory_size: usize,
        // batch_size: i64,
        // gamma: f32,
        // learning_rate: f32,
    ) -> Self {
        let memory: VecDeque<Memory> = VecDeque::with_capacity(MEMORY_SIZE);

        let dev: Cpu = Default::default();
        // initialize model
        let q_net = dev.build_module::<QNetwork, f32>();
        let target_q_net = q_net.clone();

        let grads = q_net.alloc_grads();

        let sgd: Sgd<QNetworkModel, f32, Cpu> = Sgd::new(
            &q_net,
            SgdConfig {
                lr: LEARNING_RATE,
                momentum: Some(Momentum::Nesterov(0.9)),
                weight_decay: None,
            },
        );
        DQNAgent {
            // input_dim,
            // action_size,
            memory_size: MEMORY_SIZE,
            // batch_size,
            // gamma,
            // learning_rate,
            memory,
            q_net,
            target_q_net,
            grads,
            sgd,
            dev,
            total_loss: 0.0,
        }
    }
    fn remember(
        &mut self,
        state: Vec<f32>,
        action: Action,
        reward: Reward,
        next_state: Vec<f32>,
        done: Done,
    ) {
        if self.memory.len() > 0 && self.memory.len() - 1 >= self.memory_size {
            let _ = self.memory.pop_front();
        }
        self.memory
            .push_back((state, action, reward, next_state, done));
    }
    fn choose_action(&mut self, state: &Vec<f32>, epsilon: f32) -> Action {
        if rand::random::<f32>() < epsilon {
            let mut rng = thread_rng();
            rng.gen_range(0..ACTION)
        } else {
            let state: Tensor<(Const<STATE>,), f32, Cpu> =
                self.dev.tensor_from_vec(state.clone(), (Const::<STATE>,));
            let q_values = self.q_net.forward(state.trace(self.grads.clone()));
            argmax(&q_values.array())
        }
    }
    fn learn(&mut self) {
        if self.memory.len() < BATCH {
            return ();
        }
        let memory_len = self.memory.len();
        let mut rng = thread_rng();
        let (states, actions, rewards, next_states, dones) = [(); BATCH]
            .map(|_| {
                let idx = rng.gen_range(0..memory_len);
                self.memory.get(idx).unwrap()
            })
            .into_iter()
            .fold((vec![], vec![], vec![], vec![], vec![]), |mut acc, x| {
                acc.0.append(&mut x.0.clone());
                acc.1.push(x.1);
                acc.2.push(x.2);
                acc.3.append(&mut x.3.clone());
                acc.4.push(x.4);
                acc
            });
        let (states_min, states_max) = states
            .iter()
            .fold((f32::MAX, f32::MIN), |acc, x| (x.min(acc.0), x.max(acc.1)));
        let states: Vec<f32> = states
            .iter()
            .map(|x| ((x.clone() - states_min) / (states_max - states_min).max(f32::EPSILON)))
            .collect();
        let (next_states_min, next_states_max) = next_states
            .iter()
            .fold((f32::MAX, f32::MIN), |acc, x| (x.min(acc.0), x.max(acc.1)));
        let next_states: Vec<f32> = next_states
            .iter()
            .map(|x| {
                ((x.clone() - next_states_min)
                    / (next_states_max - next_states_min).max(f32::EPSILON))
            })
            .collect();
        let (states, actions, rewards, next_states, dones) = (
            self.dev.tensor_from_vec(states, STATE_SHAPE),
            self.dev
                .tensor_from_vec::<Rank1<BATCH>>(actions, (Const::<BATCH>,)),
            self.dev.tensor_from_vec(rewards, REWARD_SHAPE),
            self.dev.tensor_from_vec(next_states, STATE_SHAPE),
            self.dev.tensor_from_vec(dones, DONE_SHAPE),
        );
        let grads = self.grads.clone();
        let q_values = self.q_net.forward(states.trace(grads));
        let action_qs = q_values.select(actions);

        // targ_q = R + discount * max(Q(S'))
        // curr_q = Q(S)[A]
        // loss = huber(curr_q, targ_q, 1)
        let next_q_values = self.target_q_net.forward(next_states);
        let max_next_q = next_q_values.max::<Rank1<BATCH>, _>();
        let target_q = (max_next_q * (-dones + 1.0)) * 0.99 + rewards;
        // let loss = huber_loss(action_qs, target_q, 1.0);
        let loss = mse_loss(action_qs, target_q);
        self.total_loss += loss.array();
        // println!("loss = {:?}", loss.array());

        // run backprop
        self.grads = loss.backward();

        // update weights with optimizer
        self.sgd
            .update(&mut self.q_net, &self.grads)
            .expect("Unused params");
        self.q_net.zero_grads(&mut self.grads);
    }
    pub fn update(
        &mut self,
        state: Vec<f32>,
        action: Action,
        reward: Reward,
        next_state: Vec<f32>,
        done: Done,
    ) {
        self.remember(state, action, reward, next_state, done);
        self.learn();
    }
    fn finish_episode(&mut self) {
        self.target_q_net.clone_from(&self.q_net);
    }
}
type QNetworkModel = (
    (modules::Linear<STATE, 64, f32, Cpu>, ReLU),
    (modules::Linear<64, 64, f32, Cpu>, ReLU),
    (modules::Linear<64, 32, f32, Cpu>, ReLU),
    modules::Linear<32, ACTION, f32, Cpu>,
);
type QNetwork = (
    (Linear<STATE, 64>, ReLU),
    (Linear<64, 64>, ReLU),
    (Linear<64, 32>, ReLU),
    Linear<32, ACTION>,
);
const STATE_SHAPE: Rank2<BATCH, STATE> = (Const::<BATCH>, Const::<NUM_FEATURES>);
const ACTION_SHAPE: Rank1<BATCH> = (Const::<BATCH>,);
const REWARD_SHAPE: Rank1<BATCH> = ACTION_SHAPE;
const DONE_SHAPE: Rank1<BATCH> = ACTION_SHAPE;
const BATCH: usize = 64;
const STATE: usize = NUM_FEATURES;
const ACTION: usize = 3;
