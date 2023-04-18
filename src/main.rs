use std::collections::VecDeque;
use std::time::{Instant, SystemTime};

use binance::agent::consts::*;
use binance::agent::dfdx::dqn::DQNAgent;
use binance::agent::Agent;

use binance::{candlestick::CandleStickData, utils::load_data};

// #[tokio::main]
fn main() {
    let data = load_data("data/BTCEUR", Some(DATA_LEN));
    let data_len = data.len();

    println!("DATA is loaded: {}", data_len);
    let mut env = TradingEnvironment::new(data, WINDOW_SIZE);

    let mut agent = DQNAgent::new();
    agent.load("data/dqn_1680867281_0.npz");

    let mut positive_profits = 0;
    let mut best_equity = 0f32;
    let mut best_profit = -INIT_BALANCE;
    for ep in 1..=EPOSIDOES {
        let time = Instant::now();
        let mut state = env.reset();
        agent.episode_started(ep);
        for _step in 0..data_len - WINDOW_SIZE - 1 {
            let cur_state = state.unwrap();
            let action = agent.choose_action(&cur_state, true);
            let (next_state, reward, done) = env.step(action);

            agent.update(cur_state.clone(), action, reward, next_state.clone(), done);
            if done == 1.0 {
                break;
            }
            state = Some(next_state);
        }
        agent.episode_finished(ep);
        println!("============= Episode {}/{} =============", ep, EPOSIDOES);

        let equity = env.data.last().unwrap().close * env.holding_stock * (1.0 - TRANSACTION_FEE)
            + env.balance;
        best_equity = best_equity.max(equity);
        let profit = equity - INIT_BALANCE;

        if profit > 0.0 {
            positive_profits += 1;
            if profit > best_profit {
                let timestamp = SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap();

                let filepath = format!("data/dqn_{}_{}.npz", timestamp.as_secs(), profit as usize);
                agent.save(filepath);
            }
        }
        best_profit = best_profit.max(profit);
        println!(
            "Balance: {:.2}; Equity: {:.2}/{:.4}; Profit: {:.2}/{:.4} ({}/{}); Transactions: {}/{}; Holding Stocks: {}; loss: {:.5}",
            &env.balance, equity, best_equity, profit, best_profit, positive_profits, ep,env.buy_num, env.sell_num, env.holding_stock, agent.eposide_loss / agent.avg_loss.max(f32::EPSILON)
        );
        // println!("{}", &agent.info());
        println!("Execution time {:.4}s", time.elapsed().as_secs_f32());
        println!("=========================================");
    }
}

#[derive(Default)]
pub struct TradingEnvironment {
    data: Vec<CandleStickData>,
    current_step: usize,
    done: bool,
    balance: f32,
    holding_stock: f32,
    price_bought: f32,
    window_size: usize,
    window_data: VecDeque<Vec<f32>>,
    buy_num: usize,
    sell_num: usize,
    cur_price: f32,
}
impl TradingEnvironment {
    pub fn new(data: Vec<CandleStickData>, window_size: usize) -> Self {
        Self {
            data,
            balance: INIT_BALANCE,
            window_size,
            window_data: VecDeque::with_capacity(window_size),
            ..Default::default()
        }
    }
    pub fn reset(&mut self) -> Option<StateTensor> {
        self.current_step = 0;
        self.done = false;
        self.balance = INIT_BALANCE;
        self.holding_stock = 0.0;
        self.price_bought = 0.0;
        self.buy_num = 0;
        self.sell_num = 0;
        self.window_data = VecDeque::with_capacity(self.window_size);
        // self.update_state();
        self.get_state()
    }
    pub fn can_train(&self) -> bool {
        true
    }
    fn get_state(&mut self) -> Option<StateTensor> {
        let mut state: Vec<f32> = self.data[self.current_step..self.current_step + WINDOW_SIZE]
            .iter()
            .map(|x| Into::<Vec<f32>>::into(x))
            .flatten()
            .collect();
        let last = {
            let last = self.data.get(self.current_step + WINDOW_SIZE).unwrap();
            let low = last.open.min(last.close);
            let high = last.open.max(last.close);
            let high = high - low;
            fastrand::f32() * high + low
        };
        self.cur_price = last;
        state.push(last);
        state.push(self.price_bought);

        let (min, max) = state.iter().fold((f32::MAX, f32::MIN), |acc, x| {
            (acc.0.min(*x), acc.1.max(*x))
        });
        let diff = max - min;
        let mut state: Vec<f32> = state.iter().map(|x| 2.0 * (x - min) / diff - 1.0).collect();

        state.push(self.holding_stock);
        if state.len() < STATE {
            None
        } else {
            Some(state)
        }
    }
    fn update_state(&mut self) {
        self.current_step += 1;
    }

    pub fn step(&mut self, action: usize) -> (StateTensor, f32, f32) {
        let mut done = if self.data.len() <= self.current_step + WINDOW_SIZE {
            1.0
        } else {
            0.0
        };

        let reward = self.calculate_reward(action);
        if self.balance <= 0.0 {
            done = 1.0;
        }

        self.update_state();
        let next_state = self.get_state().unwrap();

        (next_state, reward, done)
    }
    fn calculate_reward(&mut self, action: usize) -> f32 {
        let price = self.cur_price;

        let _reward = match action {
            //hold
            0 => {
                if self.holding_stock > 0.0 {
                    (price - self.price_bought) * self.holding_stock * (1.0 - TRANSACTION_FEE)
                } else {
                    0.0
                }
            }
            //buy, ep: usize
            1 => {
                let stocks_to_buy = (self.balance / (price * (1.0 + TRANSACTION_FEE)) * 10_000.0)
                    .floor()
                    / 10_000.0;
                // println!("stocks_to_buy = {}; balance {} price {}; stb {}", stocks_to_buy, self.balance, price, self.balance / (price * (1.0 + TRANSACTION_FEE)));
                //check if I can buy
                if stocks_to_buy > 0.001 {
                    // println!("Stock to buy {}", stocks_to_buy);
                    self.holding_stock += stocks_to_buy;
                    self.balance -= price * stocks_to_buy * (1.0 + TRANSACTION_FEE);
                    // self.balance -= price * stocks_to_buy * TRANSACTION_FEE;
                    self.price_bought = price;
                    self.buy_num += 1;
                    1.0
                } else {
                    -1.0
                }
            }
            //sell
            2 => {
                //check if I have something to sell
                if self.holding_stock > 0.0 {
                    self.balance += price * self.holding_stock * (1.0 - TRANSACTION_FEE);
                    // self.balance -= price * (1.0-TRANSACTION_FEE);

                    let reward = (price - self.price_bought) * self.holding_stock;
                    self.holding_stock = 0.0;
                    self.price_bought = 0.0;
                    self.sell_num += 1;
                    reward + 1.0
                } else {
                    -1.0
                }
            }
            _ => panic!("We should not be here"),
        };
        let equity = if self.holding_stock > 0.0 {
            let old_balance = self.holding_stock * self.price_bought + (1f32 + TRANSACTION_FEE);
            self.holding_stock * self.cur_price * (1f32 - TRANSACTION_FEE) - old_balance
        } else {
            self.balance
        };
        let reward = -equity + TARGET_BALANCE;

        // println!("Reward after action {:?}", reward);
        //# Encourage selling when losing too much
        // if self.price_bought > 0.0 && price < (1.0 - self.stop_loss) * self.price_bought {
        // if self.price_bought > 0.0 && price < self.price_bought - INIT_BALANCE * self.stop_loss
        // {
        //     reward -= (self.holding_stock * (self.price_bought - price)).powi(2);
        // }

        //# Encourage holding to make bigger profit
        // if self.holding_stock > 0.0 && price > self.price_bought {
        //     reward += (self.holding_stock * (price - self.price_bought)).powi(2);
        // }

        // //# Encourage reaching target balance
        // if self.balance + self.holding_stock * price >= self.target_balance {
        //     reward += 10.0;
        // }

        reward
    }
}
