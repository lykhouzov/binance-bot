use std::{error::Error, time::Instant};
#[allow(unused)]
use binance::{
    agent::{
        consts::{
            DATA_LEN, EPOSIDOES, IMAGE_HEIGHT, INIT_BALANCE, ONE_MINUS_TRANSACTION_FEE,
            ONE_PLUS_TRANSACTION_FEE, SAVE_ENV_IMAGE_EVERY_STEP, TARGET_BALANCE, TRANSACTION_FEE,
            WINDOW_SIZE,
        },
        conv::dqn::DQNAgent,
        dfdx::ppo::PPOAgent,
        Agent,
    },
    candlestick::CandleStickData,
    utils::{get_img_vec, load_data, logger_init},
};

use plotters::{
    prelude::*,
    style::full_palette::{GREEN_900, LIGHTBLUE, PURPLE, YELLOW_700, YELLOW_900},
};

fn main() -> Result<(), Box<dyn Error>> {
    // Init log system
    logger_init();
    //Fetch data
    let init_data = load_data("data/BTCEUR", Some(DATA_LEN));
    log::info!("Data is loaded: {}", init_data.len());
    log::info!("Current WINDOW_SIZE = {}", WINDOW_SIZE);
    log::info!(
        "Init Balance {:.2} | Target Balance {:.2} | Transaction fee {:.2}%",
        INIT_BALANCE,
        TARGET_BALANCE,
        TRANSACTION_FEE * 100.0
    );
    log::info!("Num of Eposides {}", EPOSIDOES);
    // make Windows<T> iterator
    let window = init_data.windows(WINDOW_SIZE);
    // Make data to get `last` price based on Open price of next candle
    let data = init_data[WINDOW_SIZE..].iter();
    // init Agent
    let mut agent = PPOAgent::new();
    // let mut agent = DQNAgent::new();
    let data_iter = window.enumerate().zip(data);
    for ep in 1..=EPOSIDOES {
        let mut env = Environment::new();
        // env.set_strict(ep > 5);
        agent.episode_started(ep);
        let time = Instant::now();
        // Run steps
        log::debug!("Start an episode {}", ep);
        let mut state: Option<Vec<f32>> = None;
        let mut next_state: Vec<f32>;
        let mut action = 0;

        for ((step, data), last) in data_iter.clone() {
            log::trace!("Start step {}", step);
            //
            // 1. get new state
            //
            next_state = state_from_data(data, last, step)?;
            env.set_current_price(last.open);
            //
            // 2. if it is first step, there were no previouse one
            //
            if let Some(s) = state {
                
                // calculate reward after action was made
                let reward = env.calculate_reward();
                log::trace!("I've got a reward: {:.5}", reward);
                let done = env.is_done();
                //
                // Update an agent.
                //
                agent.update(s.clone(), action, reward, next_state.clone(), done);

                if done == 1.0 {
                    log::trace!("----------> Done before data ended");
                    break;
                }
                //
                // 3. Select an action
                //
                action = agent.choose_action(&s, true);
                //
                // update environment
                //
                env.step(action);
                // let d = agent.update();
            }

            state = Some(next_state);
            log::trace!("End step {}", step);
            //every N steps we need to learn
        }
        agent.episode_finished(ep);
        log::info!("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓");
        log::info!(
            "┃Eposide #{} / {} | {:.4}s",
            ep,
            EPOSIDOES,
            time.elapsed().as_secs_f32(),
        );
        log::info!("┃{}", env.info());
        log::info!("┃{}", agent.info());
        log::info!("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛");
        if ep % SAVE_ENV_IMAGE_EVERY_STEP == 0 {
            save_episode_to_img(&init_data, &env, ep)?;
        }
    }
    Ok(())
}
#[allow(unused)]
fn compute_gae(
    next_value: &f32,
    rewards: &Vec<f32>,
    masks: &Vec<f32>,
    values: &Vec<f32>,
) -> Vec<f32> {
    let gamma = 0.99;
    let tau = 0.95;
    let mut values = values.clone();
    values.push(*next_value);
    let mut gae = 0.0;
    let mut returns = vec![];
    for step in (0..rewards.len()).into_iter().rev() {
        let delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step];
        gae = delta + gamma * tau * masks[step] * gae;
        returns.insert(0, gae + values[step]);
    }
    returns
}

fn state_from_data(
    data: &[CandleStickData],
    last: &CandleStickData,
    step: usize,
) -> Result<Vec<f32>, Box<dyn Error>> {
    Ok(get_img_vec(data, last, step)?
        .iter()
        .map(|x| (*x as f32) / 255.0)
        .collect())
}

#[derive(Debug, Default)]
struct Environment {
    balance: f32,
    current_price: f32,
    stock_to_hold: f32,
    price_of_buy: f32,
    buys: usize,
    buy_attempts: usize,
    sells: usize,
    sell_attempts: usize,
    holds: usize,
    is_done: bool,
    strict: bool,
    step: usize,
    pub records: Vec<(usize, usize)>, // (step, action)
}
#[allow(unused)]
impl Environment {
    pub fn new() -> Self {
        Self {
            balance: INIT_BALANCE,
            ..Default::default()
        }
    }
    // pub fn reset(&mut self) {
    //     self.balance = INIT_BALANCE;
    //     self.current_price = Default::default();
    //     self.stock_to_hold = Default::default();
    //     self.price_of_buy = Default::default();
    //     self.buys = Default::default();
    //     self.sells = Default::default();
    //     self.holds = Default::default();
    //     self.buy_attempts = Default::default();
    //     self.sell_attempts = Default::default();
    //     self.is_done = Default::default();
    //     self.strict = Default::default();
    //     self.step = Default::default();
    //     self.records = Default::default();
    // }

    pub fn set_strict(&mut self, strict: bool) {
        self.strict = strict;
    }

    pub fn step(&mut self, action: usize) {
        self.step += 1;
        self.act(action);
    }
    pub fn set_current_price(&mut self, price: f32) {
        self.current_price = price;
    }
    fn act(&mut self, action: usize) {
        match action {
            //
            // HOLD
            //
            0 => {
                self.holds += 1;
                log::trace!("HOLD");
            }
            //
            // BUY
            //
            1 => {
                self.buy_attempts += 1;
                log::trace!("BUY");
                let stock_to_hold = self.get_stock_to_buy();
                if stock_to_hold > 0.001 {
                    self.buys += 1;
                    log::trace!("BUY: Balance {:.5}", self.balance);
                    let price = self.current_price * stock_to_hold * ONE_PLUS_TRANSACTION_FEE;
                    self.balance -= price;
                    self.stock_to_hold += stock_to_hold;
                    self.price_of_buy = self.current_price;
                    log::trace!(
                        "Balance {:.5}; stock_to_hold={:.5}/{:.5}; current_price={:.4}; price={:.5}",
                        self.balance,
                        stock_to_hold,
                        self.stock_to_hold,
                        self.current_price,
                        price,
                    );
                    self.records.push((self.step, action));
                } else {
                    self.is_done = true;
                }
            }
            //
            // SELL
            //
            2 => {
                self.sell_attempts += 1;
                log::trace!("SELL");
                if self.stock_to_hold > 0.0 {
                    self.sells += 1;
                    log::trace!("SELL: Balance {:.5}", self.balance);
                    let price = self.current_price * self.stock_to_hold * ONE_MINUS_TRANSACTION_FEE;
                    self.balance += price;
                    log::trace!(
                        "Balance {:.5}; stock_to_hold={:.5}; current_price={:.4}; price={:.5}",
                        self.balance,
                        self.stock_to_hold,
                        self.current_price,
                        price
                    );
                    self.stock_to_hold = 0.0;
                    self.records.push((self.step, action));
                } else {
                    self.is_done = true;
                }
            }
            _ => {
                panic!("We do not have more actions here, only 0-hold, 1-buy, 2-sell")
            }
        };
    }
    fn get_stock_to_buy(&mut self) -> f32 {
        if self.balance > 0.0 {
            (self.balance / self.current_price * 10_000.0).floor() / 10_000.0
        } else {
            0.0
        }
    }
    pub fn calculate_reward(&self) -> f32 {
        // let distance = TARGET_BALANCE - self.get_equity();
        // -distance
        self.get_equity() - self.get_prev_equity()
        // Simple reward is a distance between init price and target price
        // -(distance - INIT_BALANCE) / (TARGET_BALANCE - INIT_BALANCE)
        // another example is use Equity
        // -self.get_equity() + TARGET_BALANCE
    }
    fn get_equity(&self) -> f32 {
        self.current_price * self.stock_to_hold * ONE_MINUS_TRANSACTION_FEE + self.balance
    }
    fn get_prev_equity(&self) -> f32 {
        self.price_of_buy * self.stock_to_hold * ONE_MINUS_TRANSACTION_FEE + self.balance
    }
    /// Caclulate if we done with the Epoch.
    /// It's done when current Equity is less than 50% of initial balance
    pub fn is_done(&self) -> f32 {
        if (self.strict && self.is_done) || self.get_equity() <= INIT_BALANCE * 0.5 {
            1.0
        } else {
            0.0
        }
    }
    pub fn info(&self) -> String {
        format!(
            "Balance: {:.4} | Equity: {:.4} | Stocks: {:.4} | Buys: {}/{} | Sells: {}/{} | Holds: {}",
            self.balance,
            self.get_equity(),
            self.stock_to_hold,
            self.buys,self.buy_attempts,
            self.sells, self.sell_attempts,
            self.holds
        )
    }
}

fn save_episode_to_img(
    data: &Vec<CandleStickData>,
    env: &Environment,
    ep: usize,
) -> Result<(), Box<dyn Error>> {
    let (min, max) = data.iter().fold((f32::MAX, f32::MIN), |acc, x| {
        let (min, max) = (acc.0.min(x.low), acc.1.max(x.high));

        if let Some(i) = &x.indicators {
            (
                min.min(i.sma_240).min(i.sma_60).min(i.sma_30),
                max.max(i.sma_240).max(i.sma_60).max(i.sma_30),
            )
        } else {
            (min, max)
        }
    });

    let records = env
        .records
        .iter()
        .map(|x| {
            let data_step = x.0 + WINDOW_SIZE;
            if let Some(price) = data.get(data_step) {
                let coord = (data_step as i32, price.open);
                Some(match x.1 {
                    1 => Circle::new(coord, 1, &GREEN),
                    2 => Circle::new(coord, 1, &RED),
                    // 1 => Text::new(
                    //     "▲",
                    //     coord,
                    //     "Source Code Pro".into_font().resize(15.0).color(&GREEN),
                    // ),
                    // 2 => Text::new(
                    //     "▼",
                    //     coord,
                    //     "Source Code Pro".into_font().resize(15.0).color(&RED),
                    // ),
                    _ => panic!("No no no"),
                })
            } else {
                None
            }
        })
        .filter(Option::is_some)
        .map(|x| x.unwrap());
    // let min = min.min(last.open);
    // let max = max.max(last.open);

    let x_range = 0..data.len() as i32;
    let y_range = min..max;
    {
        let filename_rgb = format!("data/stock/episode_{}_rgb.png", ep);
        // let root = BitMapBackend::with_buffer(&mut buffer, (256, 256))
        //     .into_drawing_area();
        let root = BitMapBackend::new(
            &filename_rgb,
            (5 * DATA_LEN as u32, 2 * IMAGE_HEIGHT as u32),
        )
        .into_drawing_area();

        let mut chart = ChartBuilder::on(&root)
            .margin_left(5)
            .margin_right(0)
            .build_cartesian_2d(x_range, y_range)?;
        chart
            .configure_mesh()
            .disable_x_mesh()
            .disable_y_mesh()
            .draw()?;
        chart.draw_series(data.iter().enumerate().map(|(i, x)| {
            CandleStick::new(
                i as i32,
                x.open,
                x.high,
                x.low,
                x.close,
                GREEN_900.filled(),
                YELLOW_700.filled(),
                3,
            )
        }))?;
        chart
            .draw_series(LineSeries::new(
                data.iter()
                    .enumerate()
                    .map(|(x, c)| {
                        if let Some(idicators) = &c.indicators {
                            Some((x as i32, idicators.sma_30))
                        } else {
                            None
                        }
                    })
                    .filter(Option::is_some)
                    .map(|x| x.unwrap()),
                &LIGHTBLUE,
            ))?
            .label("SMA30");
        chart
            .draw_series(LineSeries::new(
                data.iter()
                    .enumerate()
                    .map(|(x, c)| {
                        if let Some(idicators) = &c.indicators {
                            Some((x as i32, idicators.sma_60))
                        } else {
                            None
                        }
                    })
                    .filter(Option::is_some)
                    .map(|x| x.unwrap()),
                &PURPLE,
            ))?
            .label("SMA60");
        chart
            .draw_series(LineSeries::new(
                data.iter()
                    .enumerate()
                    .map(|(x, c)| {
                        if let Some(idicators) = &c.indicators {
                            Some((x as i32, idicators.sma_240))
                        } else {
                            None
                        }
                    })
                    .filter(Option::is_some)
                    .map(|x| x.unwrap()),
                &YELLOW_900,
            ))?
            .label("SMA240");
        chart.draw_series(records)?.label("Buys");
        // chart.draw_series(PointSeries::new(buys, 3, &GREEN))?.label("Buys");
        // chart.draw_series(PointSeries::new(sells, 3, &RED))?.label("Sells");
        // To avoid the IO failure being ignored silently, we manually call the present function
        root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    }
    Ok(())
}
