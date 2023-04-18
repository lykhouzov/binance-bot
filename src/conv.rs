#[allow(unused_imports)]
use std::{error::Error, f32::EPSILON, time::Instant};

use binance::{
    agent::{
        consts::{
            DATA_LEN, EPOSIDOES, INIT_BALANCE, ONE_MINUS_TRANSACTION_FEE,
            ONE_PLUS_TRANSACTION_FEE, TARGET_BALANCE, TRANSACTION_FEE, WINDOW_SIZE,
        },
        dfdx::a2c::A2CAgent,
    },
    candlestick::CandleStickData,
    utils::{load_data, logger_init, get_img_vec},
};

// #[tokio::main]
fn main() -> Result<(), Box<dyn Error>> {
    // Init log system
    logger_init();
    //Fetch data
    let data = load_data("data/BTCEUR", Some(DATA_LEN));
    log::info!("Data is loaded: {}", data.len());
    log::info!("Current WINDOW_SIZE = {}", WINDOW_SIZE);
    log::info!(
        "Init Balance {:.2} | Target Balance {:.2} | Transaction fee {:.2}%",
        INIT_BALANCE,
        TARGET_BALANCE,
        TRANSACTION_FEE * 100.0
    );
    log::info!("Num of Epochs {}", EPOSIDOES);
    // make Windows<T> iterator
    let window = data.windows(WINDOW_SIZE);
    // Make data to get `last` price based on Open price of next candle
    let data = data[WINDOW_SIZE..].iter();
    // init Agent
    let mut agent = A2CAgent::new();
    let data_iter = window.enumerate().zip(data);
    for ep in 1..=EPOSIDOES {
        let mut env = Environment::new();
        agent.reset();
        let time = Instant::now();
        // Run steps
        log::debug!("Start an epoch");
        let mut state: Option<Vec<f32>> = None;
        let mut new_state: Vec<f32>;
        let mut action = 0;
        for ((step, data), last) in data_iter.clone() {
            log::debug!("Start step {}", step);
            //
            // 1. get new state
            //
            new_state = state_from_data(data, last, step)?;
            //
            // 2. if it is first step, there were no previouse one
            //
            if let Some(s) = state {
                // calculate reward after action was made
                let reward = env.calculate_reward();
                log::debug!("I've got a reward: {:.5}", reward);
                let done = env.is_done();
                //
                // Update an agent.
                //
                agent.update(s.clone(), action, reward, new_state.clone(), done);
                if done == 1.0 {
                    log::trace!("----------> Done before data ended");
                    break;
                }
                //
                // 3. Select an action
                //
                action = agent.choose_action(s.as_slice());
                //
                // update environment
                //
                env.step(action, last.open);
                // let d = agent.update();
            }

            state = Some(new_state);
            log::debug!("End step {}", step);
        }
        agent.episode_finished();
        log::info!(
            "Epoch #{} / {} | {:.4}s | Balance: {:.4} | Equity: {:.4} | {}",
            ep,
            EPOSIDOES,
            time.elapsed().as_secs_f32(),
            env.balance,
            env.get_equity(),
            agent.info()
        );
    }
    Ok(())
}

// pub fn get_img_vec(
//     data: &[CandleStickData],
//     last: &CandleStickData,
// ) -> Result<Vec<u8>, Box<dyn Error>> {
//     let (min, max) = data.iter().fold((f32::MAX, f32::MIN), |acc, x| {
//         (
//             acc.0.min(x.low).min(x.sma_15).min(x.sma_30),
//             acc.1.max(x.high).max(x.sma_15).max(x.sma_30),
//         )
//     });
//     let min = min.min(last.open);
//     let max = max.max(last.open);
//     log::trace!("Calculated (min,max) for the range: ({},{})", min, max);
//     let x_range = 0..data.len() as i32;
//     let y_range = min..max;
//     let mut buffer = [0; BUFFER_SIZE];
//     {
//         let root = BitMapBackend::with_buffer(&mut buffer, (IMAGE_SIZE as u32, IMAGE_SIZE as u32))
//             .into_drawing_area();

//         let mut chart = ChartBuilder::on(&root)
//             .margin_left(5)
//             .margin_right(0)
//             .build_cartesian_2d(x_range, y_range)?;
//         chart
//             .configure_mesh()
//             .disable_x_mesh()
//             .disable_y_mesh()
//             .draw()?;
//         chart.draw_series(data.iter().enumerate().map(|(i, x)| {
//             CandleStick::new(
//                 i as i32,
//                 x.open,
//                 x.high,
//                 x.low,
//                 x.close,
//                 GREY_100.filled(),
//                 GREY_800.filled(),
//                 3,
//             )
//         }))?;
//         chart
//             .draw_series(
//                 LineSeries::new(
//                     data.iter().enumerate().map(|(x, c)| (x as i32, c.sma_15)),
//                     &GREY_300,
//                 )
//                 .point_size(1),
//             )?
//             .label("SMA5");
//         chart
//             .draw_series(LineSeries::new(
//                 data.iter().enumerate().map(|(x, c)| (x as i32, c.sma_30)),
//                 &GREY_400,
//             ))?
//             .label("SMA30");
//         chart
//             .draw_series(LineSeries::new(
//                 [last.open; WINDOW_SIZE]
//                     .iter()
//                     .enumerate()
//                     .map(|(x, y)| (x as i32, *y)),
//                 &WHITE,
//             ))?
//             .label("Current Price");

//         // To avoid the IO failure being ignored silently, we manually call the present function
//         root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
//     }
//     if let Some(img_buf) =
//         image::RgbImage::from_raw(IMAGE_SIZE as u32, IMAGE_SIZE as u32, buffer.to_vec())
//     {
//         // let filename = format!("data/stock/{}.png", step);
//         // let filename_rgb = format!("data/stock/{}_rgb.png", step);
//         // img_buf.save(filename_rgb);
//         let gray: image::GrayImage = image::DynamicImage::ImageRgb8(img_buf).into_luma8();
//         // gray.save(filename).expect("Cannot save image");
//         Ok(gray.to_vec())
//     } else {
//         Err("Cannot get RgbImage image from buffer".into())
//     }
// }

fn state_from_data(
    data: &[CandleStickData],
    last: &CandleStickData,
    step:usize
) -> Result<Vec<f32>, Box<dyn Error>> {
    Ok(get_img_vec(data, last, step)?
        .iter()
        .map(|x| (*x as f32) / 255.0)
        .collect())
}

struct Environment {
    balance: f32,
    current_price: f32,
    stock_to_hold: f32,
    price_of_buy: f32,
}
#[allow(unused)]
impl Environment {
    pub fn new() -> Self {
        Self {
            balance: INIT_BALANCE,
            current_price: 0.0,
            stock_to_hold: 0.0,
            price_of_buy: 0.0,
        }
    }
    pub fn reset(&mut self) {
        self.balance = INIT_BALANCE;
        self.current_price = 0.0;
        self.stock_to_hold = 0.0;
        self.price_of_buy = 0.0;
    }

    pub fn step(&mut self, action: usize, price: f32) {
        self.current_price = price;
        self.act(action);
    }
    fn act(&mut self, action: usize) {
        match action {
            //
            // HOLD
            //
            0 => {
                log::trace!("HOLD");
            }
            //
            // BUY
            //
            1 => {
                log::trace!("BUY");
                let stock_to_hold = self.get_stock_to_buy();
                if stock_to_hold > 0.001 {
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
                }
            }
            //
            // SELL
            //
            2 => {
                log::trace!("SELL");
                if self.stock_to_hold > 0.0 {
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
        let distance = TARGET_BALANCE - self.get_equity();
        -distance
        // Simple reward is a distance between init price and target price
        // -(distance - INIT_BALANCE) / (TARGET_BALANCE - INIT_BALANCE)
        // another example is use Equity
        // -self.get_equity() + TARGET_BALANCE
    }
    fn get_equity(&self) -> f32 {
        self.current_price * self.stock_to_hold + ONE_MINUS_TRANSACTION_FEE + self.balance
    }
    /// Caclulate if we done with the Epoch.
    /// It's done when current Equity is less than 50% of initial balance
    pub fn is_done(&self) -> f32 {
        if self.get_equity() <= INIT_BALANCE * 0.5 {
            1.0
        } else {
            0.0
        }
    }
}
