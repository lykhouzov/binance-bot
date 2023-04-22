use plotters::{
    prelude::{BitMapBackend, CandleStick, ChartBuilder, Circle, IntoDrawingArea, Pixel},
    series::LineSeries,
    style::{full_palette::*, Color},
};

use crate::{
    agent::consts::{BUFFER_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, TRAIN_MAX_STEPS},
    dataset::DatasetRecord,
    Kline,
};
#[allow(unused)]
use crate::{
    agent::consts::{
        ONE_MINUS_TRANSACTION_FEE, ONE_PLUS_TRANSACTION_FEE, TARGET_BALANCE, TRANSACTION_FEE,
    },
    dataset::Dataset,
};
use std::error::Error;

#[derive(Debug, Default)]
pub enum Interval {
    Minute(usize),
    Hour(usize),
    #[default]
    Day,
}
#[allow(unused)]
#[derive(Debug)]
pub struct Environment {
    balance: f32,
    current_price: f32,
    stock_to_hold: f32,
    price_of_buy: f32,
    buys: usize,
    buy_attempts: usize,
    pub holding_steps: usize,
    sells: usize,
    sell_attempts: usize,
    holds: usize,
    hold_attempts: usize,
    is_done: bool,
    strict: bool,
    pub step: usize,
    pub init_step: usize,
    // (step, action)
    pub records: Vec<(usize, usize, f32)>,
    // (step, buy price, stock, steps_held, sell price, profit)
    pub transactions: Vec<(usize, f32, f32, usize, f32, f32)>,
    init_balance: f32,
    net_pl: f32,
    interval: Interval,
    dataset: Option<Dataset>,
    action: usize,
    profit_trades: usize,
    is_profit_trade: bool,
}
#[allow(unused)]
impl Environment {
    pub fn new(init_balance: f32, interval: Interval) -> Self {
        // let dataset = Dataset::from(ds);
        Self {
            init_balance,
            balance: init_balance,
            interval,
            init_step: 0,
            dataset: None,
            current_price: Default::default(),
            stock_to_hold: Default::default(),
            price_of_buy: Default::default(),
            buys: Default::default(),
            buy_attempts: Default::default(),
            sells: Default::default(),
            sell_attempts: Default::default(),
            holds: Default::default(),
            hold_attempts: Default::default(),
            is_done: Default::default(),
            strict: Default::default(),
            step: Default::default(),
            net_pl: Default::default(),
            is_profit_trade: Default::default(),
            records: Default::default(),
            transactions: Default::default(),
            holding_steps: Default::default(),
            action: Default::default(),
            profit_trades: Default::default(),
        }
    }
    // pub fn with_ds<P: AsRef<Path>>(mut self, ds: P) -> Self {
    //     self.dataset = Some(Dataset::from(ds));
    //     self
    // }
    pub fn reset(&mut self) {
        self.balance = self.init_balance;
        self.current_price = Default::default();
        self.stock_to_hold = Default::default();
        self.price_of_buy = Default::default();
        self.buys = Default::default();
        self.sells = Default::default();
        self.holds = Default::default();
        self.hold_attempts = Default::default();
        self.buy_attempts = Default::default();
        self.sell_attempts = Default::default();
        self.is_done = Default::default();
        self.strict = Default::default();
        self.step = Default::default();
        self.records = Default::default();
        self.holding_steps = Default::default();
        self.action = Default::default();
    }

    pub fn set_strict(&mut self, strict: bool) {
        self.strict = strict;
    }

    pub fn step(&mut self, action: usize) {
        self.step += 1;
        self.is_profit_trade = false;
        self.act(action);
    }
    pub fn set_current_price(&mut self, price: f32) {
        self.current_price = price;
        self.net_pl = self.get_p_l();
        log::trace!("P/L: {:.4}", self.net_pl);
    }
    fn act(&mut self, action: usize) {
        self.action = action;
        match action {
            //
            // HOLD
            //
            0 => {
                self.hold_attempts += 1;
                if self.stock_to_hold > 0.0 {
                    self.holds += 1;
                }
                log::trace!("HOLD");
            }
            //
            // BUY
            //
            1 => {
                self.buy_attempts += 1;
                log::trace!("BUY");
                let stock_to_hold = self.get_stock_to_buy();
                let price = self.current_price * stock_to_hold * ONE_PLUS_TRANSACTION_FEE;

                if stock_to_hold > 0.0 && self.balance >= price {
                    self.balance -= price;
                    self.buys += 1;
                    log::trace!("BUY: Balance {:.5}", self.balance);

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
                    self.records.push((self.step, action, self.current_price));
                    self.transactions.push((
                        self.step,
                        self.current_price,
                        self.stock_to_hold,
                        0,
                        0.0,
                        0.0,
                    ))
                }
                // else {
                //     self.is_done = true;
                // }
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
                    let profit =
                        self.stock_to_hold * self.current_price * ONE_MINUS_TRANSACTION_FEE;
                    self.balance += profit;
                    log::trace!(
                        "Balance {:.5}; stock_to_hold={:.5}; current_price={:.4}; price={:.5}",
                        self.balance,
                        self.stock_to_hold,
                        self.current_price,
                        profit
                    );

                    self.records.push((self.step, action, self.current_price));
                    self.update_last_transaction();
                    self.stock_to_hold = 0.0;
                }
            }
            _ => {
                panic!("We do not have more actions here, only 0-hold, 1-buy, 2-sell")
            }
        };

        if self.stock_to_hold > 0.0 {
            self.holding_steps += 1;
        } else {
            self.holding_steps = 0;
        }
    }
    fn get_stock_to_buy(&mut self) -> f32 {
        if self.balance > 0.0 {
            (self.balance / self.current_price / ONE_PLUS_TRANSACTION_FEE * 10_000.0).floor()
                / 10_000.0
        } else {
            0.0
        }
    }
    fn get_p_l(&self) -> f32 {
        self.stock_to_hold
            * (self.current_price * ONE_MINUS_TRANSACTION_FEE
                - self.price_of_buy * ONE_PLUS_TRANSACTION_FEE)
    }
    pub fn calculate_reward(&mut self) -> f32 {
        // a reward is based on current Net Profit/Loss
        // and additional reward based on the action.
        -1.0 + self.net_pl
            + match self.action {
                0 => {
                    if self.holding_steps > 0 {
                        // Give big reward penalty when it holds a stock and does not sell it
                        -0.15 * (self.holding_steps as f32).log10()
                    } else {
                        // Give some reward penalty when it just doing nothing
                        -4.0
                    }
                }
                1 => {
                    // Give a reward whe it makes correct buy action
                    // that is if there is no holding stock the BUY action is legal
                    if let Some(transaction) = self.transactions.last() {
                        if transaction.4 == 0.0 {
                            -4.0
                        } else {
                            1.0
                        }
                    } else {
                        1.0
                    }
                }
                2 => {
                    // Give a reward whe SELL gaves prfitable trade
                    // That is we earn when sell
                    if let Some(transaction) = self.transactions.last() {
                        if transaction.4 > 0.0 && transaction.5 > 0.0 {
                            5.0
                        } else {
                            -4.0
                        }
                    } else {
                        -4.0
                    }
                }
                // We should not be here, so just give huge disreward
                _ => -10.0,
            }
    }
    pub fn get_equity(&self) -> f32 {
        self.current_price * self.stock_to_hold * ONE_MINUS_TRANSACTION_FEE + self.balance
    }
    fn get_last_equity(&self) -> f32 {
        self.price_of_buy * self.stock_to_hold * ONE_PLUS_TRANSACTION_FEE + self.balance
    }
    /// Caclulate if we done with the Epoch.
    /// It's done when current Equity is less than 50% of initial balance
    pub fn is_done(&self) -> f32 {
        if (self.strict && self.is_done)
            || self.get_equity() <= self.init_balance * 0.5
            || self.step >= TRAIN_MAX_STEPS
        {
            1.0
        } else {
            0.0
        }
    }
    pub fn info(&self) -> String {
        format!(
            "B: {:0<6.2} | Eqy: {:0<6.2} | Pt: {:0<6.2} | St: {:.4} | B:[{}/{}] S:[{}/{}] H:[{}/{}] | T: {}",
            self.balance,
            self.get_equity(),
            self.get_profit(),
            self.stock_to_hold,
            self.buys,
            self.buy_attempts,
            self.sells,
            self.sell_attempts,
            self.holds,
            self.hold_attempts,
            self.transactions.len()
        )
    }

    pub fn update_last_transaction(&mut self) {
        if let Some(transaction) = self.transactions.last_mut() {
            if transaction.4 == 0.0 {
                transaction.3 = self.step - transaction.0;
                transaction.4 = self.current_price;
                transaction.5 = (self.current_price * ONE_MINUS_TRANSACTION_FEE
                    - transaction.1 * ONE_PLUS_TRANSACTION_FEE)
                    * self.stock_to_hold;

                log::trace!("Transaction: {:?}", &transaction);
            }
        }
    }
    pub fn get_img_state_vec(
        &self,
        data: &[Kline<u64, f64>],
        indicators: &[Vec<f64>],
        step: usize,
        save_step_image: bool,
        save_step_rgb_image: bool,
    ) -> Result<Vec<u8>, Box<dyn Error>> {
        let (min, max) = data.iter().fold((f32::MAX, f32::MIN), |acc, x| {
            (acc.0.min(x.low as f32), acc.1.max(x.high as f32))
        });
        let (i_min, i_max) = indicators
            .iter()
            .map(|x| x.clone())
            .flatten()
            .fold((f32::MAX, f32::MIN), |acc, x| {
                (acc.0.min(x as f32), acc.1.max(x as f32))
            });
        let min = min.min(i_min);
        let max = max.max(i_max);
        log::trace!("Calculated (min,max) for the range: ({},{})", min, max);
        let x_range = 0..data.len() as i32;
        let y_range = min..max;
        let mut buffer = [0; BUFFER_SIZE];
        {
            let root =
                BitMapBackend::with_buffer(&mut buffer, (IMAGE_WIDTH as u32, IMAGE_HEIGHT as u32))
                    .into_drawing_area();

            let mut chart = ChartBuilder::on(&root)
                .margin_left(1)
                .margin_right(0)
                .build_cartesian_2d(x_range, y_range)?;
            chart
                .configure_mesh()
                .disable_x_mesh()
                .disable_y_mesh()
                .draw()?;
            if indicators.len() > 0 {
                let colors = [LIGHTBLUE, PURPLE, YELLOW_900, BLUE, BLUEGREY];
                let colors = colors.iter().cycle();
                let indicators_num = indicators.get(0).unwrap().len();
                if indicators_num > 0 {
                    for (num, color) in (0..indicators_num - 1).zip(colors) {
                        chart
                            .draw_series(LineSeries::new(
                                (0..data.len()).into_iter().map(|x| {
                                    let y = indicators[x][num] as f32;
                                    (x as i32, y)
                                }),
                                color,
                            ))?
                            .label(format!("Indicatro {}", num));
                    }
                }
            }
            let last = data.last().unwrap();
            chart
                .draw_series(LineSeries::new(
                    (0..data.len())
                        .into_iter()
                        .map(|x| (x as i32, last.close as f32)),
                    WHITE.stroke_width(1),
                ))?
                .label("Current Price");
            chart.draw_series(data.iter().enumerate().map(|(i, x)| {
                CandleStick::new(
                    i as i32,
                    x.open as f32,
                    x.high as f32,
                    x.low as f32,
                    x.close as f32,
                    GREEN.filled(),
                    RED.filled(),
                    3,
                )
            }))?;
            if self.holding_steps > 0 {
                let buy_points = [(
                    (data.len() - self.holding_steps - 1) as i32,
                    self.price_of_buy,
                )];
                chart.draw_series(buy_points.map(|coord| Pixel::new(coord, &DEEPPURPLE)));
            }
            //

            // To avoid the IO failure being ignored silently, we manually call the present function
            root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
        }
        if let Some(img_buf) =
            image::RgbImage::from_raw(IMAGE_WIDTH as u32, IMAGE_HEIGHT as u32, buffer.to_vec())
        {
            if save_step_rgb_image {
                let filename_rgb = format!("data/stock/{}_rgb.png", step);
                img_buf.save(filename_rgb);
            }

            let gray: image::GrayImage = image::DynamicImage::ImageRgb8(img_buf).into_luma8();
            if save_step_image {
                let filename = format!("data/stock/{}.png", step);
                gray.save(filename).expect("Cannot save image");
            }
            Ok(gray.to_vec())
        } else {
            Err("Cannot get RgbImage image from buffer".into())
        }
    }
    pub fn get_profit(&self) -> f32 {
        self.get_equity() - self.init_balance
    }

    pub fn save_episode_to_img<'a>(
        &self,
        iter: impl Iterator<Item = &'a DatasetRecord>,
        ep: usize,
    ) -> Result<(), Box<dyn Error>> {
        let (min, max, data, indicators) = iter.fold(
            (f32::MAX, f32::MIN, vec![], vec![]),
            |mut acc, (kline, indicators)| {
                // 1
                let (min, max) = (acc.0.min(kline.low as f32), acc.1.max(kline.high as f32));
                // 2
                let (i_min, i_max) = indicators.iter().fold((f32::MAX, f32::MIN), |acc, x| {
                    (acc.0.min(*x as f32), acc.1.max(*x as f32))
                });
                // 3
                acc.0 = min.min(i_min);
                acc.1 = max.max(i_max);
                acc.2.push(kline);
                acc.3.push(indicators);
                acc
            },
        );

        let records = self
            .records
            .iter()
            .map(|x| {
                let data_step = x.0;
                if let Some(price) = data.get(data_step) {
                    let coord = (data_step as i32 - 1, x.2);
                    Some(match x.1 {
                        1 => Circle::new(coord, 1, &GREEN_900),
                        2 => Circle::new(coord, 1, &RED),
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
                (5 * data.len() as u32, 2 * IMAGE_HEIGHT as u32),
            )
            .into_drawing_area();
            root.fill(&WHITE)?;

            let mut chart = ChartBuilder::on(&root)
                .margin_left(5)
                .margin_right(0)
                .build_cartesian_2d(x_range, y_range)?;
            chart
                .configure_mesh()
                .disable_x_mesh()
                .disable_y_mesh()
                .draw()?;

            let num_indicators = indicators.get(0).unwrap().len();
            let colors = [LIGHTBLUE, PURPLE, YELLOW_900].iter().cycle();
            for (i, color) in (0..num_indicators).zip(colors) {
                chart.draw_series(LineSeries::new(
                    indicators
                        .iter()
                        .enumerate()
                        .map(|(x, c)| {
                            if let Some(idicator) = c.get(i) {
                                Some((x as i32, *idicator as f32))
                            } else {
                                None
                            }
                        })
                        .filter(Option::is_some)
                        .map(|x| x.unwrap()),
                    color,
                ))?;
            }
            chart.draw_series(data.iter().enumerate().map(|(i, x)| {
                CandleStick::new(
                    i as i32,
                    x.open as f32,
                    x.high as f32,
                    x.low as f32,
                    x.close as f32,
                    LIGHTGREEN.filled(),
                    YELLOW_700.filled(),
                    3,
                )
            }))?;
            chart.draw_series(records)?.label("Buys");
            // chart.draw_series(PointSeries::new(buys, 3, &GREEN))?.label("Buys");
            // chart.draw_series(PointSeries::new(sells, 3, &RED))?.label("Sells");
            // To avoid the IO failure being ignored silently, we manually call the present function
            root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
        }
        Ok(())
    }
}
