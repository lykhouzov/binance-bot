use binance::agent::consts::INIT_BALANCE;
#[allow(unused)]
use binance::agent::conv::dqn::DQNAgent;
#[allow(unused)]
use binance::agent::dfdx::ppo::PPOAgent;
use binance::agent::Agent;
use binance::dataset::Dataset;
use binance::environment::{Environment, Interval};
use binance::utils::init_logger;
use chrono::{Duration, Utc};
use clap::{arg, command, Parser};
use std::collections::VecDeque;
use std::error::Error;
use std::num::ParseIntError;
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Instant;

use yata::core::PeriodType;

fn main() -> Result<(), Box<dyn Error>> {
    // logger_init();
    let log_path = format!("var/log/debug_{}.log", Utc::now().timestamp_millis());
    init_logger(log_path.as_str())?;
    let coach = Coach::parse();
    log::debug!("{:#?}", coach);
    let total_time = Instant::now();
    if coach.test {
        test(&coach)?;
    } else {
        train(&coach)?;
    }
    let duration = Duration::from_std(total_time.elapsed())?;
    let h = duration.num_hours();
    let duration = duration - Duration::hours(h);
    let m = duration.num_minutes();
    let duration = duration - Duration::minutes(m);
    let s = duration.num_milliseconds() as f32 / 1000.0;
    log::info!("Total execution time {h:0>2}:{m:0>2}:{s:0>6.3}");
    Ok(())
}
fn train(coach: &Coach) -> Result<(), Box<dyn Error>> {
    log::info!("Start Training");
    // let mut agent = PPOAgent::new();
    let mut agent = DQNAgent::new();
    if let Some(agent_path) = &coach.agent {
        agent.load(agent_path);
    }

    let mut best_profit = f32::MIN;
    for ep in 1..=coach.episodes {
        let time = Instant::now();
        let random_start_index = fastrand::usize(0..500);
        log::debug!("Random start index = {}", random_start_index);
        agent.episode_started(ep);
        let mut ds = coach.dataset();

        let mut env = Environment::new(INIT_BALANCE, random_start_index, Interval::Minute(1));

        log::debug!("Episode #{} started", ep);
        log::debug!("Skip {} steps to fill TAs", &ds.skip_index());
        for _ in 0..ds.skip_index() as usize + random_start_index {
            let _ = ds.next();
        }
        let mut state_klines = VecDeque::with_capacity(coach.window);
        let mut state_indicators = VecDeque::with_capacity(coach.window);
        let mut state: Option<Vec<f32>> = None;
        for (step, kline, tas) in ds {
            log::trace!("Current step: {}", step);
            let price = kline.close as f32;
            state_klines.push_back(kline);
            state_indicators.push_back(tas);
            if state_klines.len() < coach.window {
                continue;
            }
            // 1. update curent price, because we now on new state
            env.set_current_price(price as f32);
            // 2. calculate reward against new state
            let reward = env.calculate_reward();
            log::trace!("Reward: {}", reward);
            state_klines.make_contiguous();
            state_indicators.make_contiguous();
            let (klines, _) = state_klines.as_slices();
            let (indicators, _) = state_indicators.as_slices();

            let save_step_image = false; //step % 10 == 0;
            let save_step_rgb_image = false; //step % 10 == 0;
            let next_state = env.get_img_state_vec(
                klines,
                indicators,
                step as usize,
                save_step_image,
                save_step_rgb_image,
            )?;
            let next_state: Vec<f32> = next_state.iter().map(|x| (*x as f32) / 255.0).collect();
            if let Some(state) = state {
                // 3. select and action for current state
                let action = if coach.random {
                    agent.choose_random_action()
                } else {
                    agent.choose_action(&state, ep % 5 != 0)
                };
                // 4. Make the action
                env.step(action);
                // 5. check if we done.
                let done = env.is_done();
                // 6. Remembver the move :)
                agent.update(state, action, reward, next_state.clone(), done);
            }

            state = Some(next_state);
            //
            //
            //
            let _ = state_klines.pop_front();
            let _ = state_indicators.pop_front();
            if !coach.random {
                agent.step_finished(step as usize);
            }
            if env.is_done() > 0.0 {
                break;
            }
        }
        env.update_last_transaction();
        agent.episode_finished(ep);
        // Save agent if profit > 0
        let profit = env.get_profit();

        if profit > 0.0
        /*&& profit > best_profit*/
        {
            agent.save(format!("data/agent/dqn_{profit:.2}_{ep}.npz"));
            env.save_episode_to_img(coach.dataset(), ep)?;
        }
        best_profit = best_profit.max(profit);
        log::debug!("Episode #{} finished", ep);
        log::info!("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓");
        log::info!(
            "┃Eposide #{} / {} | {:.4}s | {best_profit:.2} | {}",
            ep,
            coach.episodes,
            time.elapsed().as_secs_f32(),
            env.step
        );

        log::info!("┃{}", env.info());
        log::info!("┃{}", agent.info());
        log::info!("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛");
        log::debug!(
            "┏━━━━━━━━━━╾┤ Transactions[{: >2}] ├╼━━━━━━━━┓",
            env.transactions.len()
        );
        log::debug!("┃Step│  Buy  │ Stock│Held│  Sell │ Profit┃");
        let mut sum = 0.0;
        let mut positive_transaction = 0.0f32;
        for transaction in &env.transactions {
            log::debug!(
                "┃{: >3} │{:0>7.2}│{:0>6.4}│{: >3} │{:0>7.2}│{: >7.3}┃",
                transaction.0,
                transaction.1,
                transaction.2,
                transaction.3,
                transaction.4,
                transaction.5
            );
            sum += transaction.5;
            if transaction.5 > 0.0 {
                positive_transaction += 1.0;
            }
        }
        log::debug!("┠                                │{: >6.2} ┃", sum);
        log::debug!(
            "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙ {: >5.2}%┛",
            positive_transaction / (env.transactions.len() as f32) * 100.0
        );
    }
    Ok(())
}

fn test(coach: &Coach) -> Result<(), Box<dyn Error>> {
    log::info!("Start Testing");
    let ep = 0;
    let time = Instant::now();
    let mut agent = DQNAgent::new();
    // let mut agent = PPOAgent::new();
    if let Some(agent_path) = &coach.agent {
        agent.load(agent_path);
    } else {
        return Err("You should specify path to agent weights".into());
    }
    agent.episode_started(ep);
    let mut env = Environment::new(INIT_BALANCE, 0, Interval::Minute(1));
    let mut ds = coach.dataset();

    log::debug!("Skip {} steps to fill TAs", &ds.skip_index());
    for _ in 0..ds.skip_index() {
        let _ = ds.next();
    }
    let mut state_klines = VecDeque::with_capacity(coach.window);
    let mut state_indicators = VecDeque::with_capacity(coach.window);
    let mut state: Option<Vec<f32>> = None;
    for (step, kline, tas) in ds {
        log::trace!("Current step: {}", step);
        let price = kline.close as f32;
        state_klines.push_back(kline);
        state_indicators.push_back(tas);
        if state_klines.len() < coach.window {
            continue;
        }

        state_klines.make_contiguous();
        state_indicators.make_contiguous();
        let (klines, _) = state_klines.as_slices();
        let (indicators, _) = state_indicators.as_slices();

        let save_step_image = false; //step % 10 == 0;
        let save_step_rgb_image = false; //step % 10 == 0;
        let next_state = env.get_img_state_vec(
            klines,
            indicators,
            step as usize,
            save_step_image,
            save_step_rgb_image,
        )?;
        let next_state: Vec<f32> = next_state.iter().map(|x| (*x as f32) / 255.0).collect();
        if let Some(state) = state {
            // 3. update curent price, because we now on new state
            env.set_current_price(price as f32);
            // 1. select and action for current state
            let action = agent.choose_action(&state, false);
            // 2. Make the action
            env.step(action);
        }

        state = Some(next_state);
        //
        //
        //
        let _ = state_klines.pop_front();
        let _ = state_indicators.pop_front();
    }
    agent.episode_finished(ep);

    env.save_episode_to_img(coach.dataset(), ep)?;

    log::debug!("Episode #{} finished", ep);
    log::info!("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓");
    log::info!(
        "┃Eposide #{} / {} | {:.4}s | {}",
        ep,
        coach.episodes,
        time.elapsed().as_secs_f32(),
        env.step,
    );
    log::info!("┃{}", env.info());
    log::info!("┃{}", agent.info());
    log::info!("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛");
    Ok(())
}
/// Simple program to download binance Kline data
#[derive(Debug, Parser)] // requires `derive` feature
#[command(name = "coach")]
#[command(about = "Coach for environments", long_about = None)]
struct Coach {
    /// Number of episodes
    #[arg(short, long, value_parser = clap::value_parser!(usize), default_value="10")]
    episodes: usize,
    /// Window size to train on
    #[arg(short, long, value_parser = clap::value_parser!(usize), default_value="1")]
    window: usize,
    /// File path to Dataset
    #[arg(short, long, value_parser = clap::value_parser!(PathBuf))]
    file_path: PathBuf,
    /// WMA  indicators
    #[arg(long, value_parser = Coach::parse_ta)]
    with_wma: Option<VecWrap>,
    /// SMA  indicators
    // #[arg(long, value_parser = Coach::parse_ta, default_value="[]")]
    with_sma: Vec<PeriodType>,

    #[arg(long, value_parser = clap::value_parser!(bool), default_value="false")]
    test: bool,
    /// File path to Saved Agent weights
    #[arg(short, long, value_parser = clap::value_parser!(PathBuf))]
    agent: Option<PathBuf>,

    #[arg(long, value_parser = clap::value_parser!(bool), default_value="false")]
    random: bool,
}
#[derive(Debug, Clone)]
struct VecWrap(pub Vec<PeriodType>);
impl FromStr for VecWrap {
    type Err = ParseIntError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let v = s
            .trim()
            .split(",")
            .map(|s| s.trim().parse::<PeriodType>())
            .filter(|x| x.is_ok())
            .map(|x| x.unwrap())
            .collect::<Vec<PeriodType>>();
        Ok(Self(v))
    }
}
impl Coach {
    fn parse_ta(arg: &str) -> Result<VecWrap, Box<ParseIntError>> {
        if arg.len() == 0 {
            Ok(VecWrap(vec![]))
        } else {
            Ok(arg.parse::<VecWrap>().unwrap())
        }
    }

    pub fn dataset(&self) -> Dataset {
        let ds = Dataset::from(&self.file_path);
        let ds = if let Some(wma) = &self.with_wma {
            wma.0.iter().fold(ds, |ds, x| ds.with_wma(*x, &0.0))
        } else {
            ds
        };
        let ds = self.with_sma.iter().fold(ds, |ds, x| ds.with_sma(*x, &0.0));
        ds
    }
}
