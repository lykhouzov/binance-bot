#![feature(let_chains)]

use binance::agent::consts::{INIT_BALANCE, TRAIN_MAX_STEPS, WINDOW_SIZE};
#[allow(unused)]
use binance::agent::conv::dqn::DQNAgent;
#[allow(unused)]
use binance::agent::dfdx::ppo::PPOAgent;
use binance::agent::Agent;
use binance::dataset::{Dataset, DatasetBuilder, DatasetRecord};
use binance::environment::{Environment, Interval};
use binance::utils::init_logger;
use chrono::{Duration, Utc};
use clap::{arg, command, Parser};
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
    let ds = coach.dataset();
    for ep in 1..=coach.episodes {
        let time = Instant::now();

        let iter = ds.windows(coach.window);
        let iter_len = iter.len();
        let random_start_index: usize =
            fastrand::usize(ds.skip_index() as usize..iter_len - TRAIN_MAX_STEPS * 2);
        log::debug!("Random start index = {}", random_start_index);

        let (profit, env) = run(
            ep,
            &mut agent,
            iter.clone().skip(random_start_index).take(TRAIN_MAX_STEPS),
            true,
        )?;
        if profit > 0.0 {
            env.save_episode_to_img(
                ds.iter()
                    .skip(random_start_index + coach.window)
                    .take(TRAIN_MAX_STEPS),
                ep,
            )?;
            agent.save(format!("data/agent/{}_{}_{}.npz", agent.name(), profit, ep));
        }
        let env_info = env.info();
        let agent_info = agent.info();
        best_profit = best_profit.max(profit);

        let (_, eval_env) = run(
            ep,
            &mut agent,
            iter.skip(iter_len - TRAIN_MAX_STEPS * 2),
            false,
        )?;

        log::debug!("Episode #{} finished", ep);
        log::info!("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓");
        log::info!(
            "┃Eposide #{}/{} | {:.4}s | {best_profit:.2} | {}",
            ep,
            coach.episodes,
            time.elapsed().as_secs_f32(),
            env.step
        );

        log::info!("┃{}", env_info);
        log::info!("┃{}", eval_env.info());
        log::info!("┃{}", agent_info);
        log::info!("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛");
        log_transaction(&env);
    }
    agent.save(format!(
        "data/agent/last_{}_{}.npz",
        agent.name(),
        Utc::now().format("%Y-%m-%dT%H-%M-%S")
    ));
    Ok(())
}

fn log_transaction(env: &Environment) {
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

fn test(coach: &Coach) -> Result<(), Box<dyn Error>> {
    log::info!("Start Testing");
    let time = Instant::now();

    // let mut agent = PPOAgent::new();

    let files = if let Some(agent_path) = &coach.agent {
        // agent.load(agent_path);
        if agent_path.is_dir() {
            let mut out = vec![];
            for entry in std::fs::read_dir(agent_path).expect("Unable to list") {
                let entry = entry.expect("unable to get entry").path();
                if let Some(ext) = entry.extension() && ext.to_str().unwrap() == "npz" {
                    out.push(entry);
                }
            }
            out
        } else {
            vec![agent_path.clone()]
        }
    } else {
        log::error!("You should specify path to agent weights");
        vec![]
    };
    log::info!("Found files {}", files.len());
    for (ep, file) in files.iter().enumerate() {
        log::info!("Loading saved agent file : {:?}", file);
        let mut agent = DQNAgent::new();
        agent.load(file);
        let ds = coach.dataset();
        let iter = ds.windows(coach.window);
        let (_profit, env) = run(0, &mut agent, iter, false)?;
        env.save_episode_to_img(ds.iter().skip(coach.window), ep)?;
        log::debug!("Episode #{} finished", ep);
        log::info!("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓");
        log::info!(
            "┃Eposide #{} / {} | {:.4}s | {}",
            ep,
            files.len(),
            time.elapsed().as_secs_f32(),
            env.step,
        );
        log::info!("┃{}", env.info());
        log::info!("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛");
        log_transaction(&env);
    }
    Ok(())
}

fn run<'a>(
    episode: usize,
    agent: &mut impl Agent,
    iter: impl Iterator<Item = &'a [DatasetRecord]>,
    train: bool,
) -> Result<(f32, Environment), Box<dyn Error>> {
    agent.episode_started(episode);
    let mut env = Environment::new(INIT_BALANCE, Interval::Minute(1));

    let mut state: Option<Vec<f32>> = None;
    for (step, record) in iter.enumerate() {
        log::trace!("Current step: {}", step);
        let (klines, indicators) = record.iter().fold((vec![], vec![]), |mut acc, x| {
            acc.0.push(x.0.clone());
            acc.1.push(x.1.clone());
            acc
        });
        let price = klines.last().unwrap().close;
        // 1. update curent price, because we now on new state
        env.set_current_price(price as f32);
        // 2. calculate reward against new state
        let reward = env.calculate_reward();
        let save_step_image = false; //step % 10 == 0;
        let save_step_rgb_image = false; //step % 10 == 0;
        let next_state = env.get_img_state_vec(
            &klines,
            &indicators,
            step as usize,
            save_step_image,
            save_step_rgb_image,
        )?;
        let next_state: Vec<f32> = next_state.iter().map(|x| (*x as f32) / 255.0).collect();
        if let Some(state) = state {
            // 1. select and action for current state
            let action = agent.choose_action(&state, train);
            // 2. Make the action
            env.step(action);
            if train {
                // 5. check if we done.
                let done = env.is_done();
                // 6. Remembver the move :)

                agent.update(state, action, reward, next_state.clone(), done);
                agent.step_finished(step);

                if done > 0.0 {
                    break;
                }
            }
        }

        state = Some(next_state);
    }
    agent.episode_finished(episode);
    // Save agent if profit > 0
    let profit = env.get_profit();

    Ok((profit, env))
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
    #[arg(short, long, value_parser = clap::value_parser!(usize), default_value={WINDOW_SIZE.to_string()})]
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
        let ds = DatasetBuilder::from(&self.file_path);
        let ds = if let Some(wma) = &self.with_wma {
            wma.0.iter().fold(ds, |ds, x| ds.with_wma(*x))
        } else {
            ds
        };
        let ds = self.with_sma.iter().fold(ds, |ds, x| ds.with_sma(*x));
        ds.build()
    }
}
