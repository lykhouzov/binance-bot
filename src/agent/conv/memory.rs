use std::collections::VecDeque;

use rand::{thread_rng, Rng};

type MemoryType<S, A, R, N, D> = (S, A, R, N, D);
// pub type Sample<const B: usize, S, A, R, N, D> = ([S; B], [A; B], [R; B], [N; B], [D; B]);
pub type Sample<const B: usize, S, A, R, N, D> = (Vec<S>, Vec<A>, Vec<R>, Vec<N>, Vec<D>);
pub struct Memory<const BATCH: usize,S, A, R, N, D> {
    memory: VecDeque<MemoryType<S, A, R, N, D>>,
    memory_size: usize,
    rng: rand::rngs::ThreadRng,
}
impl<const BATCH: usize,S, A, R, N, D> Memory<BATCH, S, A, R, N, D>
where
    S: Copy + Default,
    A: Copy + Default,
    R: Copy + Default,
    N: Copy + Default,
    D: Copy + Default,
{
    pub fn new<const MS: usize>() -> Self {
        let rng = thread_rng();
        Self {
            memory: VecDeque::<MemoryType<S, A, R, N, D>>::with_capacity(MS),
            memory_size: MS,
            rng,
        }
    }
    pub fn remember(&mut self, state: S, action: A, reward: R, new_state: N, done: D) {
        if self.memory.len() > 0 && self.memory.len() >= self.memory_size {
            let _ = self.memory.pop_front();
        }
        self.memory
            .push_back((state, action, reward, new_state, done));
    }

    pub fn sample(&mut self) -> Sample<BATCH, S, A, R, N, D> {
        log::trace!("Start sampling with batch size = {}", BATCH);
        let memory_len = self.memory.len();
        log::trace!("Memory size {}", memory_len);
        //let (states, actions, rewards, dones)
        let (states, actions, rewards, next_states, dones) = [(); BATCH]
            .map(|_| {
                let idx = self.rng.gen_range(0..memory_len);
                self.memory.get(idx).unwrap()
            })
            .iter()
            .enumerate()
            .fold(
                (
                    vec![Default::default(); BATCH],
                    vec![Default::default(); BATCH],
                    vec![Default::default(); BATCH],
                    vec![Default::default(); BATCH],
                    vec![Default::default(); BATCH],
                ),
                |mut acc, (idx, (state, action, reward, new_state, done))| {
                    acc.0[idx] = state.clone();
                    acc.1[idx] = action.clone();
                    acc.2[idx] = reward.clone();
                    acc.3[idx] = new_state.clone();
                    acc.4[idx] = done.clone();
                    acc
                },
            );
        log::trace!("Finish sampling");
        (states, actions, rewards, next_states, dones)
    }
    pub fn len(&self) -> usize {
        self.memory.len()
    }
}
