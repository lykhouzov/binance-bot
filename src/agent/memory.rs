use std::collections::VecDeque;
use std::ops::*;

// State, Action, Next State, Done
type MemoryType<S, A, R, N, D> = (S, A, R, N, D);
pub type Sample<S, A, R, N, D> = (Vec<S>, Vec<A>, Vec<R>, Vec<N>, Vec<D>);
pub struct Memory<S, A, R, N, D> {
    memory: VecDeque<MemoryType<S, A, R, N, D>>,
    memory_size: usize,
}
impl<S, A, R, N, D> Memory<S, A, R, N, D>
where
    S: Clone + Default,
    A: Copy + Default,
    R: Copy + Default + Add<R, Output = R> + Mul<D, Output = R>,
    N: Clone + Default,
    D: Copy + Default + Mul<R, Output = R> + Add<D, Output = D>,
{
    pub fn new<const MS: usize>() -> Self {
        Self {
            memory: VecDeque::<MemoryType<S, A, R, N, D>>::with_capacity(MS),
            memory_size: MS,
        }
    }
    pub fn remember(&mut self, state: S, action: A, reward: R, new_state: N, done: D) {
        if self.memory.len() > 0 && self.memory.len() >= self.memory_size {
            let _ = self.memory.pop_front();
        }
        self.memory
            .push_back((state, action, reward, new_state, done));
    }
    pub fn get_last_of<const BATCH: usize>(&mut self) -> Sample<S, A, R, N, D> {
        let memory_len = self.memory.len();
        let (slice, _) = self.memory.as_slices();
        let iter: Vec<&MemoryType<S, A, R, N, D>> =
            slice[(memory_len - BATCH)..memory_len].iter().collect();
        self.sample_from_iter::<BATCH>(iter.iter())
    }

    // pub fn recalculate_gae(&mut self, next_value: V) {
    //     let mut gae = R::default();
    //     let mut next_value = next_value;
    //     self.memory
    //         .make_contiguous()
    //         .iter_mut()
    //         .rev()
    //         .for_each(|x| {
    //             let delta = x.2 + x.4 * next_value;
    //             gae = delta + x.4 * gae;
    //             next_value = x.5.clone();
    //             x.2 = gae;
    //         });
    // }

    fn sample_from_iter<'a, const BATCH: usize>(
        &self,
        iter: std::slice::Iter<'a, &MemoryType<S, A, R, N, D>>,
    ) -> Sample<S, A, R, N, D> {
        iter.enumerate().fold(
            (
                vec![Default::default(); BATCH],
                vec![Default::default(); BATCH],
                vec![Default::default(); BATCH],
                vec![Default::default(); BATCH],
                vec![Default::default(); BATCH],
            ),
            |mut acc, (idx, x)| {
                //(state, action, reward, new_state, done) = x;
                acc.0[idx] = x.0.clone();
                acc.1[idx] = x.1.clone();
                acc.2[idx] = x.2.clone();
                acc.3[idx] = x.3.clone();
                acc.4[idx] = x.4.clone();
                acc
            },
        )
    }

    pub fn sample<const BATCH: usize>(&mut self) -> Sample<S, A, R, N, D> {
        log::trace!("Start sampling with batch size = {}", BATCH);
        // do not take the last element, because it does not have calculated return
        let memory_len = self.memory.len();
        log::trace!("Memory size {}", memory_len);
        let array = [(); BATCH].map(|_| {
            // let idx = self.rng.gen_range(0..memory_len);
            let idx = fastrand::usize(0..memory_len);
            let d = self.memory.get(idx).unwrap();
            d
        });
        // let (states, actions, rewards, next_states, dones) =
        self.sample_from_iter::<BATCH>(array.iter())
    }
    pub fn len(&self) -> usize {
        self.memory.len()
    }
}


