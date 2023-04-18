use std::collections::VecDeque;
use std::ops::*;
// State, Action, Next State, Done, Value
type MemoryType<S, A, R, N, D> = (S, A, R, N, D);
pub type Sample<S, A, R, N, D> = (Vec<S>, Vec<A>, Vec<R>, Vec<N>, Vec<D>);
// pub type Sample<S, A, R, N, D> = (Vec<S>, Vec<A>, Vec<R>, Vec<N>, Vec<D>, Vec<V>);
pub struct Memory<S, A, R, N, D> {
    memory: VecDeque<MemoryType<S, A, R, N, D>>,
    memory_size: usize,
}
impl<S, A, R, N, D> Memory<S, A, R, N, D>
where
    S: Copy + Default,
    A: Copy + Default,
    R: Copy + Default + Add<R, Output = R> + Mul<D, Output = R>,
    N: Copy + Default,
    D: Copy + Default + Mul<R, Output = R> + Add<D, Output = D> 
    // + Mul<V, Output = R>
    ,

    // V: Copy + Default,
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
        // self.recalculate_gae(value);
        self.memory
            .push_back((state, action, reward, new_state, done));
            // .push_back((state, action, reward, new_state, done, value));
    }
    pub fn sample_ordered<const BATCH: usize>(&mut self) -> Sample<S, A, R, N, D> {
        let memory_len = self.memory.len();
        let idx_start = fastrand::usize(0..(memory_len-BATCH));
        // log::debug!("Mem len Idx Start = {memory_len}:{idx_start}");
        // TODO check if it is required at all
        let _ = self.memory.make_contiguous();
        // let iter: Vec<&MemoryType<S, A, R, N, D>> = self.memory.iter().collect();
        let (slice, _) = self.memory.as_slices();

        
        let iter: Vec<&MemoryType<S, A, R, N, D>> = slice[idx_start..(idx_start+BATCH)].iter().collect();
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
                // vec![Default::default(); BATCH],
            ),
            |mut acc, (idx, x)| {
                //(state, action, reward, new_state, done, value) = x;
                acc.0[idx] = x.0.clone();
                acc.1[idx] = x.1.clone();
                acc.2[idx] = x.2.clone();
                acc.3[idx] = x.3.clone();
                acc.4[idx] = x.4.clone();
                // acc.5[idx] = x.5.clone();
                acc
            },
        )
    }

    pub fn sample<const BATCH: usize>(&mut self) -> Sample<S, A, R, N, D> {
        // do not take the last element, because it does not have calculated return
        let memory_len = self.memory.len();
        let array = [(); BATCH].map(|_| {
            let idx = fastrand::usize(0..memory_len);
            let d = self.memory.get(idx).unwrap();
            d
        });
        //let (states, actions, rewards, dones)
        // let (states, actions, rewards, next_states, dones, values) =
        self.sample_from_iter::<BATCH>(array.iter())
    }
    pub fn len(&self) -> usize {
        self.memory.len()
    }
}
