use std::path::Path;

use self::consts::StateTensor;

pub mod consts;
pub mod conv;
pub mod dfdx;
pub mod memory;
// pub mod tch;

pub trait Agent {
    fn name(&self) -> String;
    fn choose_action(&mut self, state: &StateTensor, train: bool) -> usize;
    fn choose_random_action(&mut self) -> usize;
    fn update(
        &mut self,
        state: StateTensor,
        action: usize,
        reward: f32,
        next_state: StateTensor,
        done: f32,
    );
    fn step_finished(&mut self, step: usize);
    fn episode_started(&mut self, _ep: usize) {
        ()
    }
    fn episode_finished(&mut self, _ep: usize) {
        ()
    }
    fn info(&mut self) -> String;

    fn save<P: AsRef<Path>>(&self, path: P);
    fn load<P: AsRef<Path>>(&mut self, path: P);
}

// pub type MemoryType = (StateTensor, i64, f32, StateTensor, f32);
// pub type Sample = (
//     [StateTensor; BATCH],
//     [usize; BATCH],
//     [f32; BATCH],
//     [StateTensor; BATCH],
//     [f32; BATCH],
// );

// pub struct Memory {
//     memory: VecDeque<MemoryType>,
//     memory_size: usize,
//     rng: rand::rngs::ThreadRng,
// }
// impl Memory {
//     pub fn new<const MS: usize>() -> Self {
//         let rng = thread_rng();
//         Self {
//             memory: VecDeque::<MemoryType>::with_capacity(MS),
//             memory_size: MS,
//             rng,
//         }
//     }
//     pub fn remember(&mut self, memory: MemoryType) {
//         if self.memory.len() > 0 && self.memory.len() >= self.memory_size {
//             let _ = self.memory.pop_front();
//         }
//         self.memory.push_back(memory);
//     }

//     pub fn sample<const BATCH: usize>(&mut self) -> Sample {
//         let memory_len = self.memory.len();

//         //let (states, actions, rewards, dones)
//         let (states, actions, rewards, next_states, dones) = [(); consts::BATCH]
//             .map(|_| {
//                 let idx = self.rng.gen_range(0..memory_len);
//                 self.memory.get(idx).unwrap()
//             })
//             .iter()
//             .enumerate()
//             .fold(
//                 (vec![], vec![], vec![], vec![], vec![]),
//                 |mut acc, (_idx, x)| {
//                     acc.0.push(x.0.clone());
//                     // acc.0[idx] = x.0.clone();
//                     acc.1.push(x.1 as usize);
//                     acc.2.push(x.2);
//                     acc.3.push(x.3.clone());
//                     acc.4.push(x.4);
//                     acc
//                 },
//             );

//         (
//             to_array(states),
//             to_array(actions),
//             to_array(rewards),
//             to_array(next_states),
//             to_array(dones),
//         )
//     }
//     pub fn len(&self) -> usize {
//         self.memory.len()
//     }
// }
