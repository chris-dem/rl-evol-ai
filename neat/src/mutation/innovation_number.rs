#[derive(Debug, Clone ,Copy, Default)]
pub struct InnovNumber {
    curr_innov: usize
}

impl InnovNumber {
    pub fn next(&mut self) -> usize {
        self.curr_innov += 1;
        self.curr_innov
    }
}