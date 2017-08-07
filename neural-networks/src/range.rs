#[derive(Debug)]
pub struct Range {
    curr : usize,
    end : usize,
    step : usize,
}

impl Range {
    pub fn new (start: usize, end: usize, step: Option<usize>) -> Range {
        Range {
            curr: start,
            end: end,
            step: match step {
                Some(step_val) => step_val,
                None => 1,
            }
        }
    }
}

impl Iterator for Range {
    type Item = usize;
    fn next (&mut self) -> Option<usize> {
        let value = self.curr;
        self.curr += self.step;
        if value < self.end {
            return Some(value);
        } else {
            return None;
        }
    }
}
