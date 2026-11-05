//! The final step: the soft limiter (threshold 0.85, limit 0.95). The signal is untouched up to the threshold, and above it a
//! tanh knee bends it towards the limit. The arithmetic is float32, step by step as numpy does it.

pub const THRESHOLD: f32 = 0.85;
const HEADROOM: f32 = (0.95f64 - 0.85f64) as f32; // computed in float64, then taken as float32

pub fn soft_limit(y: &mut [f32]) {
    for v in y {
        let a = v.abs();
        if a > THRESHOLD {
            *v = (THRESHOLD + HEADROOM * ((a - THRESHOLD) / HEADROOM).tanh()).copysign(*v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn untouched_below_the_threshold_and_bounded_above() {
        let mut y = [0.0, 0.5, -0.85, 0.85, 0.9, -0.9, 3.0, -50.0];
        soft_limit(&mut y);
        assert_eq!(y[..4], [0.0, 0.5, -0.85, 0.85]);
        assert!(y[4] > 0.85 && y[4] < 0.95 && y[5] < -0.85 && y[5] > -0.95);
        assert!(y[6] <= 0.95 + 1e-6 && y[7] >= -0.95 - 1e-6);
    }
}
