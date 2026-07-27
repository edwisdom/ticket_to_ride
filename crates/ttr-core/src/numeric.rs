//! Float arithmetic that has to match CPython bit for bit.
//!
//! One function, and it exists because of a trap that costs an afternoon if you meet it
//! cold: **CPython's builtin `sum()` is not naive summation.** Since 3.12 it has a fast
//! path for floats using Neumaier compensated summation, while Rust's `Iterator::sum` adds
//! left to right with no compensation. They differ in the last ULP on perfectly ordinary
//! inputs -- `sum([0.0, 1.0, 2/3, 1/3])` is exactly `2.0` in Python and
//! `1.9999999999999998` in Rust -- and the gap propagates into every value derived from
//! the mean.
//!
//! The giveaway probe: `sum([1e16, 1.0, -1e16])` is `1.0` in Python and `0.0` naively.
//!
//! Matching Python rather than the other way round, for two reasons. The Python engine is
//! the oracle, so the port follows it; and compensated summation is strictly *more*
//! accurate, so this is not a fidelity-for-parity trade.

/// Neumaier compensated summation -- CPython's `sum()` over floats, transcribed.
///
/// Tracks the low-order bits lost at each addition in a running compensation term and
/// folds them back in at the end. The branch on magnitude is what makes it Neumaier's
/// variant rather than Kahan's: it is also correct when the incoming value is larger than
/// the accumulator, which Kahan's original is not.
pub fn compensated_sum(values: &[f64]) -> f64 {
    let mut sum = 0.0f64;
    let mut compensation = 0.0f64;
    for &x in values {
        let t = sum + x;
        if sum.abs() >= x.abs() {
            compensation += (sum - t) + x;
        } else {
            compensation += (x - t) + sum;
        }
        sum = t;
    }
    sum + compensation
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pythons_sum_where_naive_addition_does_not() {
        // The exact case that made the differential harness fail on `returns()`: four
        // ordinary rank values whose naive sum is one ULP short of 2.0.
        let values = [0.0, 1.0, 0.666_666_666_666_666_6, 0.333_333_333_333_333_3];
        let naive: f64 = values.iter().sum();
        assert_eq!(compensated_sum(&values), 2.0);
        assert_ne!(
            naive, 2.0,
            "if naive summation now agrees, this test proves nothing"
        );
    }

    #[test]
    fn recovers_the_bit_naive_summation_loses() {
        // The classic probe. Python's `sum` gives 1.0 here; a plain fold gives 0.0.
        let values = [1e16, 1.0, -1e16];
        assert_eq!(compensated_sum(&values), 1.0);
        assert_eq!(values.iter().sum::<f64>(), 0.0);
    }

    #[test]
    fn is_exact_on_inputs_where_naive_summation_already_is() {
        assert_eq!(compensated_sum(&[]), 0.0);
        assert_eq!(compensated_sum(&[1.5]), 1.5);
        assert_eq!(compensated_sum(&[0.25, 0.25, 0.5]), 1.0);
    }
}
