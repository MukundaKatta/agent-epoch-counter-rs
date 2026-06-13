/*!
`agent-epoch-counter`: named counters that reset on epoch boundaries.

Useful for per-session, per-turn, or per-window counting of events like
token usage, tool calls, retries, or errors. An "epoch" is any logical
unit boundary — a session, a batch, a model turn, or a sliding window.

Within an epoch you count events with [`EpochCounters::inc`] or
[`EpochCounters::add`]. When the epoch advances via
[`EpochCounters::reset_epoch`], the current counters reset to zero while a
separate **lifetime** total keeps accumulating across resets. This makes it
easy to enforce per-epoch limits (e.g. "no more than 20 tool calls per turn")
without losing the cumulative picture.

All counters are [`u64`] and saturate at [`u64::MAX`] rather than overflowing,
so a long-running agent will never panic (in debug builds) or silently wrap
(in release builds) on a busy counter.

```rust
use agent_epoch_counter::EpochCounters;

let mut c = EpochCounters::new();
c.inc("tool_calls");
c.inc("tool_calls");
c.add("tokens", 150);
assert_eq!(c.get("tool_calls"), 2);
assert_eq!(c.get("tokens"), 150);

c.reset_epoch();
assert_eq!(c.get("tool_calls"), 0); // current-epoch value reset
assert_eq!(c.lifetime("tokens"), 150); // lifetime persists
assert_eq!(c.epoch(), 1);
```
*/

#![forbid(unsafe_code)]

use std::collections::HashMap;

/// Named counter store with epoch-based resets.
///
/// See the [crate-level documentation](crate) for an overview.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct EpochCounters {
    counters: HashMap<String, u64>,
    epoch: u64,
    lifetime: HashMap<String, u64>, // cumulative across resets
}

impl EpochCounters {
    /// Create an empty counter store at epoch `0`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Increment a counter by 1.
    ///
    /// Equivalent to `add(name, 1)`. The value saturates at [`u64::MAX`].
    pub fn inc(&mut self, name: impl Into<String>) {
        self.add(name, 1);
    }

    /// Add `n` to a counter, updating both the current-epoch and lifetime
    /// totals.
    ///
    /// Both totals saturate at [`u64::MAX`] instead of overflowing, so this
    /// never panics in debug builds or wraps silently in release builds.
    pub fn add(&mut self, name: impl Into<String>, n: u64) {
        let name = name.into();
        let cur = self.counters.entry(name.clone()).or_insert(0);
        *cur = cur.saturating_add(n);
        let life = self.lifetime.entry(name).or_insert(0);
        *life = life.saturating_add(n);
    }

    /// Subtract `n` from the current-epoch value of a counter, saturating at
    /// `0`.
    ///
    /// Returns the new current-epoch value. The lifetime total is left
    /// unchanged, since lifetime totals are intended to be monotonic.
    /// Subtracting from a counter that does not exist is a no-op and returns
    /// `0`.
    pub fn sub(&mut self, name: &str, n: u64) -> u64 {
        if let Some(cur) = self.counters.get_mut(name) {
            *cur = cur.saturating_sub(n);
            *cur
        } else {
            0
        }
    }

    /// Current epoch value for a counter (resets on a new epoch).
    ///
    /// Returns `0` for counters that have never been set or were reset.
    pub fn get(&self, name: &str) -> u64 {
        self.counters.get(name).copied().unwrap_or(0)
    }

    /// Lifetime value of a counter, summed across all epochs.
    ///
    /// Returns `0` for counters that have never been set.
    pub fn lifetime(&self, name: &str) -> u64 {
        self.lifetime.get(name).copied().unwrap_or(0)
    }

    /// Current epoch number. Starts at `0` and advances by 1 on every
    /// [`reset_epoch`](Self::reset_epoch).
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Reset all current-epoch counters and advance the epoch number by 1.
    ///
    /// Lifetime totals are preserved.
    pub fn reset_epoch(&mut self) {
        self.counters.clear();
        self.epoch = self.epoch.saturating_add(1);
    }

    /// Reset a single counter's current-epoch value to `0`.
    ///
    /// The lifetime total for the counter is preserved. This does not advance
    /// the epoch.
    pub fn reset_one(&mut self, name: &str) {
        self.counters.remove(name);
    }

    /// Remove a counter entirely, discarding both its current-epoch value and
    /// its lifetime total.
    ///
    /// Returns the lifetime total that was removed, or `0` if the counter did
    /// not exist.
    pub fn remove(&mut self, name: &str) -> u64 {
        self.counters.remove(name);
        self.lifetime.remove(name).unwrap_or(0)
    }

    /// Names of counters with a non-zero entry in the current epoch.
    ///
    /// The order is unspecified.
    pub fn active_names(&self) -> Vec<&str> {
        self.counters.keys().map(String::as_str).collect()
    }

    /// Iterate over `(name, value)` pairs for the current epoch.
    ///
    /// The order is unspecified.
    pub fn iter(&self) -> impl Iterator<Item = (&str, u64)> {
        self.counters.iter().map(|(k, v)| (k.as_str(), *v))
    }

    /// Number of distinct counters with an entry in the current epoch.
    pub fn len(&self) -> usize {
        self.counters.len()
    }

    /// True if no counter has an entry in the current epoch.
    pub fn is_empty(&self) -> bool {
        self.counters.is_empty()
    }

    /// Sum of all counter values in the current epoch.
    ///
    /// The sum saturates at [`u64::MAX`].
    pub fn total(&self) -> u64 {
        self.counters
            .values()
            .copied()
            .fold(0u64, u64::saturating_add)
    }

    /// Sum of all lifetime totals across every counter.
    ///
    /// The sum saturates at [`u64::MAX`].
    pub fn lifetime_total(&self) -> u64 {
        self.lifetime
            .values()
            .copied()
            .fold(0u64, u64::saturating_add)
    }

    /// True if a counter's current-epoch value is strictly greater than
    /// `limit`.
    pub fn exceeds(&self, name: &str, limit: u64) -> bool {
        self.get(name) > limit
    }

    /// True if a counter's current-epoch value is at or above `limit`.
    pub fn at_or_above(&self, name: &str, limit: u64) -> bool {
        self.get(name) >= limit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inc_and_get() {
        let mut c = EpochCounters::new();
        c.inc("calls");
        c.inc("calls");
        assert_eq!(c.get("calls"), 2);
    }

    #[test]
    fn add_bulk() {
        let mut c = EpochCounters::new();
        c.add("tokens", 500);
        assert_eq!(c.get("tokens"), 500);
    }

    #[test]
    fn unknown_counter_is_zero() {
        let c = EpochCounters::new();
        assert_eq!(c.get("never_set"), 0);
        assert_eq!(c.lifetime("never_set"), 0);
    }

    #[test]
    fn reset_epoch_clears_counters() {
        let mut c = EpochCounters::new();
        c.inc("x");
        c.reset_epoch();
        assert_eq!(c.get("x"), 0);
    }

    #[test]
    fn reset_epoch_advances_epoch() {
        let mut c = EpochCounters::new();
        c.reset_epoch();
        assert_eq!(c.epoch(), 1);
    }

    #[test]
    fn lifetime_persists_across_resets() {
        let mut c = EpochCounters::new();
        c.add("tokens", 100);
        c.reset_epoch();
        c.add("tokens", 200);
        assert_eq!(c.lifetime("tokens"), 300);
    }

    #[test]
    fn reset_one_clears_single() {
        let mut c = EpochCounters::new();
        c.inc("a");
        c.inc("b");
        c.reset_one("a");
        assert_eq!(c.get("a"), 0);
        assert_eq!(c.get("b"), 1);
    }

    #[test]
    fn reset_one_preserves_lifetime() {
        let mut c = EpochCounters::new();
        c.add("a", 5);
        c.reset_one("a");
        assert_eq!(c.get("a"), 0);
        assert_eq!(c.lifetime("a"), 5);
    }

    #[test]
    fn total_sums_all() {
        let mut c = EpochCounters::new();
        c.add("a", 3);
        c.add("b", 7);
        assert_eq!(c.total(), 10);
    }

    #[test]
    fn lifetime_total_sums_across_counters_and_epochs() {
        let mut c = EpochCounters::new();
        c.add("a", 3);
        c.add("b", 7);
        c.reset_epoch();
        c.add("a", 5);
        // current epoch only has a=5
        assert_eq!(c.total(), 5);
        // lifetime: a=8, b=7
        assert_eq!(c.lifetime_total(), 15);
    }

    #[test]
    fn exceeds() {
        let mut c = EpochCounters::new();
        c.add("calls", 5);
        assert!(c.exceeds("calls", 4));
        assert!(!c.exceeds("calls", 5));
    }

    #[test]
    fn at_or_above() {
        let mut c = EpochCounters::new();
        c.add("x", 3);
        assert!(c.at_or_above("x", 3));
        assert!(!c.at_or_above("x", 4));
    }

    #[test]
    fn active_names_and_len() {
        let mut c = EpochCounters::new();
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);
        c.inc("a");
        c.inc("b");
        let mut names = c.active_names();
        names.sort_unstable();
        assert_eq!(names, vec!["a", "b"]);
        assert_eq!(c.len(), 2);
        assert!(!c.is_empty());
    }

    #[test]
    fn iter_yields_all_pairs() {
        let mut c = EpochCounters::new();
        c.add("a", 1);
        c.add("b", 2);
        let mut pairs: Vec<(&str, u64)> = c.iter().collect();
        pairs.sort_unstable();
        assert_eq!(pairs, vec![("a", 1), ("b", 2)]);
    }

    #[test]
    fn sub_saturates_at_zero() {
        let mut c = EpochCounters::new();
        c.add("x", 3);
        assert_eq!(c.sub("x", 1), 2);
        assert_eq!(c.get("x"), 2);
        assert_eq!(c.sub("x", 10), 0);
        assert_eq!(c.get("x"), 0);
    }

    #[test]
    fn sub_missing_counter_is_noop() {
        let mut c = EpochCounters::new();
        assert_eq!(c.sub("nope", 5), 0);
        assert_eq!(c.get("nope"), 0);
    }

    #[test]
    fn sub_does_not_touch_lifetime() {
        let mut c = EpochCounters::new();
        c.add("x", 10);
        c.sub("x", 4);
        assert_eq!(c.get("x"), 6);
        assert_eq!(c.lifetime("x"), 10);
    }

    #[test]
    fn remove_discards_both_totals() {
        let mut c = EpochCounters::new();
        c.add("x", 4);
        c.reset_epoch();
        c.add("x", 6);
        // lifetime is 10
        assert_eq!(c.remove("x"), 10);
        assert_eq!(c.get("x"), 0);
        assert_eq!(c.lifetime("x"), 0);
    }

    #[test]
    fn remove_missing_returns_zero() {
        let mut c = EpochCounters::new();
        assert_eq!(c.remove("nope"), 0);
    }

    #[test]
    fn add_saturates_instead_of_overflowing() {
        let mut c = EpochCounters::new();
        c.add("big", u64::MAX);
        c.add("big", 100);
        assert_eq!(c.get("big"), u64::MAX);
        assert_eq!(c.lifetime("big"), u64::MAX);
    }

    #[test]
    fn total_saturates() {
        let mut c = EpochCounters::new();
        c.add("a", u64::MAX);
        c.add("b", 50);
        assert_eq!(c.total(), u64::MAX);
    }

    #[test]
    fn multiple_epochs() {
        let mut c = EpochCounters::new();
        c.inc("x");
        c.reset_epoch();
        c.inc("x");
        c.reset_epoch();
        assert_eq!(c.epoch(), 2);
        assert_eq!(c.get("x"), 0);
        assert_eq!(c.lifetime("x"), 2);
    }

    #[test]
    fn clone_and_eq() {
        let mut c = EpochCounters::new();
        c.add("x", 3);
        let d = c.clone();
        assert_eq!(c, d);
    }
}
