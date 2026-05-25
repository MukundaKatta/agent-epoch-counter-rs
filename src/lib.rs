/*!
agent-epoch-counter: named counters that reset on epoch boundaries.

Useful for per-session, per-turn, or per-window counting of events like
token usage, tool calls, retries, or errors. An "epoch" is any logical
unit boundary — a session, a batch, a model turn.

```rust
use agent_epoch_counter::EpochCounters;

let mut c = EpochCounters::new();
c.inc("tool_calls");
c.inc("tool_calls");
c.add("tokens", 150);
assert_eq!(c.get("tool_calls"), 2);
assert_eq!(c.get("tokens"), 150);
c.reset_epoch();
assert_eq!(c.get("tool_calls"), 0);
```
*/

use std::collections::HashMap;

/// Named counter store with epoch-based resets.
#[derive(Debug, Default)]
pub struct EpochCounters {
    counters: HashMap<String, u64>,
    epoch: u64,
    lifetime: HashMap<String, u64>, // cumulative across resets
}

impl EpochCounters {
    pub fn new() -> Self { Self::default() }

    /// Increment a counter by 1.
    pub fn inc(&mut self, name: impl Into<String>) {
        self.add(name, 1);
    }

    /// Add `n` to a counter.
    pub fn add(&mut self, name: impl Into<String>, n: u64) {
        let name = name.into();
        *self.counters.entry(name.clone()).or_insert(0) += n;
        *self.lifetime.entry(name).or_insert(0) += n;
    }

    /// Current epoch value for a counter (resets on new epoch).
    pub fn get(&self, name: &str) -> u64 {
        *self.counters.get(name).unwrap_or(&0)
    }

    /// Lifetime value across all epochs.
    pub fn lifetime(&self, name: &str) -> u64 {
        *self.lifetime.get(name).unwrap_or(&0)
    }

    /// Current epoch number.
    pub fn epoch(&self) -> u64 { self.epoch }

    /// Reset all current-epoch counters and advance the epoch.
    pub fn reset_epoch(&mut self) {
        self.counters.clear();
        self.epoch += 1;
    }

    /// Reset a single counter.
    pub fn reset_one(&mut self, name: &str) {
        self.counters.remove(name);
    }

    /// All counter names with non-zero values in current epoch.
    pub fn active_names(&self) -> Vec<&str> {
        self.counters.keys().map(String::as_str).collect()
    }

    /// Sum of all counters in current epoch.
    pub fn total(&self) -> u64 { self.counters.values().sum() }

    /// True if counter exceeds `limit`.
    pub fn exceeds(&self, name: &str, limit: u64) -> bool {
        self.get(name) > limit
    }

    /// True if counter is at or above `limit`.
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
    fn total_sums_all() {
        let mut c = EpochCounters::new();
        c.add("a", 3);
        c.add("b", 7);
        assert_eq!(c.total(), 10);
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
    fn active_names() {
        let mut c = EpochCounters::new();
        c.inc("a");
        c.inc("b");
        let names = c.active_names();
        assert_eq!(names.len(), 2);
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
}
