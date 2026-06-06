# agent-epoch-counter

Named counters that reset on epoch boundaries, for per-session, per-turn, or
per-window event tracking in LLM agents and other long-running loops.

An **epoch** is any logical unit boundary you choose — a session, a batch, a
model turn, or a sliding window. Within an epoch you count events such as token
usage, tool calls, retries, or errors. When the epoch advances, the current
counters reset to zero while a separate **lifetime** total keeps accumulating
across resets. This makes it easy to enforce per-epoch limits (e.g. "no more
than 20 tool calls per turn") without losing the cumulative picture.

The crate has **no dependencies** beyond the standard library.

## Features

- Named `u64` counters keyed by string.
- `inc` / `add` to count events.
- `reset_epoch` clears the current-epoch counters and advances the epoch number.
- `reset_one` clears a single counter.
- Per-epoch values via `get`, plus `lifetime` totals that survive resets.
- Convenience helpers: `total`, `active_names`, `exceeds`, `at_or_above`.

## Installation

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
agent-epoch-counter = "0.1"
```

Or with cargo:

```sh
cargo add agent-epoch-counter
```

## Usage

```rust
use agent_epoch_counter::EpochCounters;

let mut c = EpochCounters::new();

// Count events within the current epoch.
c.inc("tool_calls");
c.inc("tool_calls");
c.add("tokens", 150);

assert_eq!(c.get("tool_calls"), 2);
assert_eq!(c.get("tokens"), 150);

// Enforce a per-epoch limit.
if c.exceeds("tool_calls", 20) {
    // back off...
}

// Advance to the next epoch: current counters reset, lifetime persists.
c.reset_epoch();
assert_eq!(c.get("tool_calls"), 0);
assert_eq!(c.lifetime("tokens"), 150);
assert_eq!(c.epoch(), 1);
```

## API overview

| Method | Description |
| --- | --- |
| `new()` | Create an empty counter store. |
| `inc(name)` | Increment a counter by 1. |
| `add(name, n)` | Add `n` to a counter. |
| `get(name)` | Current-epoch value (0 if unset). |
| `lifetime(name)` | Cumulative value across all epochs. |
| `epoch()` | Current epoch number. |
| `reset_epoch()` | Clear current counters and advance the epoch. |
| `reset_one(name)` | Reset a single counter. |
| `active_names()` | Names with entries in the current epoch. |
| `total()` | Sum of all current-epoch counters. |
| `exceeds(name, limit)` | True if the counter is strictly above `limit`. |
| `at_or_above(name, limit)` | True if the counter is at or above `limit`. |

## Building and testing

```sh
cargo build
cargo test
```

## Tech stack

- **Language:** Rust (edition 2021)
- **Dependencies:** none (standard library only)

## License

Licensed under the [MIT License](LICENSE).
