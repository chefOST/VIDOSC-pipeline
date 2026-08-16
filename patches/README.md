# Patches to upstream VidOSC

Local modifications to [facebookresearch/VidOSC](https://github.com/facebookresearch/VidOSC),
kept as patches so this repo stays free of upstream code.

Generated against commit `59575773c97878ea30a0228f28ab00a3ee2f1ea2`.

## Applying

```bash
git -C /path/to/VidOSC apply /path/to/VIDOSC-pipeline/patches/*.patch
```

Check before applying, and reverse if needed:

```bash
git -C /path/to/VidOSC apply --check  .../patches/*.patch
git -C /path/to/VidOSC apply --reverse .../patches/*.patch
```

## Regenerating

After editing files in the VidOSC checkout:

```bash
git -C /path/to/VidOSC diff -- task.py train.py dataset.py data_scripts/ \
    > /path/to/VIDOSC-pipeline/patches/0001-macos-and-multi-transition-states.patch
```

---

## `0001-macos-and-multi-transition-states.patch`

Touches `task.py`, `train.py`, `dataset.py`, `data_scripts/evaluator.py`,
`data_scripts/read_ann.py`. Two concerns, combined into one patch because
`task.py` contains hunks from both and they do not split cleanly.

### macOS / Apple Silicon support

- **`task.py`** — `import lookforthechange` wrapped in `try/except ImportError`
  with a `_HAS_LOOKFORTHECHANGE` flag. The upstream causal-ordering constraint
  is a Linux-only compiled CUDA extension that cannot build on macOS.
- **`train.py`** — `accelerator="auto"` so PyTorch Lightning selects MPS.

### Generalizing to N transitioning states

Upstream hardcodes 3 states (initial / transitioning / end) plus background.
These changes make the count configurable via `args.num_transition_states`
(default `1`, which reproduces upstream behavior).

- **`task.py`** — adds `ordered_state_indices()`, an O(T·S) dynamic program
  that enforces the temporal-ordering constraint for an arbitrary number of
  states. Used whenever `num_transition_states > 1`, since `lookforthechange`
  only handles the fixed 2-state + 1-action case. Vocab size and metric class
  counts derive from `states_per_category` instead of the literal `3`/`4`.
- **`dataset.py`** — drops the dead `derive_label` method that shadowed the
  import from `data_scripts.read_ann`; threads `num_transition_states` through.
- **`data_scripts/read_ann.py`** — `derive_label()` takes `num_transition_states`
  and subdivides the single annotated transitioning range into K equal segments,
  labeled `2..K+1` in temporal order.
- **`data_scripts/evaluator.py`** — `StatePrec1` and `EvalClip` take a state
  count instead of fixed-size tensors and hardcoded `s0`/`s1`/`s2` keys.
