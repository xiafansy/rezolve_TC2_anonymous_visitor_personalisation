# Changelog

## v3.1 — correctness pass, test suite, reproducible runs

The engine's logic is unchanged: same intents, same rules, same weights, same
two stages. What changed is that four of them now actually do what the
docstrings say, and there is a test suite that would have caught it.

### Fixed

**1. `views_before_first_commercial == 0` never fired the Decisive override.**
`intent_engine.score_aggregates` read features through

```python
g = lambda k, d=0.0: (f.get(k) if f.get(k) == f.get(k) else d) or d
```

The trailing `or d` was meant to be a NaN guard, but it also swallows a
legitimate zero. Since that key defaults to `99`, a session that added to cart
with **zero** prior views — the visitor arriving already decided, the highest
converting segment in either dataset — read as `99`, failed the `<= 2` test,
fell through to the micro-visit branch and was served the **Low-intent** page.
Replaced with `_num()`, which distinguishes "missing" from "zero".

Consequence for the published tables: on RetailRocket and REES46, Decisive
coverage was under-counted and Low-intent's purchase rate was inflated by the
cart-first sessions hiding inside it. Both real-data tables need a re-run —
see *Numbers to regenerate* in the README.

**2. `gate=0.0` silently became `gate=0.45`.**
`self.gate = gate or self.DEFAULTS["gate"]` treats an explicit 0.0 as absent.
`evaluate_synthetic.py` built its calibration engine with `gate=0.0` and a
comment saying `# ungated pass` — it was gated at 0.45, so every confidence
below 0.45 arrived pre-labelled `Unclear`, was scored as **wrong**, and the
gate was fitted against its own output. Same bug for `decay=0.0` and
`cold_gate=0.0`. All four now use `is None`; `temperature` and `decay` are
range-checked.

With a genuinely ungated fit split, the fitted gate drops **0.41 → 0.35** and
end-of-session decided accuracy reads **84.5%**, not 86% — the honest number.

**3. The censoring assertion could not fail.** `smoke_test_real_pipeline.py`
asserted `views_before_first_commercial <= n_views + n_events_total`, true for
any three non-negative numbers. The leakage fix — this project's central
methodological claim — had no real test. It now re-derives the expected
pre-commercial counts directly from the fabricated `events.csv` and demands
equality, and the fixture gained a cart-first archetype so the override path
above is exercised end to end.

**4. At arrival, the engine and the renderer disagreed about what to serve.**
`current()` returned `served=<winning intent>` as soon as the cold gate was
cleared -- which at the fitted `T=1.0` happens routinely, so the engine told
callers to serve the full **Low-intent** page off arrival context alone.
`personalisation.render()` ignored that and drew a neutral cold-accent page,
so the shipped demo was right by accident and any other consumer of
`Inference.served` was wrong. In cold mode `served` is now always
`Neutral (accent: <top substantive intent>)`, using the same accent rule as
`render()`. `intent` still carries the prior's argmax, so prefix-accuracy at
checkpoint 0 is unchanged.

### Changed

- **Fitted parameters flow from the fit.** `evaluate_synthetic.py` writes
  `engine_params.json`; `demo.py` reads it. They were hand-copied and had
  already drifted (demo said `gate=0.43`, the fit said `0.41`, the truth is
  `0.35`).
- **Deterministic fit/test split.** The split walked `random.random()` down
  the frame, so regenerating the dataset at a different size reshuffled which
  sessions were held out. Now hashed per `session_id`.
- `state_evidence()` was recomputed three times per decision, on every event
  of every session; computed once and shared.
- `fit_gate()` no longer sorts pointlessly, handles empty input, and returns a
  usable fallback when the precision target is unreachable at any threshold.
- Removed a dead `TwoStageEngine()` construction in `evaluate_synthetic.py`.

### Added

- **`tests/` — 102 unit tests**, none before. Rule behaviour, decay, the idle
  cap, the sticky price trait, the override boundary (0/1/2 prior views in,
  3 out), gate behaviour, softmax/temperature/ECE/gate-fitting properties,
  layout completeness for every intent in the taxonomy, and the censoring
  logic checked against hand-computed values on a hand-written event log.
  The four fixes above each have a named regression guard that fails on the
  previous code.
- **`run_all.py`** — reproduces everything that needs no download, with the
  interpreter that runs it. The README's `/usr/bin/python3` paths worked on
  one machine.
- **`requirements.txt`**, `pytest.ini`, and **GitHub Actions CI** running the
  suite plus the smoke test on Python 3.10 and 3.12.

### Not changed

No rule weight, threshold, intent or layout was touched. The point of this
pass was to make the published behaviour match the described behaviour, not
to chase a better number.
