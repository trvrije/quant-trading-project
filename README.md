# Market Maker Simulation (Multi-Asset Synthetic LOB)

A self-contained Python project that builds a synthetic, multi-asset limit order book and a configurable market-making agent. It demonstrates practical microstructure modelling, inventory/risk management, and performance measurement in a reproducible, object-oriented codebase.

## Summary

* Implements a full market-making stack: price process → order book → order flow → MM quoting/hedging → execution → PnL & risk.
* Models realistic features (OU/GBM, Student-t shocks, GARCH, jumps, intraday U-shape, Hawkes arrivals, flow impact, latency/staleness, inventory constraints).
* Clean OOP design with testable components, reproducible configs, and interpretable diagnostics/metrics.

## What’s included

**Core classes**

* `PriceProcess` — multi-asset midprice engine: OU or GBM; Student-t innovations; optional GARCH(1,1); Merton jumps; intraday U-shape; scheduled “event” spikes; regime switching; optional ETF AP tether.
* `OrderBook` — simple L2 limit order book with tick snapping, price-time priority, pro-rata market sweeps, background liquidity, and depth/imbalance queries.
* `MarketMaker` — Avellaneda–Stoikov-style quoting with volatility- and inventory-aware spreads, inventory skew, laddered quotes, IOC hedging, taker-unwind logic, per-share & bps fee model.
* `MarketSimulation` — orchestrates steps: evolve prices → place quotes → generate external orders (Poisson + Hawkes + trend/vol tilts) → match → apply impact/mark-out → update inventory/cash → record PnL & diagnostics.

**Metrics & diagnostics**

* Equity (conservative and mid-mark), realized vs. revaluation PnL, fees/rebates decomposition.
* Daily returns, annualized Sharpe/Sortino, max drawdown, turnover, maker/taker notional, hit-ratio with k-step mark-out.
* Optional progress logs and a simple Matplotlib PnL plot.

## Repository layout

Only the simulator and this README are exposed publicly.

```
.
├─ src/
│  └─ strategies/
│     └─ market_maker_sim.py
└─ README.md
```

## Quick start

### Requirements

* Python 3.10+
* Packages: `numpy`, `pandas`, `matplotlib`

```bash
pip install numpy pandas matplotlib
```

### Run

From repository root:

```bash
python src/strategies/market_maker_sim.py
```

The script contains a `__main__` block with a default configuration. It runs a simulation, prints periodic diagnostics (if `verbose=True`), computes performance, and shows a PnL plot.

> Tip: The default example may be long (seconds-granularity for multiple weeks). For a quick demo, lower `steps`, raise `dt`, or reduce the symbol list in the config at the bottom of the file.

## Configuration (overview)

All components are driven by a single `SimulationConfig` dataclass. The `__main__` section shows an example configuration. Key knobs:

* **Global:** `symbols`, `steps`, `dt`, `model` (`"OU"` or `"GBM"`), `tick_size`, `seed`.
* **Market maker (`mm_params`):** `spread`, `size`, `risk_aversion`, `vol_sensitivity`, `inventory_limit`, `inventory_skew_gamma`, `levels`/`level_step_ticks`/`depth_decay`, `taker_unwind_*`, `quote_update_cost_ps`, `ioc_cooldown_steps`.
* **Price process (`process_params`):** `mu/vol` (GBM), `theta/kappa/sigma` (OU), `correlation`, `innovations` (`"gaussian"` or `"student_t"`), `df`, `garch_*`, `jump_*`, `intraday_u_shape` & `u_open_close/u_midday`, `event_times_steps` & `event_sigma_mult`, `regime_on` & `regimes`, micro/flow impact controls, background book liquidity, maker mark-out penalties, anchor/impact decay.
* **External flow (`external_order_params`):** baseline intensity `lambda`, size mix (`retail_mu/inst_mu/retail_frac`), buy probability & aggression tilts (`use_price_bias`, `bias_mode`, `trend_eta`, `vol_eta`, `mkt_*`), Hawkes (`hawkes_*`).
* **Fees:** `fee_rate` (taker bps), `rebate_rate` (maker bps), `routing_fee_rate`, and optional per-share fees (`per_share_fees`).
* **ETF AP (optional):** `etf_ap.enabled`, premium threshold, tether strength, costs.

Configurations are plain Python dicts; pass them into `SimulationConfig(**config)`.

## How the simulation works (high level)

1. **Price dynamics:** For each step, `PriceProcess.step()` updates mids jointly using the chosen model (OU/GBM). Variance can evolve with GARCH; jumps/events/regimes adjust shock statistics; an intraday factor reproduces U-shaped volatility.

2. **Order environment:** `OrderBook` maintains two-sided depth with tick rounding and price-time priority. Background “passive” depth can be injected to avoid empty books.

3. **External flow:** Per symbol, arrivals are Poisson with Hawkes self/cross-excitation. Trend and vol tilt both the buy/sell split and the fraction that is marketable. Sizes are drawn from a retail/institutional mix. Marketables sweep pro-rata at the touch; limits rest at offset ticks.

4. **Quoting & inventory:** `MarketMaker.quote_ladder()` builds a ladder of quotes. The spread scales with (i) current daily vol, (ii) inventory pressure, and (iii) base spread. A skew term shifts the center to lean against inventory. IOC hedges and taker-unwind guards reduce inventory when limits or adverse conditions are hit.

5. **Impact/mark-out:** Marketable trades can move the synthetic mid via micro/flow impact; stale maker fills penalize the MM (adverse selection). A subsequent “maker drag” and anchor decay back to the process keep mids realistic.

6. **PnL & risk:** Cash and inventory update on every fill. Fees/rebates include both per-notional bps and per-share components (incl. FINRA TAF placeholder). Equity is tracked with a conservative mark (haircuts on bid/ask). The report includes Sharpe/Sortino, max drawdown, turnover, and hit-ratio.

## Limitations

* Synthetic order flow; calibration is illustrative, not instrument-grade.
* Matching is simplified relative to real venues; no hidden/priority rules beyond basic price-time.
* Impact/latency models are stylized; no cross-venue routing or queue position modelling.
* No persistence layer; runs are in-memory with optional Matplotlib output.
* Simplified hedging; project is illustrative, not a full recreation of true MM operations & environment.
* Simplified ETF creation & redemption mechanisms. Can be turned off. 


