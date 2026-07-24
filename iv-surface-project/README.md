# IV Surface Forecasting & Options Trading Strategy

Deep learning models (LSTM-Attention and ConvLSTM) that forecast the S&P 500 implied volatility surface and drive a systematic, risk-managed options trading strategy — benchmarked against a rule-based hedging approach.

## Highlights

- 🧠 Two deep sequence architectures — **LSTM with additive attention** and **ConvLSTM** — trained to forecast a 10-maturity × 18-delta implied volatility grid
- 📈 **26 years of daily SPX options data** (1996–2022), built into a no-arbitrage-aware surface representation with rolling context features
- 💰 Forecasts are wired into a **live trading strategy**: a 30-day ±10Δ short strangle with daily delta-hedging, VIX-based entry gates, stop-losses, and explicit transaction costs
- 🏆 **ConvLSTM strategy: 31.3% annualised return, 1.37 Sharpe ratio** (2004–2022 backtest) vs. 24.0% / 0.96 for the rule-based benchmark
- ✅ Walk-forward validation, arbitrage-penalty losses (MAE + RMSE + QLIKE), and stationary bootstrap robustness checks

## Why this project

Most implied volatility research stops at forecast accuracy. This project pushes further: it tests whether more accurate surface forecasts actually make money once realistic hedging, gating, and costs are included — and finds that the model with better *structural* consistency (ConvLSTM) outperforms in trading even when it isn't always the single most accurate model pointwise.

## Approach

1. **Data & grid construction** — raw SPX option quotes are pivoted into a fixed delta-tenor grid (10 maturities × 18 deltas), with light no-arbitrage smoothing and 5-/22-day rolling features for short/medium-term context.
2. **Modeling** — two architectures trained on walk-forward splits:
   - **LSTM + Attention**: learns to weight historically informative time windows, strong at picking up regime shifts.
   - **ConvLSTM**: treats the IV surface as an evolving spatio-temporal grid, convolving across neighbouring deltas/maturities to preserve cross-maturity coherence.
   - Loss combines MAE, RMSE, and QLIKE with a no-arbitrage penalty term.
3. **Evaluation** — MAE, RMSE, QLIKE, vega-weighted error, directional hit rate, and surface-shape diagnostics (skew/term-structure integrity), plus attention-map interpretability.
4. **Trading strategy** — a 30-day ±10Δ short strangle, hedged daily, with a VIX-based entry gate/cooldown, a notional stop-loss, and full cost accounting, run separately for realized-IV, LSTM-Attention, and ConvLSTM inputs.
5. **Backtesting & robustness** — common-window backtests (2004–2022), stationary bootstrap resampling, and sensitivity analysis over stop-loss, VIX threshold, hedge smoothing, and cost assumptions.

## Repository structure

```
.
├── src/
│   └── pipeline.py           # Full pipeline: data prep, models, training, backtest, plotting
├── docs/
│   ├── technical_report.pdf  # Full write-up: methodology, results, discussion, references
│   └── abstract.md           # Short summary of the project
└── README.md
```

## Results snapshot

| Strategy              | Annualised Return | Sharpe Ratio |
|------------------------|-------------------|--------------|
| ConvLSTM-driven hedging | **31.3%**         | **1.37**     |
| Rule-based (realized IV) hedging | 24.0%     | 0.96         |
| LSTM-Attention-driven hedging | 22.2%       | 0.89         |

ConvLSTM also showed the strongest resilience during high-VIX regimes.

## Running the code

`src/pipeline.py` contains the full pipeline (data prep → feature engineering → model training → backtest → plotting), originally developed and run in Google Colab. To run it locally:

1. Replace the `/content/drive/MyDrive/...` paths with your local data paths.
2. Install dependencies: `pandas`, `numpy`, `tensorflow`, `scikit-learn`, `matplotlib`.
3. Supply your own SPX option dataset (grid + features CSVs) in the expected format, or adapt the data-loading section to your source.

> Note: this file was reconstructed from the project's full code listing, so some deeply nested indentation may need small fixes before it runs end-to-end — treat it as the complete reference implementation rather than a plug-and-play script.

## Tech stack

Python · TensorFlow/Keras · pandas · NumPy · scikit-learn · matplotlib

## Full write-up

For the complete methodology, literature grounding, and detailed results/discussion, see [`docs/technical_report.pdf`](docs/technical_report.pdf).

## License

Add a license of your choice here (e.g. MIT). See [choosealicense.com](https://choosealicense.com/) for guidance.
