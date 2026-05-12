# 🐍 BAKOME Python Gold Scalper

## *Professional ICT‑based Trading System for XAUUSD – Python, Backtesting, Drips Integration*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![Drips](https://img.shields.io/badge/Drips-Fundraising-4E44CE?logo=ethereum&logoColor=white)](https://drips.network)
[![XAUUSD](https://img.shields.io/badge/Symbol-XAUUSD-F7931A?logo=gold&logoColor=white)](https://www.investopedia.com/terms/x/xauusd.asp)

---

## 🚀 Why This Project Exists

| Problem | Solution |
|---------|----------|
| Most Gold EAs are closed source | **100% open source (MIT)** |
| Backtesting is hidden or fake | **Transparent Python backtest engine** |
| No fundraising for developers | **Drips integration – crypto donations directly to your wallet** |
| MQL5 is proprietary | **Python version – accessible to all** |

---

## 🔥 Features

| Feature | Description |
|---------|-------------|
| **ICT Concepts** | FVG detection, Order Blocks, Liquidity Sweeps, Silver Bullet windows |
| **Risk Management** | ATR‑based position sizing, trailing stop, break‑even, daily limits |
| **Session Filters** | London (8‑9h) & New York (15‑16h) – Asian optional |
| **Kill Zones** | Silver Bullet windows (London 8‑9h, NY 15‑16h) |
| **Backtest Engine** | Realistic simulation with win rate, Sharpe ratio, max drawdown |
| **Drips Integration** | Fetch and display fundraising stats from Drips network |
| **Modular Code** | 1800+ lines, fully commented, easy to extend |

---

## 📊 Backtest Results (Example)

| Metric | Value |
|--------|-------|
| **Total Trades** | 342 |
| **Win Rate** | 68.7% |
| **Profit Factor** | 1.82 |
| **Max Drawdown** | 12.4% |
| **Sharpe Ratio** | 1.45 |

*Results from synthetic data – replace with real XAUUSD M5 data for actual performance.*

---

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- pip

### Steps

```bash
git clone https://github.com/BAKOME-Hub/BAKOME_Python_Gold_Scalper.git
cd BAKOME_Python_Gold_Scalper
pip install pandas numpy matplotlib requests
python bakome_gold_scalper.py
