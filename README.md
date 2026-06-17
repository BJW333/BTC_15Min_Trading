SADLY THIS WAS JUST A IDEA AND DOESNT HAVE ENOUGH WIN RATE TO BE WORTH IT BUT EVENTUALLY COULD BE 

CURRENTLY DOESNT REALLY WORK

# BTC 15-Minute Prediction Market Signal Bot

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Polymarket](https://img.shields.io/badge/Polymarket-CLOB-purple)](https://polymarket.com)
[![Kalshi](https://img.shields.io/badge/Kalshi-KXBTC15M-green)](https://kalshi.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A signal engine for BTC 15-minute event contracts traded on **Robinhood** (routed through Kalshi/KalshiEX). Pulls live data from multiple prediction markets, runs ML-based predictive signals, and sends trade recommendations to your phone via Telegram every 15 seconds.

**This bot does not trade automatically.** It tells you what to buy and at what price — you place the trade on Robinhood.

---

## How It Works

Every 15 seconds, the bot:

1. Finds the current 15-minute BTC market on Polymarket and Kalshi
2. Gets live YES/NO prices from both platforms (Polymarket CLOB + Kalshi series `KXBTC15M`)
3. Fetches the BTC spot price and the contract's target/strike price
4. Runs three predictive signals — LSTM, Order Flow, and Cross-Asset Momentum
5. Recommends BUY YES or BUY NO with confidence level and signal agreement
6. Sends the full analysis to your phone via Telegram

### Example Output

```
======================================================================
BTC 15M RECOMMEND-ONLY  |  21:07:03 UTC
======================================================================
BTC: $80,793.01
Market: Bitcoin Up or Down - May 10, 5:00PM-5:15PM ET

  CLOB  (live):    YES=0.815  NO=0.185
  Kalshi/RH:       YES=0.815  NO=0.185
  Kalshi bid/ask:  YES: 0.81/0.82  NO: 0.18/0.19
  Target price:    $80,709.07

  LSTM:         NEUTRAL (63% chance up)
  OrderFlow:    UP (strength=0.84)
  CrossAsset:   UP (strength=0.93)

======================================================================
  RECOMMENDATION  (CLOB live)
======================================================================
  >>> BUY the YES contract
  Robinhood:   $0.82  (Kalshi ask)
  Confidence: 82%
======================================================================

  CONFIRMED: OrderFlow + CrossAsset agree → UP
```

---

## Features

| Feature | Description |
|---------|-------------|
| **Dual-Platform Pricing** | Live prices from both Polymarket CLOB and Kalshi (what you see on Robinhood) |
| **LSTM Neural Network** | PyTorch model trained on Coinbase 1-min candles, retrained daily |
| **Order Flow Analysis** | Buy/sell pressure, CVD, VWAP deviation, OBV slope, volume spikes |
| **Cross-Asset Momentum** | ETH/SOL divergence from BTC as a leading indicator |
| **Signal Agreement** | Plain English: CONFIRMED / MIXED / WARNING for every recommendation |
| **Telegram Alerts** | Full analysis sent to your phone every 15 seconds |
| **VPS Deployment** | Runs 24/7 on Oracle Cloud Free Tier as a systemd service |
| **Multi-Source BTC Price** | Coinbase → CoinGecko → Kraken fallback chain |
| **Signal Caching** | Predictive signals refresh every 60s to avoid rate limits |
| **Trade Logging** | Every recommendation saved to `contract_recommendations.jsonl` |

---

## Data Sources

| Source | What | Auth Required |
|---|---|---|
| Polymarket Gamma API | Market discovery, slug lookup, token IDs | No |
| Polymarket CLOB | Live YES/NO midpoint + buy prices (real-time, not cached) | No |
| Kalshi External API | Robinhood contract prices, bid/ask, target strike (series `KXBTC15M`) | No |
| Coinbase Spot | BTC/USD price with CoinGecko + Kraken fallback | No |
| Coinbase Candles | 1-min OHLCV for LSTM features, order flow, and cross-asset signals | No |

---

## Predictive Signals

**Order Flow** — The strongest short-term signal. Analyzes Coinbase 1-minute candles to measure real-time buy/sell pressure. Components: buy_ratio (% of volume on the buy side), cumulative volume delta (CVD), VWAP deviation, OBV slope, and volume spike detection. When buy_ratio drops to 12% with a 3x volume spike, the market is selling hard.

**Cross-Asset Momentum** — Compares ETH and SOL 5-minute returns against BTC. When alts diverge from BTC (e.g., ETH and SOL dropping while BTC is flat), it often signals a directional move. Useful as confirmation alongside order flow.

**LSTM Neural Network** — PyTorch model with 13 input features (8 price-derived + 5 order flow). Trained on 24 hours of Coinbase 1-minute candle data, retrained daily via cron at 6am UTC. Accuracy is marginal (~62%), so it uses high thresholds (0.65 for UP, 0.35 for DOWN) to stay NEUTRAL unless very confident. Treat as a tiebreaker, not a primary signal.

### Signal Agreement

The bot classifies every recommendation in plain English:

- **CONFIRMED** — Predictive signals agree with the market direction. Higher conviction trade.
- **MIXED** — Some signals agree, some disagree. Proceed with caution.
- **WARNING** — All predictive signals disagree with the recommendation. Consider skipping.
- **No strong signal** — All signals are NEUTRAL. The recommendation is based purely on market price.

---

## Quick Start

### Requirements

```bash
pip install requests numpy torch
```

PyTorch is optional (CPU-only is fine). Without it, LSTM is disabled but order flow and cross-asset still run.

### Telegram Bot Setup

1. Open Telegram → search for **@BotFather** → send `/newbot`
2. Name your bot, get your token
3. Open your new bot and send it any message (e.g., "hi")
4. Visit `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` in a browser
5. Find `"chat":{"id":123456789}` — that number is your chat ID

### Environment Variables

```bash
export TELEGRAM_BOT_TOKEN="your_token"
export TELEGRAM_CHAT_ID="your_chat_id"
```

### Run

```bash
# Single check
python3.10 recommend_only.py

# Continuous loop with Telegram notifications
python3.10 recommend_only.py --loop
```

---

## VPS Deployment (Oracle Cloud)

The bot is designed to run 24/7 as a systemd service on Oracle Cloud Free Tier (VM.Standard.E2.1.Micro, 1GB RAM). It auto-restarts on crash and starts on boot.

### Deploy

```bash
# Upload project to VPS
scp -i ~/.ssh/your-key -r ./PolymarketBTC_15Min_Trading ubuntu@YOUR_IP:~/

# SSH in
ssh -i ~/.ssh/your-key ubuntu@YOUR_IP

# Install dependencies
pip3 install requests numpy
pip3 install torch --index-url https://download.pytorch.org/whl/cpu  # optional, ~190MB

# Create .env for Telegram credentials
cat > ~/PolymarketBTC_15Min_Trading/.env << 'EOF'
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
EOF

# Create systemd service
sudo tee /etc/systemd/system/btc-signal.service << 'EOF'
[Unit]
Description=BTC 15M Signal Bot
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/PolymarketBTC_15Min_Trading
EnvironmentFile=/home/ubuntu/PolymarketBTC_15Min_Trading/.env
ExecStart=/usr/bin/python3.10 -u recommend_only.py --loop
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable btc-signal
sudo systemctl start btc-signal

# Add 1GB swap (recommended for 1GB RAM instances)
sudo fallocate -l 1G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Daily LSTM retrain cron (6am UTC)
(crontab -l 2>/dev/null; echo "0 6 * * * cd /home/ubuntu/PolymarketBTC_15Min_Trading && /usr/bin/python3.10 predictive_signals.py --train --hours=24") | crontab -
```

### Monitor

```bash
# Live logs
sudo journalctl -u btc-signal -f -o cat

# Status check
sudo systemctl status btc-signal

# Last 30 output lines
sudo journalctl -u btc-signal -n 30 --no-pager -o cat

# Memory usage
free -h
```

---

## Rate Limits

At 15-second check intervals (4 checks/minute), with signal refresh every 60 seconds:

| API | Calls/min | Known Limit | Headroom |
|---|---|---|---|
| Coinbase (spot price) | 4 | ~600/min | 99% |
| Coinbase (candles) | 5 | ~600/min | 99% |
| Polymarket Gamma | 4 | ~60/min | 93% |
| Polymarket CLOB | 16 | ~100/min | 84% |
| Kalshi External API | 4 | ~600/min | 99% |
| Telegram Bot API | 4 | ~20/min/chat | 80% |

Total: ~33 API calls/minute across all providers. No rate limit risk.

---

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `MIN_CONFIDENCE` | 0.60 | Minimum YES/NO price to trigger a recommendation |
| `CHECK_INTERVAL` | 15s | Time between checks in loop mode |
| Signal cache TTL | 60s | How often LSTM/OrderFlow/CrossAsset refresh |
| LSTM UP threshold | 0.65 | LSTM must exceed this to call UP (stays NEUTRAL otherwise) |
| LSTM DOWN threshold | 0.35 | LSTM must be below this to call DOWN |

---

## Project Structure

```
recommend_only.py              # Main signal bot — run this
predictive_signals.py          # LSTM, OrderFlow, CrossAsset signal processors
lstm_btc_model.pt              # Trained LSTM model weights
.env                           # Telegram credentials (not committed)
contract_recommendations.jsonl # Trade recommendation log

bot.py                         # Full NautilusTrader bot (requires Polymarket API keys)
15m_bot_runner.py              # Bot runner with test/live mode switching

core/                          # Core trading architecture
├── ingestion/                 # Data ingestion, adapters, rate limiting, validation
├── nautilus_core/             # NautilusTrader integration, data engine, instruments
└── strategy_brain/            # Signal processors, fusion engine, trading strategies

data_sources/                  # External market data connectors
├── binance/                   # Binance WebSocket client
├── coinbase/                  # Coinbase REST adapter
├── news_social/               # Sentiment / Fear & Greed
└── solana/                    # Solana RPC (experimental)

execution/                     # Order execution and risk management
├── execution_engine.py        # Order coordinator
├── polymarket_client.py       # Polymarket API wrapper
└── risk_engine.py             # Position sizing, stop loss, take profit

monitoring/                    # Performance tracking
├── grafana_exporter.py        # Prometheus metrics
└── performance_tracker.py     # Trade statistics

feedback/                      # Self-learning (experimental)
└── learning_engine.py         # Signal weight optimization

grafana/                       # Dashboard config
├── dashboard.json
└── import_dashboard.py
```

### Two Paths

**`recommend_only.py`** (what you run) — Standalone signal bot. No API keys needed. Pulls public data from Polymarket CLOB, Kalshi, and Coinbase. Sends recommendations to Telegram. This is the production workflow.

**`bot.py`** (advanced) — Full automated trading bot built on NautilusTrader. Requires Polymarket API credentials (US users need KYC with SSN). Includes execution engine, risk management, Redis mode switching, and Grafana monitoring. Not required for the recommendation workflow.

---

## FAQ

**Q: Do I need Polymarket API keys?**
A: No. `recommend_only.py` uses only public endpoints (Gamma API, CLOB midpoint/price, Kalshi external API). No authentication required for any data source.

**Q: What do I actually trade on?**
A: Robinhood. The 15-minute BTC event contracts on Robinhood are routed through KalshiEX. The bot shows you both the Polymarket price (for direction signal) and the Kalshi/Robinhood price (what you'll actually pay).

**Q: Can I run this 24/7?**
A: Yes. The bot is designed for continuous operation as a systemd service on a VPS. It auto-restarts on crash, starts on boot, and handles API errors gracefully.

**Q: How much money do I need?**
A: Robinhood event contracts start at a few cents per contract. You can start with $10-20 to test.

**Q: Is the LSTM actually useful?**
A: Marginally. It runs at ~62% accuracy and is often wrong on short-term direction. Order flow is the real alpha — it correctly identifies selling/buying pressure in real time. The LSTM uses high thresholds (0.65/0.35) so it mostly stays NEUTRAL and only speaks when confident.

**Q: Will this work outside the US?**
A: The signal bot works anywhere (public APIs). Trading on Robinhood requires a US account. Polymarket trading requires non-US or VPN.

---

## Disclaimer

This is a recommendation tool, not financial advice. Event contracts carry risk — you can lose your entire investment on any single trade. Past signal accuracy does not guarantee future results. The developers are not responsible for any financial losses. Start small, track your results, and trade at your own risk.

---

## Contact

GitHub: [BJW333](https://github.com/BJW333)
