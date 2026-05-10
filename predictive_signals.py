"""
predictive_signals.py — Predictive signal processors for BTC 15-min Polymarket trading.

All data from Coinbase (works in US, no auth needed).

Signals:
  1. LSTM on 1-min BTC candles  → predicts UP/DOWN for next 15 min
  2. Order flow analysis         → buy/sell pressure, CVD, VWAP, OBV, volume spikes
  3. Cross-asset momentum        → ETH/SOL leading-indicator for BTC
  4. Volatility forecast          → should you trade this interval at all?

Usage:
  python3.10 predictive_signals.py              # run all signals
  python3.10 predictive_signals.py --train      # train LSTM on 24h
  python3.10 predictive_signals.py --train --hours=72
  python3.10 predictive_signals.py --predict    # LSTM only
  python3.10 predictive_signals.py --flow       # order flow only

Install:
  pip install torch numpy requests
"""

import os, sys, json, time, math
import numpy as np
import requests
from datetime import datetime, timezone
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not installed. Run: pip install torch")
    print("LSTM disabled. Other signals still work.\n")


# =============================================================================
# DATA FETCHING — Coinbase only (works in US, no auth, reliable)
# =============================================================================

COINBASE_CANDLES = "https://api.exchange.coinbase.com/products/{}/candles"


def fetch_candles(pair="BTC-USD", granularity=60, count=300):
    """Fetch 1-min candles from Coinbase. Sorted oldest -> newest."""
    url = COINBASE_CANDLES.format(pair)
    resp = requests.get(url, params={"granularity": granularity}, timeout=15)
    resp.raise_for_status()
    raw = resp.json()

    candles = []
    for row in raw[:count]:
        candles.append({
            "time": row[0],
            "low": float(row[1]),
            "high": float(row[2]),
            "open": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]),
        })

    candles.sort(key=lambda c: c["time"])
    return candles


def estimate_buy_sell_volume(candles):
    """
    Estimate taker buy/sell volume from candle structure.

    Uses the Money Flow Multiplier: where the close sits within the
    high-low range tells you who dominated that candle.

      close near high → buyers dominated → most volume was buying
      close near low  → sellers dominated → most volume was selling

    Formula: buy_ratio = (close - low) / (high - low)

    This is the same math behind Chaikin Money Flow, Accumulation/Distribution,
    and other institutional-grade volume indicators. It's surprisingly accurate
    on 1-minute candles because each candle captures a short enough window
    that the close position genuinely reflects order flow direction.
    """
    for c in candles:
        hl_range = c["high"] - c["low"]
        if hl_range > 0:
            # Money Flow Multiplier: 0 = close at low, 1 = close at high
            buy_ratio = (c["close"] - c["low"]) / hl_range
        else:
            # Doji candle (open == close == high == low): neutral
            buy_ratio = 0.5

        c["buy_ratio"] = buy_ratio
        c["taker_buy_volume"] = c["volume"] * buy_ratio
        c["taker_sell_volume"] = c["volume"] * (1.0 - buy_ratio)

    return candles


# =============================================================================
# FEATURE ENGINEERING (13 features: 8 price + 5 order flow)
# =============================================================================

def compute_features(candles):
    """
    Returns numpy array (N, 13).

    Price (0-7):
      0: log return              5: Bollinger Band position
      1: high-low range          6: 5-min momentum
      2: volume change           7: 15-min momentum
      3: RSI (14)
      4: MACD histogram

    Order flow (8-12):
      8:  buy/sell ratio (money flow multiplier, centered at 0)
      9:  CVD rate (15-min cumulative volume delta, normalized)
      10: OBV slope (on-balance volume trend)
      11: VWAP deviation (price vs VWAP, %)
      12: volume spike (current vol / 20-min avg)
    """
    # Ensure buy/sell estimates exist
    if "buy_ratio" not in candles[0]:
        estimate_buy_sell_volume(candles)

    closes = np.array([c["close"] for c in candles])
    highs = np.array([c["high"] for c in candles])
    lows = np.array([c["low"] for c in candles])
    volumes = np.array([c["volume"] for c in candles])
    buy_vols = np.array([c["taker_buy_volume"] for c in candles])
    sell_vols = np.array([c["taker_sell_volume"] for c in candles])
    n = len(closes)

    features = np.zeros((n, 13))

    # --- Price features (0-7) ---

    features[1:, 0] = np.diff(np.log(closes))

    features[:, 1] = (highs - lows) / closes

    vol_mean = volumes.mean() or 1.0
    features[:, 2] = (volumes - vol_mean) / vol_mean

    # RSI (14)
    period = 14
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    if len(gains) >= period:
        avg_gain[period] = gains[:period].mean()
        avg_loss[period] = losses[:period].mean()
        for i in range(period + 1, n):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 50.0)
    rsi = 100 - (100 / (1 + rs))
    features[:, 3] = (rsi - 50) / 50

    # MACD
    def ema(data, span):
        alpha = 2 / (span + 1)
        result = np.zeros_like(data)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result

    ema12 = ema(closes, 12)
    ema26 = ema(closes, 26)
    macd_hist = (ema12 - ema26) - ema(ema12 - ema26, 9)
    features[:, 4] = macd_hist / closes * 1000

    # Bollinger Band position
    for i in range(20, n):
        w = closes[i-20:i]
        mid = w.mean()
        std = w.std() or 1e-8
        features[i, 5] = (closes[i] - mid) / (2 * std)

    # Momentum
    for i in range(5, n):
        features[i, 6] = (closes[i] - closes[i-5]) / closes[i-5] * 100
    for i in range(15, n):
        features[i, 7] = (closes[i] - closes[i-15]) / closes[i-15] * 100

    # --- Order flow features (8-12) ---

    # 8: Buy/sell ratio (centered at 0)
    total_vols = buy_vols + sell_vols
    safe_total = np.where(total_vols > 0, total_vols, 1.0)
    features[:, 8] = (buy_vols / safe_total) - 0.5

    # 9: CVD rate (rolling 15-min window)
    volume_delta = buy_vols - sell_vols
    cvd = np.cumsum(volume_delta)
    for i in range(15, n):
        window_cvd = cvd[i] - cvd[i-15]
        avg_vol = total_vols[i-15:i].mean() or 1.0
        features[i, 9] = window_cvd / (avg_vol * 15) * 10

    # 10: OBV slope (15-min linear regression)
    obv = np.zeros(n)
    for i in range(1, n):
        if closes[i] > closes[i-1]:
            obv[i] = obv[i-1] + volumes[i]
        elif closes[i] < closes[i-1]:
            obv[i] = obv[i-1] - volumes[i]
        else:
            obv[i] = obv[i-1]
    for i in range(15, n):
        obv_w = obv[i-15:i+1]
        x = np.arange(len(obv_w))
        slope = np.polyfit(x, obv_w, 1)[0]
        avg_vol = volumes[i-15:i+1].mean() or 1.0
        features[i, 10] = slope / avg_vol

    # 11: VWAP deviation
    cum_vp = np.cumsum(closes * volumes)
    cum_v = np.cumsum(volumes)
    for i in range(15, n):
        pvp = cum_vp[i] - cum_vp[i-15]
        pv = cum_v[i] - cum_v[i-15]
        if pv > 0:
            vwap = pvp / pv
            features[i, 11] = (closes[i] - vwap) / vwap * 100

    # 12: Volume spike
    for i in range(20, n):
        avg20 = volumes[i-20:i].mean() or 1.0
        features[i, 12] = (volumes[i] / avg20) - 1.0

    return features


def make_sequences(features, labels=None, seq_len=60):
    X = []
    y = []
    for i in range(seq_len, len(features)):
        X.append(features[i - seq_len:i])
        if labels is not None:
            y.append(labels[i])
    X = np.array(X)
    if labels is not None:
        return X, np.array(y)
    return X


# =============================================================================
# LSTM MODEL
# =============================================================================

if TORCH_AVAILABLE:
    class BTCDirectionLSTM(nn.Module):
        def __init__(self, input_size=13, hidden_size=64, num_layers=2, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size,
                num_layers=num_layers, batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(32, 1), nn.Sigmoid(),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)


# =============================================================================
# LSTM PREDICTOR
# =============================================================================

MODEL_PATH = Path(__file__).parent / "lstm_btc_model.pt"
SEQ_LEN = 60


class LSTMPredictor:
    def __init__(self):
        if not TORCH_AVAILABLE:
            self.model = None
            return
        self.model = BTCDirectionLSTM(input_size=13, hidden_size=64, num_layers=2)
        if MODEL_PATH.exists():
            self.model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
            self.model.eval()
            print(f"  Loaded LSTM from {MODEL_PATH}")
        else:
            print(f"  No trained model found. Run with --train first.")

    def train_on_history(self, hours=24, epochs=30, lr=0.001):
        if not TORCH_AVAILABLE:
            print("PyTorch required.")
            return

        print(f"\nFetching {hours}h of 1-min BTC candles...")
        all_candles = []
        end_time = int(time.time())
        target = hours * 60

        while len(all_candles) < target:
            start_time = end_time - 300 * 60
            url = COINBASE_CANDLES.format("BTC-USD")
            resp = requests.get(url, params={
                "granularity": 60,
                "start": datetime.fromtimestamp(start_time, tz=timezone.utc).isoformat(),
                "end": datetime.fromtimestamp(end_time, tz=timezone.utc).isoformat(),
            }, timeout=15)
            resp.raise_for_status()
            raw = resp.json()
            for r in raw:
                all_candles.append({
                    "time": r[0], "low": float(r[1]), "high": float(r[2]),
                    "open": float(r[3]), "close": float(r[4]), "volume": float(r[5]),
                })
            end_time = start_time
            time.sleep(0.3)

        all_candles.sort(key=lambda c: c["time"])
        seen = set()
        candles = [c for c in all_candles if not (c["time"] in seen or seen.add(c["time"]))]
        print(f"  {len(candles)} candles ({len(candles)/60:.1f}h)")

        # Estimate buy/sell from candle structure
        estimate_buy_sell_volume(candles)
        features = compute_features(candles)
        closes = np.array([c["close"] for c in candles])

        labels = np.zeros(len(candles))
        for i in range(len(candles) - 15):
            labels[i] = 1.0 if closes[i + 15] > closes[i] else 0.0

        features = features[:-15]
        labels = labels[:-15]

        X, y = make_sequences(features, labels, seq_len=SEQ_LEN)
        print(f"  Samples: {len(X)}  UP ratio: {y.mean():.1%}")

        split = int(len(X) * 0.8)
        X_train = torch.FloatTensor(X[:split])
        y_train = torch.FloatTensor(y[:split])
        X_val = torch.FloatTensor(X[split:])
        y_val = torch.FloatTensor(y[split:])

        model = BTCDirectionLSTM(input_size=13, hidden_size=64, num_layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        print(f"\n  Training {epochs} epochs...")
        best_acc = 0
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(X_train), y_train)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                vp = model(X_val)
                vl = criterion(vp, y_val)
                va = ((vp > 0.5).float() == y_val).float().mean()

            if va > best_acc:
                best_acc = va
                torch.save(model.state_dict(), MODEL_PATH)

            if (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1:>3}/{epochs}  loss={loss:.4f}  val_loss={vl:.4f}  val_acc={va:.1%}")

        print(f"\n  Best val accuracy: {best_acc:.1%}")
        print(f"  Saved to {MODEL_PATH}")
        self.model = model
        self.model.eval()

    def predict(self):
        if not TORCH_AVAILABLE or self.model is None:
            return None
        try:
            candles = fetch_candles("BTC-USD", granularity=60, count=120)
            estimate_buy_sell_volume(candles)
            features = compute_features(candles)
            seq = features[-SEQ_LEN:]
            X = torch.FloatTensor(seq).unsqueeze(0)

            self.model.eval()
            with torch.no_grad():
                return self.model(X).item()
        except Exception as e:
            print(f"  LSTM error: {e}")
            return None


# =============================================================================
# ORDER FLOW SIGNAL (standalone, no LSTM, no Binance)
# =============================================================================

class OrderFlowSignal:
    """
    Buy/sell pressure analysis from Coinbase candle structure.

    Uses Money Flow Multiplier to estimate taker buy/sell volume,
    then computes CVD, VWAP deviation, OBV, and volume spikes.
    """

    def get_signal(self):
        try:
            candles = fetch_candles("BTC-USD", granularity=60, count=60)
            if len(candles) < 20:
                return {"direction": "NEUTRAL", "strength": 0.0, "detail": "insufficient data"}

            estimate_buy_sell_volume(candles)

            closes = np.array([c["close"] for c in candles])
            volumes = np.array([c["volume"] for c in candles])
            buy_vols = np.array([c["taker_buy_volume"] for c in candles])
            sell_vols = np.array([c["taker_sell_volume"] for c in candles])

            # Buy/sell ratio (last 5 min)
            rb = buy_vols[-5:].sum()
            rs = sell_vols[-5:].sum()
            rt = rb + rs
            buy_ratio = rb / rt if rt > 0 else 0.5

            # CVD (last 15 min)
            delta_15 = (buy_vols[-15:] - sell_vols[-15:]).sum()
            avg_vol = volumes[-15:].mean() or 1.0
            cvd_norm = delta_15 / (avg_vol * 15)

            # VWAP deviation
            cv = np.cumsum(closes[-15:] * volumes[-15:])
            v = np.cumsum(volumes[-15:])
            vwap = cv[-1] / v[-1] if v[-1] > 0 else closes[-1]
            vwap_dev = (closes[-1] - vwap) / vwap * 100

            # OBV slope (last 15 min)
            obv = [0.0]
            for i in range(1, min(15, len(closes))):
                idx = len(closes) - 15 + i
                if idx < 1:
                    obv.append(obv[-1])
                    continue
                if closes[idx] > closes[idx-1]:
                    obv.append(obv[-1] + volumes[idx])
                elif closes[idx] < closes[idx-1]:
                    obv.append(obv[-1] - volumes[idx])
                else:
                    obv.append(obv[-1])
            obv_arr = np.array(obv)
            obv_slope = np.polyfit(np.arange(len(obv_arr)), obv_arr, 1)[0]
            obv_norm = obv_slope / (avg_vol or 1.0)

            # Volume spike
            avg20 = volumes[-20:].mean() or 1.0
            vol_spike = volumes[-1] / avg20

            # Weighted score
            score = 0.0
            score += (buy_ratio - 0.5) * 4.0
            score += cvd_norm * 2.0
            score += vwap_dev * 0.5
            score += obv_norm * 1.0
            if vol_spike > 2.0:
                score *= 1.5

            if score > 0.1:
                direction = "UP"
            elif score < -0.1:
                direction = "DOWN"
            else:
                direction = "NEUTRAL"
            strength = min(abs(score), 1.0)

            detail = (
                f"buy_ratio={buy_ratio:.1%}  cvd={cvd_norm:+.3f}  "
                f"vwap={vwap_dev:+.3f}%  obv={obv_norm:+.3f}  "
                f"vol_spike={vol_spike:.1f}x"
            )

            return {
                "direction": direction,
                "strength": strength,
                "score": round(score, 4),
                "detail": detail,
                "components": {
                    "buy_sell_ratio": round(buy_ratio, 4),
                    "cvd_15m": round(cvd_norm, 4),
                    "vwap_deviation": round(vwap_dev, 4),
                    "obv_slope": round(obv_norm, 4),
                    "volume_spike": round(vol_spike, 2),
                },
            }
        except Exception as e:
            return {"direction": "NEUTRAL", "strength": 0.0, "detail": f"error: {e}"}


# =============================================================================
# CROSS-ASSET MOMENTUM
# =============================================================================

class CrossAssetMomentum:
    """ETH/SOL leading indicator for BTC direction."""

    def get_signal(self):
        try:
            changes = {}
            for pair in ["BTC-USD", "ETH-USD", "SOL-USD"]:
                candles = fetch_candles(pair, granularity=60, count=10)
                if len(candles) < 5:
                    continue
                c_now = candles[-1]["close"]
                c_5m = candles[-5]["close"]
                changes[pair] = (c_now - c_5m) / c_5m * 100

            btc = changes.get("BTC-USD", 0)
            eth = changes.get("ETH-USD", 0)
            sol = changes.get("SOL-USD", 0)
            div = ((eth + sol) / 2) - btc

            detail = f"BTC={btc:+.2f}%  ETH={eth:+.2f}%  SOL={sol:+.2f}%  div={div:+.2f}%"

            if abs(div) < 0.1:
                return {"direction": "NEUTRAL", "strength": 0.0, "detail": detail}
            d = "UP" if div > 0 else "DOWN"
            return {"direction": d, "strength": min(abs(div) / 0.5, 1.0), "detail": detail}
        except Exception as e:
            return {"direction": "NEUTRAL", "strength": 0.0, "detail": f"error: {e}"}


# =============================================================================
# VOLATILITY FORECAST
# =============================================================================

class VolatilityForecast:
    def get_signal(self):
        try:
            candles = fetch_candles("BTC-USD", granularity=60, count=120)
            closes = np.array([c["close"] for c in candles])
            rets = np.diff(np.log(closes))

            recent = rets[-15:].std() * math.sqrt(15)
            hist = rets.std() * math.sqrt(15)
            ratio = recent / hist if hist > 0 else 1.0
            move = (closes[-1] - closes[-15]) / closes[-15] * 100
            trending = abs(move) > 0.05
            tradeable = ratio > 0.8 and trending

            detail = f"vol_ratio={ratio:.2f}  15m_move={move:+.3f}%  trending={trending}"
            return {"tradeable": tradeable, "volatility": recent, "vol_ratio": ratio,
                    "trending": trending, "detail": detail}
        except Exception as e:
            return {"tradeable": True, "volatility": 0, "vol_ratio": 1.0,
                    "trending": True, "detail": f"error: {e}"}


# =============================================================================
# COMBINED SIGNAL
# =============================================================================

def get_predictive_signal():
    """
    Run all predictive processors.

    Weights:
      Order Flow:  0.35  (buy/sell pressure — strongest short-term)
      LSTM:        0.30  (pattern recognition)
      Cross-Asset: 0.20  (ETH/SOL leading indicator)
      Volatility:  filter (should we trade at all?)
    """
    print("  Running predictive signals...")

    # 1. LSTM
    lstm = LSTMPredictor()
    lp = lstm.predict()
    ld, ls = "NEUTRAL", 0.0
    if lp is not None:
        if lp > 0.55:
            ld, ls = "UP", (lp - 0.5) * 2
        elif lp < 0.45:
            ld, ls = "DOWN", (0.5 - lp) * 2
        print(f"    LSTM:        P(up)={lp:.1%}  {ld} (str={ls:.2f})")
    else:
        print(f"    LSTM:        not available")

    # 2. Order flow
    flow = OrderFlowSignal()
    fs = flow.get_signal()
    print(f"    OrderFlow:   {fs['direction']} (str={fs['strength']:.2f})  {fs['detail']}")

    # 3. Cross-asset
    cross = CrossAssetMomentum()
    cs = cross.get_signal()
    print(f"    CrossAsset:  {cs['direction']} (str={cs['strength']:.2f})  {cs['detail']}")

    # 4. Volatility
    vol = VolatilityForecast()
    vs = vol.get_signal()
    print(f"    Volatility:  {'TRADEABLE' if vs['tradeable'] else 'SKIP'}  {vs['detail']}")

    # Weighted vote
    votes = {"UP": 0.0, "DOWN": 0.0}
    if ld != "NEUTRAL":
        votes[ld] += ls * 0.30
    if fs["direction"] != "NEUTRAL":
        votes[fs["direction"]] += fs["strength"] * 0.35
    if cs["direction"] != "NEUTRAL":
        votes[cs["direction"]] += cs["strength"] * 0.20

    if votes["UP"] > votes["DOWN"] and votes["UP"] > 0.05:
        direction, confidence = "UP", min(votes["UP"], 1.0)
    elif votes["DOWN"] > votes["UP"] and votes["DOWN"] > 0.05:
        direction, confidence = "DOWN", min(votes["DOWN"], 1.0)
    else:
        direction, confidence = "NEUTRAL", 0.0

    result = {
        "direction": direction,
        "confidence": confidence,
        "tradeable": vs["tradeable"],
        "components": {
            "lstm": {"prob": lp, "direction": ld, "strength": ls},
            "order_flow": fs,
            "cross_asset": cs,
            "volatility": vs,
        },
    }

    print(f"\n    COMBINED:    {direction}  confidence={confidence:.0%}  tradeable={vs['tradeable']}")
    return result


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    if "--train" in sys.argv:
        hours, epochs = 24, 30
        for a in sys.argv:
            if a.startswith("--hours="):
                hours = int(a.split("=")[1])
            if a.startswith("--epochs="):
                epochs = int(a.split("=")[1])
        LSTMPredictor().train_on_history(hours=hours, epochs=epochs)

    elif "--predict" in sys.argv:
        p = LSTMPredictor().predict()
        if p is not None:
            print(f"\nLSTM P(up): {p:.1%}  Signal: {'UP' if p > 0.55 else 'DOWN' if p < 0.45 else 'NEUTRAL'}")

    elif "--flow" in sys.argv:
        print("=" * 60)
        print("ORDER FLOW ANALYSIS — BTC")
        print("=" * 60)
        s = OrderFlowSignal().get_signal()
        print(f"\n  Direction: {s['direction']}")
        print(f"  Strength:  {s['strength']:.2f}")
        print(f"  Score:     {s.get('score', 'N/A')}")
        print(f"  {s['detail']}")
        if "components" in s:
            print(f"\n  Components:")
            for k, v in s["components"].items():
                print(f"    {k}: {v}")

    else:
        print("=" * 60)
        print("PREDICTIVE SIGNALS — BTC 15M")
        print("=" * 60)
        sig = get_predictive_signal()
        print()
        print(json.dumps(sig, indent=2, default=str))
