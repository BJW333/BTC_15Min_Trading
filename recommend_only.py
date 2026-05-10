import time, json, requests, os, sys
from datetime import datetime, timezone
from predictive_signals import LSTMPredictor, OrderFlowSignal, CrossAssetMomentum

# =============================================================================
# CONFIGURATION
# =============================================================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
CHECK_INTERVAL = 30
MIN_CONFIDENCE = 0.60

GAMMA = "https://gamma-api.polymarket.com/markets"
CLOB = "https://clob.polymarket.com"
KALSHI_API = "https://external-api.kalshi.com/trade-api/v2"

# Multiple BTC price sources
PRICE_SOURCES = [
    ("Coinbase", "https://api.coinbase.com/v2/prices/BTC-USD/spot",
     lambda r: float(r.json()["data"]["amount"])),
    ("CoinGecko", "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd",
     lambda r: float(r.json()["bitcoin"]["usd"])),
    ("Kraken", "https://api.kraken.com/0/public/Ticker?pair=XBTUSD",
     lambda r: float(r.json()["result"]["XXBTZUSD"]["c"][0])),
]

_signal_cache = {"time": 0, "lstm_prob": None, "flow": None, "cross": None}
_last_notification = {"slug": None, "pick": None}

# =============================================================================
# HELPERS
# =============================================================================

def btc_price():
    for name, url, parse in PRICE_SOURCES:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return parse(resp)
        except Exception:
            continue
    raise RuntimeError("All BTC price sources failed")


def current_slug():
    now = int(time.time())
    interval = (now // 900) * 900
    return f"btc-updown-15m-{interval}"


def get_market(slug):
    """Get market metadata + token IDs from Gamma API."""
    r = requests.get(GAMMA, params={"slug": slug}, timeout=10)
    data = r.json()
    if isinstance(data, list) and data:
        return data[0]
    return None


def clob_midpoint(token_id):
    """Get REAL-TIME midpoint price from CLOB. No auth needed."""
    try:
        resp = requests.get(f"{CLOB}/midpoint", params={"token_id": token_id}, timeout=10)
        resp.raise_for_status()
        return float(resp.json().get("mid", 0))
    except Exception:
        return None


def clob_buy_price(token_id):
    """Get REAL-TIME buy price from CLOB. No auth needed."""
    try:
        resp = requests.get(f"{CLOB}/price", params={"token_id": token_id, "side": "buy"}, timeout=10)
        resp.raise_for_status()
        return float(resp.json().get("price", 0))
    except Exception:
        return None

def find_kalshi_btc_market():
    """Find the current active BTC 15-min contract on Kalshi. No auth needed."""
    try:
        resp = requests.get(f"{KALSHI_API}/markets", params={
            "series_ticker": "KXBTC15M",
            "status": "open",
            "limit": 5,
        }, timeout=10)
        resp.raise_for_status()
        markets = resp.json().get("markets", [])

        if markets:
            markets.sort(key=lambda m: m.get("close_time", ""))
            return markets[0]
        return None
    except Exception as e:
        print(f"  (Kalshi lookup failed: {e})")
        return None

def get_kalshi_prices(market):
    """Extract YES/NO prices from a Kalshi market object."""
    if not market:
        return None, None, None
    try:
        yes_ask = float(market.get("yes_ask_dollars", 0))
        no_ask = float(market.get("no_ask_dollars", 0))
        yes_bid = float(market.get("yes_bid_dollars", 0))
        no_bid = float(market.get("no_bid_dollars", 0))

        # Use midpoint of bid/ask for decision price
        yes_mid = (yes_bid + yes_ask) / 2 if (yes_bid and yes_ask) else yes_ask
        no_mid = (no_bid + no_ask) / 2 if (no_bid and no_ask) else no_ask

        # Get the target/strike price
        strike = market.get("floor_strike")
        subtitle = market.get("yes_sub_title", "")

        return yes_mid, no_mid, {
            "yes_bid": yes_bid, "yes_ask": yes_ask,
            "no_bid": no_bid, "no_ask": no_ask,
            "strike": strike, "subtitle": subtitle,
            "ticker": market.get("ticker", ""),
        }
    except Exception:
        return None, None, None
    
def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        resp = requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
        }, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"  (Telegram send failed: {e})")
        return False


# =============================================================================
# MAIN
# =============================================================================

def check_once():
    """Run one recommendation check. Returns True if a recommendation was made."""
    slug = current_slug()
    
    # Only check during the profitable window (minutes 5-12 of each interval)
    now_ts = int(time.time())
    interval_start = (now_ts // 900) * 900
    minutes_in = (now_ts - interval_start) / 60
    

    price = btc_price()
    market = get_market(slug)

    now_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    print("=" * 70)
    print(f"BTC 15M RECOMMEND-ONLY  |  {now_str}")
    print("=" * 70)
    print(f"BTC: ${price:,.2f}")
    print(f"Slug: {slug}")

    if not market:
        print("No active Polymarket market found for this interval.")
        return False

    question = market.get("question", "")
    gamma_prices = market.get("outcomePrices")
    clob_token_ids = market.get("clobTokenIds")

    if isinstance(gamma_prices, str):
        gamma_prices = json.loads(gamma_prices)
    if isinstance(clob_token_ids, str):
        clob_token_ids = json.loads(clob_token_ids)

    print(f"Market: {question}")

    gamma_yes = float(gamma_prices[0])
    gamma_no = float(gamma_prices[1])

    # -----------------------------------------------------------------
    # CLOB /midpoint and /price — the REAL live data
    # -----------------------------------------------------------------
    yes_mid = None
    no_mid = None
    yes_buy = None
    no_buy = None

    if clob_token_ids and len(clob_token_ids) >= 2:
        yes_mid = clob_midpoint(clob_token_ids[0])
        no_mid = clob_midpoint(clob_token_ids[1])
        yes_buy = clob_buy_price(clob_token_ids[0])
        no_buy = clob_buy_price(clob_token_ids[1])

    # Use CLOB if available, Gamma as fallback
    if yes_mid is not None and yes_mid > 0:
        yes_price = yes_mid
        no_price = no_mid if no_mid else 1.0 - yes_mid
        data_source = "CLOB live"
    else:
        yes_price = gamma_yes
        no_price = gamma_no
        data_source = "Gamma (cached)"

    # Print comparison
    print(f"\n  Gamma (cached):  YES={gamma_yes:.3f}  NO={gamma_no:.3f}")
    if yes_mid is not None:
        print(f"  CLOB  (live):    YES={yes_mid:.3f}  NO={no_mid:.3f}")
        if yes_buy is not None:
            print(f"  CLOB buy price:  YES={yes_buy:.2f}  NO={no_buy:.2f}")
        drift = abs(yes_mid - gamma_yes)
        if drift > 0.01:
            print(f"  ** Gamma is stale by {drift:.1%} **")
    print(f"  >>> Using: {data_source}")

    # --- Kalshi prices (what you see on Robinhood) ---
    kalshi_market = find_kalshi_btc_market()
    kalshi_yes, kalshi_no, kalshi_info = get_kalshi_prices(kalshi_market)

    if kalshi_yes is not None and kalshi_info:
        strike = kalshi_info.get("strike", "?")
        print(f"\n  Kalshi/RH:       YES={kalshi_yes:.3f}  NO={kalshi_no:.3f}")
        print(f"  Kalshi bid/ask:  YES: {kalshi_info['yes_bid']:.2f}/{kalshi_info['yes_ask']:.2f}  NO: {kalshi_info['no_bid']:.2f}/{kalshi_info['no_ask']:.2f}")
        print(f"  Target price:    ${strike:,.2f}")
    else:
        print(f"\n  Kalshi/RH:       not available")

    # --- Predictive signals (refresh every 60s to avoid rate limits) ---
    now_ts = time.time()
    if now_ts - _signal_cache["time"] > 60:
        lstm = LSTMPredictor()
        _signal_cache["lstm_prob"] = lstm.predict()
        _signal_cache["flow"] = OrderFlowSignal().get_signal()
        _signal_cache["cross"] = CrossAssetMomentum().get_signal()
        _signal_cache["time"] = now_ts

    lstm_prob = _signal_cache["lstm_prob"]
    flow = _signal_cache["flow"]
    cross = _signal_cache["cross"]
    
    lstm_dir = "NEUTRAL"
    if lstm_prob is not None:
        if lstm_prob > 0.65:
            lstm_dir = "UP"
        elif lstm_prob < 0.35:
            lstm_dir = "DOWN"
        print(f"\n  LSTM:         {lstm_dir} ({lstm_prob:.0%} chance up)")
    else:
        print(f"\n  LSTM:         not available")

    print(f"  OrderFlow:    {flow['direction']} (strength={flow['strength']:.2f})")
    print(f"  CrossAsset:   {cross['direction']} (strength={cross['strength']:.2f})")
    
    # -----------------------------------------------------------------
    # Decision
    # -----------------------------------------------------------------
    if yes_price > MIN_CONFIDENCE:
        pick = "YES"
        limit = round(yes_buy if yes_buy else min(yes_price + 0.01, 0.99), 2)
        confidence = yes_price
    elif no_price > MIN_CONFIDENCE:
        pick = "NO"
        limit = round(no_buy if no_buy else min(no_price + 0.01, 0.99), 2)
        confidence = no_price
    else:
        print(f"\n  SKIP -- too close to 50/50 (YES={yes_price:.1%} / NO={no_price:.1%})")
        return False

    rec = {
        "time": datetime.now(timezone.utc).isoformat(),
        "slug": slug,
        "btc_price": price,
        "market_question": question,
        "recommendation": f"BUY LIMIT {pick}",
        "limit_price": limit,
        "confidence": confidence,
        "data_source": data_source,
        "clob_yes_mid": yes_mid, "clob_no_mid": no_mid,
        "clob_yes_buy": yes_buy, "clob_no_buy": no_buy,
        "gamma_yes": gamma_yes, "gamma_no": gamma_no,
    }

    print()
    print("=" * 70)
    print(f"  RECOMMENDATION  ({data_source})")
    print("=" * 70)
    print(f"  >>> BUY the {pick} contract on: {question}")
    print(f"  Limit price: ${limit}  (Polymarket)")
    if kalshi_yes is not None and kalshi_info:
        k_price = kalshi_info['no_ask'] if pick == "NO" else kalshi_info['yes_ask']
        print(f"  Robinhood:   ${k_price:.2f}  (Kalshi ask)")
    print(f"  Confidence: {confidence:.0%}")
    print("=" * 70)
    # Count how many signals agree with the CLOB direction
    # Signal agreement — plain English
    clob_dir = "UP" if pick == "YES" else "DOWN"
    signals = {
        "LSTM": lstm_dir,
        "OrderFlow": flow["direction"],
        "CrossAsset": cross["direction"],
    }
    agreeing = [name for name, d in signals.items() if d == clob_dir]
    disagreeing = [name for name, d in signals.items() if d != "NEUTRAL" and d != clob_dir]

    if disagreeing and not agreeing:
        print(f"\n  WARNING: {' + '.join(disagreeing)} say {('UP' if clob_dir == 'DOWN' else 'DOWN')} — DISAGREES with recommendation")
        print(f"  Consider SKIPPING this trade")
    elif agreeing and not disagreeing:
        print(f"\n  CONFIRMED: {' + '.join(agreeing)} agree → {clob_dir}")
    elif agreeing and disagreeing:
        print(f"\n  MIXED: {' + '.join(agreeing)} say {clob_dir}, but {' + '.join(disagreeing)} disagree")
    else:
        print(f"\n  No strong signal from predictive models")
        
    with open("contract_recommendations.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")

    # Telegram
    new_interval = (slug != _last_notification.get("slug"))
    direction_changed = (pick != _last_notification.get("pick"))

    if True:  # send every cycle
        # Build the same output as terminal
        lines = []
        lines.append(f"BTC 15M | {now_str}")
        lines.append(f"BTC: ${price:,.2f}")
        lines.append(f"Market: {question}")
        lines.append("")
        lines.append(f"CLOB (live): YES={yes_mid:.3f}  NO={no_mid:.3f}" if yes_mid else f"Gamma: YES={gamma_yes:.3f}  NO={gamma_no:.3f}")
        if yes_buy is not None:
            lines.append(f"CLOB buy:    YES={yes_buy:.2f}  NO={no_buy:.2f}")
        if kalshi_yes is not None and kalshi_info:
            lines.append(f"Kalshi/RH:   YES={kalshi_yes:.3f}  NO={kalshi_no:.3f}")
            lines.append(f"Kalshi ask:  YES: {kalshi_info['yes_ask']:.2f}  NO: {kalshi_info['no_ask']:.2f}")
            lines.append(f"Target:      ${kalshi_info['strike']:,.2f}")
        lines.append("")
        if lstm_prob is not None:
            lines.append(f"LSTM:        {lstm_dir} ({lstm_prob:.0%} chance up)")
        lines.append(f"OrderFlow:   {flow['direction']} (str={flow['strength']:.2f})")
        lines.append(f"CrossAsset:  {cross['direction']} (str={cross['strength']:.2f})")
        lines.append("")
        lines.append(f">>> BUY {pick}")
        lines.append(f"Polymarket:  ${limit}")
        if kalshi_yes is not None and kalshi_info:
            k = kalshi_info['no_ask'] if pick == 'NO' else kalshi_info['yes_ask']
            lines.append(f"Robinhood:   ${k:.2f}")
        lines.append(f"Confidence:  {confidence:.0%}")
        lines.append("")
        if disagreeing and not agreeing:
            lines.append(f"WARNING: {' + '.join(disagreeing)} DISAGREE -- consider skipping")
        elif agreeing and not disagreeing:
            lines.append(f"CONFIRMED: {' + '.join(agreeing)} agree")
        elif agreeing and disagreeing:
            lines.append(f"MIXED: {' + '.join(agreeing)} agree, {' + '.join(disagreeing)} disagree")
        else:
            lines.append("No strong signal from predictive models")

        telegram_msg = "\n".join(lines)
        sent = send_telegram(telegram_msg)
        if sent:
            print(f"  Telegram sent ({'New interval' if new_interval else 'Direction flipped'})")
        _last_notification["slug"] = slug
        _last_notification["pick"] = pick

    return True


def main():
    loop_mode = "--loop" in sys.argv

    if loop_mode:
        tg_status = "ON" if (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID) else "OFF"
        print(f"Loop mode (10s check, 60s signals | Telegram: {tg_status})")
        print("Ctrl+C to stop.\n")
        try:
            while True:
                check_once()
                print()
                time.sleep(15)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        check_once()
        
if __name__ == "__main__":
    main()