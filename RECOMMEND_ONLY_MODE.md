# Recommend-Only Mode

This patch adds a safe mode that tells you which Polymarket BTC 15-minute limit contract to buy without submitting any real or paper trade.

## Run recommendation-only mode

```bash
python3 15m_bot_runner.py --recommend-only
```

or directly:

```bash
python3 bot.py --recommend-only
```

## What it does

When the strategy reaches a valid trade window, it prints the **market question** and which side to buy:

```
================================================================================
  RECOMMENDATION
================================================================================
  >>> BUY the YES contract on: Will the price of BTC be above $103,500 at 7:30 PM ET?
  Limit price: $0.7200
  Current bid/ask: $0.7100 / $0.7200
  Confidence: 72%
  Signal: score=65.3  confidence=78%
  Size: $1.00
================================================================================
```

Recommendations are also appended to:

```text
contract_recommendations.jsonl
```

Each JSONL record includes the `market_question` and `market_slug` so you can look it up on Polymarket.

## Standalone lightweight version

If you don't want to run the full Nautilus stack, there's a standalone script
that hits the Gamma API + CoinGecko directly:

```bash
python3 recommend_only.py
```

## Safety behavior

`--recommend-only` overrides `--live`, so this will not place real orders even if you accidentally pass both:

```bash
python3 bot.py --live --recommend-only
```

## Switch modes

Recommendation only:

```bash
python3 bot.py --recommend-only
```

Paper/simulation mode:

```bash
python3 bot.py
```

Live trading mode:

```bash
python3 bot.py --live
```
