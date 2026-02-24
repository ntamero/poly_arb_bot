# PolyARB Trading Terminal

> Real-time Polymarket arbitrage & edge detection bot with multi-engine analysis, automated trading, and a live web dashboard.

![Version](https://img.shields.io/badge/version-1.5.0-green)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-yellow)
![Mode](https://img.shields.io/badge/mode-simulation-orange)

---

## Features

- **Universal Market Engine** — Scans ALL Polymarket markets using extreme-price and momentum sub-strategies
- **BTC Price Engine** — 5-exchange aggregation (Binance, OKX, Coinbase, Upbit, Bybit) with log-normal volatility probability model
- **Weather Engine** — NOAA real-time data vs Polymarket weather contracts
- **Metals Engine** — XAU/XAG technical analysis (gold & silver)
- **AutoTrader** — Automated scanning, bet placement, and position management with configurable intervals
- **Smart Exits** — TP (+20%) / SL (-30%) / Time-stop (7 day) sell logic with realistic PnL
- **Confidence Sizing** — Bet sizing scaled by confidence: high (≥0.7) = full, medium = 60%, low = 30%
- **Web Dashboard** — Real-time portfolio, positions, PnL chart, live BTC prices, trade history
- **Telegram Bot** — Trade notifications, daily reports, and remote command control
- **Simulation Mode** — Paper trading with real market data (no real money at risk)

---

## Quick Start

```bash
# Clone
git clone https://github.com/ntamero/poly_arb_bot.git
cd poly_arb_bot

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run
python src/web_server.py
```

Open **http://localhost:8080** for the live trading dashboard.

---

## Dashboard

The web dashboard provides a full trading terminal experience:

| Section | Description |
|---------|-------------|
| **Top Bar** | Balance, Daily P&L, Open Positions, Live BTC price, Win Rate |
| **Portfolio Sidebar** | Balance breakdown (Realized / Unrealized / Invested) |
| **Win Rate Ring** | Visual donut chart with W/L/Open counts |
| **BTC Live Price** | 5-exchange average with sparkline chart and per-exchange spread |
| **Auto Trader Controls** | Start/Stop, Scan Now, interval & edge config |
| **Daily Strip** | Start balance, Total P&L, Wins, Losses, Invested |
| **PnL Chart** | Time-series portfolio value with 1H/6H/24H/7D periods |
| **Opportunities** | Live-updating opportunity table with edge, confidence, price |
| **Trade History** | All closed trades with PnL and exit reasons |
| **Live Log** | Real-time cycle logs, scan results, and trade events |

---

## Architecture

```
polymarket-arb-bot/
├── src/
│   ├── bot.py              # Core engines + AutoTrader + SimulationEngine
│   │   ├── UniversalMarketEngine  # Scans ALL Polymarket markets
│   │   │   ├── Extreme Price      # YES ≤ 0.05 or ≥ 0.95 detection
│   │   │   └── Momentum           # 24h price movement + volume analysis
│   │   ├── BTCArbitrageEngine     # 5-exchange BTC vs Polymarket
│   │   │   └── Log-normal model   # Volatility-based probability (no lookup tables)
│   │   ├── WeatherEngine          # NOAA forecast vs Polymarket weather
│   │   ├── MetalsEngine           # XAU/XAG technical analysis
│   │   ├── SimulationEngine       # Paper trading with place/sell/resolve
│   │   └── AutoTrader             # Automated scan → bet → exit loop
│   ├── web_server.py       # aiohttp server + REST API + WebSocket broadcast
│   ├── xau_xag_engine.py   # Gold/Silver price engine
│   └── telegram_bot.py     # Telegram bot (notifications + commands)
├── static/
│   └── index.html          # Single-page trading terminal (vanilla JS)
├── data/
│   ├── sim_state.json      # Simulation state (auto-generated)
│   ├── pnl_history.json    # PnL snapshots for charting
│   └── last_scan.json      # Latest scan results
├── logs/                   # Application logs
├── .env                    # Configuration (not committed)
├── requirements.txt        # Python dependencies
└── VERSION                 # Current version
```

---

## Configuration

### Environment Variables (`.env`)

| Variable | Description | Default |
|----------|-------------|---------|
| `SIMULATION_MODE` | Paper trading mode (no real money) | `true` |
| `INITIAL_BALANCE` | Starting simulation balance | `100` |
| `MIN_EDGE_PCT` | Minimum edge % to place a bet | `15` |
| `LOOP_INTERVAL` | Auto-scan interval in seconds | `120` |
| `MAX_ORDER_USD` | Maximum bet size per trade | `50` |
| `MAX_DAILY_USD` | Daily spending limit (`0` = unlimited) | `0` |
| `WEB_PORT` | Dashboard port | `8080` |
| `NOAA_TOKEN` | NOAA weather API token | — |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token (from @BotFather) | — |
| `TELEGRAM_CHAT_ID` | Telegram chat ID for notifications | — |
| `POLY_PRIVATE_KEY` | Polymarket private key (for live trading) | — |
| `POLY_ADDRESS` | Polymarket wallet address | — |

### Exchange API Keys (Optional)

| Variable | Exchange |
|----------|----------|
| `BINANCE_API_KEY` / `BINANCE_SECRET` | Binance |
| `OKX_API_KEY` / `OKX_SECRET` / `OKX_PASSPHRASE` | OKX |
| `COINBASE_API_KEY` / `COINBASE_SECRET` | Coinbase |
| `UPBIT_ACCESS_KEY` / `UPBIT_SECRET_KEY` | Upbit |

---

## Telegram Bot

### Setup

1. Create a bot via [@BotFather](https://t.me/BotFather) on Telegram
2. Get your chat ID via [@userinfobot](https://t.me/userinfobot)
3. Add to `.env`:
   ```
   TELEGRAM_BOT_TOKEN=your_bot_token
   TELEGRAM_CHAT_ID=your_chat_id
   ```
4. Restart the bot — notifications start automatically

### Commands

| Command | Description |
|---------|-------------|
| `/status` | Portfolio status & open positions |
| `/history` | Recent closed trades |
| `/start` | Start AutoTrader |
| `/stop` | Stop AutoTrader |
| `/scan` | Run manual scan |
| `/report` | Daily performance report |
| `/help` | Show available commands |

### Notifications

- **Trade Open** — New position with entry price, edge, confidence
- **Trade Close** — Exit with PnL, reason (TP/SL/time-stop/resolve)
- **Daily Report** — Midnight summary: balance, P&L, win rate, trades

---

## Trading Logic

### Probability Models

| Engine | Method |
|--------|--------|
| **BTC** | Log-normal volatility model: `d = (ln(S/K) + 0.5σ²T) / (σ√T)`, BTC annual vol ≈ 70% |
| **Weather** | NOAA 7-day forecast vs market threshold, confidence from forecast certainty |
| **Universal** | Extreme price (≤5¢ or ≥95¢) + momentum (24h price change + volume) |
| **Metals** | Distance-to-target with time-weighted discount |

### Position Management

| Exit Type | Trigger | Action |
|-----------|---------|--------|
| **Take Profit** | Price moved ≥ +20% from entry | Sell at market |
| **Stop Loss** | Price moved ≥ -30% from entry | Sell at market |
| **Time Stop** | Open > 7 days, < 5% move | Sell (capital efficiency) |
| **Binary Resolve** | Market resolves YES/NO | Full payout or zero |

### Filters

| Filter | Value |
|--------|-------|
| Minimum edge | 15% |
| Max open positions | 15 |
| Max bets per cycle | 5 |
| Expiry filter (BTC/Weather) | > 72 hours |
| Expiry filter (Universal) | > 48 hours |
| Expiry filter (Metals) | > 7 days |
| Confidence floor | ≥ 30% |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/portfolio` | Portfolio summary + positions |
| `GET` | `/api/opportunities` | Current market opportunities |
| `GET` | `/api/btc-prices` | Live BTC prices from 5 exchanges |
| `GET` | `/api/trade-history` | Closed trades history |
| `GET` | `/api/pnl-history` | PnL time-series for charting |
| `GET` | `/api/auto/status` | AutoTrader status & stats |
| `POST` | `/api/auto/start` | Start AutoTrader |
| `POST` | `/api/auto/stop` | Stop AutoTrader |
| `POST` | `/api/auto/config` | Update scan interval / min edge |
| `POST` | `/api/auto/scan-now` | Trigger immediate scan |
| `POST` | `/api/place-bet` | Manual bet placement |
| `POST` | `/api/reset-sim` | Reset simulation |
| `WS` | `/ws` | Real-time WebSocket (3s broadcast) |

---

## Version Management

```bash
python bump_version.py patch  # 1.5.0 → 1.5.1
python bump_version.py minor  # 1.5.0 → 1.6.0
python bump_version.py major  # 1.5.0 → 2.0.0
```

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| **v1.5.0** | 2025-02-24 | Smart strategy overhaul: `UniversalMarketEngine` scans ALL Polymarket markets (extreme price + momentum + volume spike), BTC log-normal probability model (replaces fabricated lookup table), `sell_position()` with TP/SL/time-stop exits, confidence-based bet sizing, per-type expiry filters, max 15 positions, 5 bets/cycle, dashboard: history auto-update, 15s chart refresh, 30 opportunities via WS, daily P&L fix, auto-scanner stats persistence, polling fallback |
| **v1.4.0** | 2025-02-24 | Strategy overhaul: fix directional edge filter (abs→signed), add 72h expiry filter, confidence floor ≥30%, max 6 positions cap, metals distance-to-target discount, MIN_EDGE 25%, real-time closed trades in dashboard, full English codebase |
| **v1.3.0** | 2025-02-24 | Full English UI, professional Forex EA-style Telegram notifications (trade open/close/daily report), auto-trade event detection, `/report` command |
| **v1.2.0** | 2025-02-24 | New v9 Trading Terminal dashboard, Telegram bot live integration, WS scan_interval broadcast, live PnL/charts/stats |
| **v1.1.0** | 2025-02-24 | Full English UI, chart stats fix, Telegram bot integration, live statistics panel |
| **v1.0.0** | 2025-02-24 | Initial release: BTC/Weather/Metals engines, auto-trader, web dashboard, simulation mode |

---

## License

MIT
