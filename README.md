# Poly Arb Bot

Polymarket arbitrage/edge detection bot with real-time multi-source analysis.

## Features

- **Multi-Engine Analysis**: BTC (5 exchanges), Weather (NOAA), Metals (XAU/XAG)
- **Real-Time Data**: Live orderbook prices from Polymarket CLOB API
- **Auto-Trader**: Automated scanning, betting, and position management
- **Smart Filters**: Liquidity, spread, confidence, and EV-based opportunity ranking
- **Position Management**: Tiered TP/SL thresholds + 1-hour force close with realistic PnL
- **Web Dashboard**: Real-time portfolio, positions, PnL chart, live prices
- **Telegram Bot**: Real-time notifications + command control via Telegram
- **Simulation Mode**: Paper trading with real market data

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and configure .env
cp .env.example .env
# Edit .env with your API keys

# Run the bot
python src/web_server.py
```

Open `http://localhost:8080` for the dashboard.

## Telegram Bot Setup

1. Create a bot via [@BotFather](https://t.me/BotFather) on Telegram
2. Get your chat ID via [@userinfobot](https://t.me/userinfobot)
3. Add to `.env`:
   ```
   TELEGRAM_BOT_TOKEN=your_bot_token
   TELEGRAM_CHAT_ID=your_chat_id
   ```
4. Restart the bot - Telegram notifications will start automatically

### Telegram Commands

| Command | Description |
|---------|-------------|
| `/status` | Portfolio status & open positions |
| `/history` | Recent closed trades |
| `/start` | Start AutoTrader |
| `/stop` | Stop AutoTrader |
| `/scan` | Run manual scan |
| `/help` | Show available commands |

## Architecture

```
src/
  bot.py            - Core engines (BTC, Weather, Metals, AutoTrader, SimulationEngine)
  web_server.py     - aiohttp web server + REST API + WebSocket
  xau_xag_engine.py - Gold/Silver technical analysis engine
  telegram_bot.py   - Telegram bot integration (notifications + commands)
static/
  index.html        - Single-page dashboard (vanilla JS)
data/
  sim_state.json    - Simulation state (auto-generated)
```

## Configuration (.env)

| Variable | Description | Default |
|----------|-------------|---------|
| `MIN_EDGE_PCT` | Minimum edge % to place bet | 8 |
| `LOOP_INTERVAL` | Scan interval in seconds | 120 |
| `MAX_ORDER_USD` | Max bet size per trade | 50 |
| `MAX_DAILY_USD` | Daily spending limit (0=unlimited) | 0 |
| `SIMULATION_MODE` | Paper trading mode | true |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token | — |
| `TELEGRAM_CHAT_ID` | Telegram chat ID for notifications | — |

## Version Management

```bash
python bump_version.py patch  # 1.0.0 -> 1.0.1
python bump_version.py minor  # 1.0.0 -> 1.1.0
python bump_version.py major  # 1.0.0 -> 2.0.0
```

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| v1.5.0 | 2026-02-24 | Smart strategy overhaul: UniversalMarketEngine scans ALL Polymarket markets (extreme price + momentum + volume spike), BTC log-normal probability model (replaces fabricated lookup), sell_position() with TP/SL/time-stop exits, confidence-based bet sizing, per-type expiry filters, max 15 positions, 5 bets/cycle, dashboard: history auto-update, 15s chart refresh, 30 opportunities via WS |
| v1.4.0 | 2026-02-24 | Strategy overhaul: fix directional edge filter (abs→signed), add 72h expiry filter, confidence floor ≥30%, max 6 positions cap, metals distance-to-target discount, MIN_EDGE 25%, real-time closed trades in dashboard, full English codebase |
| v1.3.0 | 2026-02-24 | Full English UI, professional Forex EA-style Telegram notifications (trade open/close/daily report), auto-trade event detection, /report command |
| v1.2.0 | 2026-02-24 | New v9 Trading Terminal dashboard, Telegram bot live integration, WS scan_interval broadcast, live PnL/charts/stats |
| v1.1.0 | 2026-02-24 | Full English UI, chart stats fix, Telegram bot integration, live statistics panel |
| v1.0.0 | 2026-02-24 | Initial release: BTC/Weather/Metals engines, auto-trader, web dashboard, simulation mode |

## License

MIT
