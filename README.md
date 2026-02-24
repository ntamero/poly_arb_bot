# Poly Arb Bot

Polymarket arbitrage/edge detection bot with real-time multi-source analysis.

## Features

- **Multi-Engine Analysis**: BTC (5 exchanges), Weather (NOAA), Metals (XAU/XAG)
- **Real-Time Data**: Live orderbook prices from Polymarket CLOB API
- **Auto-Trader**: Automated scanning, betting, and position management
- **Smart Filters**: Liquidity, spread, confidence, and EV-based opportunity ranking
- **Position Management**: Tiered TP/SL thresholds + 1-hour force close with realistic PnL
- **Web Dashboard**: Real-time portfolio, positions, PnL chart, live prices
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

## Architecture

```
src/
  bot.py          - Core engines (BTC, Weather, Metals, AutoTrader, SimulationEngine)
  web_server.py   - aiohttp web server + REST API + WebSocket
  xau_xag_engine.py - Gold/Silver technical analysis engine
static/
  index.html      - Single-page dashboard (vanilla JS)
data/
  sim_state.json  - Simulation state (auto-generated)
```

## Configuration (.env)

| Variable | Description | Default |
|----------|-------------|---------|
| `MIN_EDGE_PCT` | Minimum edge % to place bet | 8 |
| `LOOP_INTERVAL` | Scan interval in seconds | 120 |
| `MAX_ORDER_USD` | Max bet size per trade | 50 |
| `MAX_DAILY_USD` | Daily spending limit (0=unlimited) | 0 |
| `SIMULATION_MODE` | Paper trading mode | true |

## Version Management

```bash
python bump_version.py patch  # 1.0.0 -> 1.0.1
python bump_version.py minor  # 1.0.0 -> 1.1.0
python bump_version.py major  # 1.0.0 -> 2.0.0
```

## License

MIT
