"""
Polymarket Simulation Web Dashboard v3.0 - AUTOMATIC TRADING
- Fetches real Polymarket data
- Virtual betting (manual + automatic)
- AutoTrader: Automatic BTC (5 exchanges) and NOAA analysis betting
- Portfolio tracking, win rate analysis
- Probability analysis dashboard
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
from aiohttp import web

# Set up project path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bot import (
    CFG, PolymarketClient, PriceAggregator, NOAAClient,
    WeatherArbEngine, BTCArbEngine, MetalsArbEngine, GeneralMarketScanner,
    SimulationEngine, AutoTrader, DATA_DIR, LOG_DIR
)

from telegram_bot import get_telegram_bot

import logging
log = logging.getLogger("web-server")

# --- GLOBAL STATE ---
sim_engine = SimulationEngine(CFG.initial_balance)
auto_trader = AutoTrader(sim_engine)
poly_client = PolymarketClient()
price_agg = PriceAggregator()
weather_engine = WeatherArbEngine()
btc_engine = BTCArbEngine()
metals_engine = MetalsArbEngine()
scanner = GeneralMarketScanner()
cached_data = {"markets": [], "btc_prices": {}, "weather": {}, "metals": {}, "opportunities": [], "last_update": None}
# PnL time series
pnl_history = []
MAX_PNL_HISTORY = 500


def record_pnl_snapshot():
    summary = sim_engine.get_portfolio_summary()
    total_invested = sum(p["amount"] for p in sim_engine.positions)
    realized_pnl = sum(t.get("pnl", 0) for t in sim_engine.closed_trades)
    unrealized_pnl = 0
    try:
        unrealized_pnl = sum(
            lp.get("unrealized_pnl", 0)
            for lp in live_cache.position_prices.values()
        )
    except Exception:
        pass
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "ts_ms": int(datetime.now(timezone.utc).timestamp() * 1000),
        "balance": round(summary["balance"], 4),
        "total_invested": round(total_invested, 4),
        "portfolio_value": round(summary["balance"] + total_invested, 4),
        "realized_pnl": round(realized_pnl, 4),
        "unrealized_pnl": round(unrealized_pnl, 4),
        "total_pnl": round(realized_pnl + unrealized_pnl, 4),
        "open_positions": summary["open_positions"],
        "closed_trades": summary["closed_trades"],
        "win_rate": summary["win_rate"],
    }
    pnl_history.append(entry)
    if len(pnl_history) > MAX_PNL_HISTORY:
        pnl_history.pop(0)
    try:
        with open(DATA_DIR / "pnl_history.json", "w") as f:
            json.dump(pnl_history[-MAX_PNL_HISTORY:], f)
    except Exception:
        pass
    return entry


def load_pnl_history():
    try:
        pf = DATA_DIR / "pnl_history.json"
        if pf.exists():
            with open(pf) as f:
                pnl_history.extend(json.load(f))
            log.info(f"PnL history loaded: {len(pnl_history)} records")
    except Exception as e:
        log.warning(f"PnL history load: {e}")


# --- LIVE PRICE CACHE (for 3s polling) ----------------------------------------

class LivePriceCache:
    """Caches BTC + open position orderbook prices.
    Frontend polls every 3s, backend calls real API at most every 5s."""

    def __init__(self):
        self.btc_prices = {}
        self.btc_avg = 0
        self.btc_spread = 0
        self.btc_spread_pct = 0
        self.position_prices = {}   # market_id -> {price, spread, liquid, side, market}
        self.last_btc_fetch = 0
        self.last_pos_fetch = 0
        self.btc_ttl = 5            # BTC prices 5s cache
        self.pos_ttl = 8            # Position prices 8s cache
        self._lock = asyncio.Lock()
        self.tick_count = 0

    async def get_live_data(self):
        """Return live prices, refresh if stale"""
        now = time.time()
        btc_stale = (now - self.last_btc_fetch) >= self.btc_ttl
        pos_stale = (now - self.last_pos_fetch) >= self.pos_ttl

        tasks = []
        if btc_stale:
            tasks.append(self._refresh_btc())
        if pos_stale:
            tasks.append(self._refresh_positions())

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self.tick_count += 1
        return {
            "btc": {
                "prices": self.btc_prices,
                "average": self.btc_avg,
                "spread": self.btc_spread,
                "spread_pct": self.btc_spread_pct,
            },
            "positions": self.position_prices,
            "portfolio": {
                "balance": sim_engine.balance,
                "total_pnl": round(sum(p.get("pnl", 0) for p in sim_engine.positions), 2),
                "open_count": len(sim_engine.positions),
            },
            "auto_running": auto_trader.running,
            "scan_count": auto_trader.scan_count,
            "auto_bets": auto_trader.auto_bets_placed,
            "ts": datetime.now(timezone.utc).isoformat(),
            "tick": self.tick_count,
        }

    async def _refresh_btc(self):
        try:
            prices = await price_agg.get_all_prices()
            if prices:
                self.btc_prices = prices
                vals = list(prices.values())
                self.btc_avg = round(sum(vals) / len(vals), 2)
                if len(vals) >= 2:
                    self.btc_spread = round(max(vals) - min(vals), 2)
                    self.btc_spread_pct = round(self.btc_spread / self.btc_avg * 100, 3) if self.btc_avg > 0 else 0
                self.last_btc_fetch = time.time()
        except Exception as e:
            log.error(f"Live BTC refresh error: {e}")

    async def _refresh_positions(self):
        try:
            positions = sim_engine.positions
            if not positions:
                self.position_prices = {}
                self.last_pos_fetch = time.time()
                return

            result = {}
            tasks = []
            for pos in positions:
                # Get orderbook price for each position
                token_id = pos.get("token_id", "")
                no_token_id = pos.get("no_token_id", "")
                tid = token_id if pos.get("side") == "YES" else no_token_id
                if tid:
                    tasks.append((pos, poly_client.get_smart_price(tid)))

            if tasks:
                prices = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)
                for i, (pos, _) in enumerate(tasks):
                    price_data = prices[i]
                    if isinstance(price_data, tuple) and len(price_data) == 3:
                        mp, sp, liq = price_data
                    else:
                        mp, sp, liq = 0, 1.0, False

                    mid = pos.get("market_id", "")
                    entry = pos.get("entry_price", 0)
                    side = pos.get("side", "YES")
                    amount = pos.get("amount", 0)
                    # Calculate PnL: shares * current_price - amount
                    # tid is already the correct token (YES token for YES, NO token for NO)
                    if entry > 0 and mp > 0:
                        shares = amount / entry
                        current_val = shares * mp
                    else:
                        current_val = amount
                    unrealized_pnl = round(current_val - amount, 4)

                    result[mid] = {
                        "price": mp,
                        "spread": sp,
                        "liquid": liq,
                        "side": side,
                        "market": (pos.get("market", ""))[:50],
                        "entry_price": entry,
                        "amount": pos.get("amount", 0),
                        "unrealized_pnl": unrealized_pnl,
                        "type": pos.get("type", ""),
                    }

            self.position_prices = result
            self.last_pos_fetch = time.time()
        except Exception as e:
            log.error(f"Live position refresh error: {e}")


live_cache = LivePriceCache()


# --- API ROUTES ---------------------------------------------------------------

async def handle_index(request):
    """Main page - dashboard HTML"""
    html_path = PROJECT_ROOT / "static" / "index.html"
    if html_path.exists():
        return web.FileResponse(html_path)
    return web.Response(text="static/index.html not found", status=404)


async def handle_api_status(request):
    """System status"""
    return web.json_response({
        "status": "online",
        "mode": "simulation" if CFG.simulation_mode else "live",
        "version": "3.0",
        "uptime": datetime.now(timezone.utc).isoformat(),
        "auto_trader_running": auto_trader.running,
        "config": {
            "min_edge_pct": CFG.min_edge_pct,
            "max_order_usd": CFG.max_order_usd,
            "max_daily_usd": CFG.max_daily_usd,
            "min_liquidity": CFG.min_liquidity,
            "loop_interval": CFG.loop_interval_sec,
        }
    })


async def handle_api_portfolio(request):
    """Portfolio summary — with unrealized PnL"""
    summary = sim_engine.get_portfolio_summary()
    summary["positions"] = sim_engine.positions
    summary["recent_trades"] = sim_engine.closed_trades[-20:]

    # Calculate unrealized PnL (from live cache)
    try:
        unrealized = sum(
            lp.get("unrealized_pnl", 0)
            for lp in live_cache.position_prices.values()
        )
        summary["unrealized_pnl"] = round(unrealized, 4)
        summary["total_pnl"] = round(summary["realized_pnl"] + unrealized, 4)
    except Exception:
        pass

    return web.json_response(summary)


async def handle_api_markets(request):
    """Fetch all active markets"""
    category = request.query.get("category", "all")
    try:
        if category == "weather":
            markets = await poly_client.get_weather_markets()
        elif category == "btc":
            markets = await poly_client.get_btc_markets()
        else:
            markets = await poly_client.get_all_active_markets(limit=100)

        # Format required fields for each market + real price from CLOB
        cleaned = []
        token_ids_to_fetch = []
        market_token_map = []
        for m in markets[:50]:
            tokens = m.get("tokens", [])
            yes_token = next((t for t in tokens if t.get("outcome", "").upper() == "YES"), None)
            no_token = next((t for t in tokens if t.get("outcome", "").upper() == "NO"), None)
            yes_tid = yes_token.get("token_id", "") if yes_token else ""
            no_tid = no_token.get("token_id", "") if no_token else ""
            if yes_tid:
                token_ids_to_fetch.append(yes_tid)
            if no_tid:
                token_ids_to_fetch.append(no_tid)
            market_token_map.append((m, yes_token, no_token, yes_tid, no_tid))

        # Batch price fetch
        prices = await poly_client.get_token_prices_batch(token_ids_to_fetch) if token_ids_to_fetch else {}

        for m, yes_token, no_token, yes_tid, no_tid in market_token_map:
            yes_price = prices.get(yes_tid, 0)
            no_price = prices.get(no_tid, 0)
            if yes_price == 0 and no_price > 0:
                yes_price = round(1 - no_price, 4)
            elif no_price == 0 and yes_price > 0:
                no_price = round(1 - yes_price, 4)

            cleaned.append({
                "id": m.get("condition_id", "")[:20],
                "question": m.get("question", ""),
                "description": (m.get("description", "") or "")[:200],
                "yes_price": yes_price,
                "no_price": no_price,
                "yes_token_id": yes_tid,
                "no_token_id": no_tid,
                "liquidity": 0,
                "volume": 0,
                "end_date": m.get("end_date_iso", ""),
                "category": "",
                "condition_id": m.get("condition_id", ""),
                "image": "",
            })

        cached_data["markets"] = cleaned
        cached_data["last_update"] = datetime.now(timezone.utc).isoformat()
        return web.json_response({"markets": cleaned, "count": len(cleaned)})
    except Exception as e:
        log.error(f"Markets API error: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_api_opportunities(request):
    """Scan for arbitrage opportunities"""
    try:
        weather_opps, btc_opps, metals_opps = await asyncio.gather(
            weather_engine.find_opportunities(),
            btc_engine.find_opportunities(),
            metals_engine.find_opportunities(),
        )
        all_opps = weather_opps + btc_opps + metals_opps
        all_opps.sort(key=lambda x: abs(x["edge"]), reverse=True)

        cached_data["opportunities"] = all_opps
        return web.json_response({
            "opportunities": all_opps,
            "weather_count": len(weather_opps),
            "btc_count": len(btc_opps),
            "metals_count": len(metals_opps),
            "total": len(all_opps),
            "scanned_at": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        log.error(f"Opportunities API error: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_api_btc_prices(request):
    """Fetch BTC prices"""
    try:
        prices = await price_agg.get_all_prices()
        avg = sum(prices.values()) / len(prices) if prices else 0
        result = {
            "prices": prices,
            "average": round(avg, 2),
            "spread": round((max(prices.values()) - min(prices.values())), 2) if len(prices) >= 2 else 0,
            "spread_pct": round(((max(prices.values()) - min(prices.values())) / avg) * 100, 3) if avg > 0 and len(prices) >= 2 else 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        cached_data["btc_prices"] = result
        return web.json_response(result)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_api_weather(request):
    """Fetch NOAA weather data"""
    try:
        noaa = NOAAClient()
        forecasts = await asyncio.gather(*[
            noaa.get_hourly_forecast(lat, lon, city)
            for lat, lon, city in CFG.cities
        ])
        result = [f for f in forecasts if f]
        cached_data["weather"] = result
        return web.json_response({
            "forecasts": result,
            "cities_requested": len(CFG.cities),
            "cities_responded": len(result),
        })
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_api_place_bet(request):
    """Place virtual bet"""
    try:
        data = await request.json()
        market_id = data.get("market_id", "")
        market_name = data.get("market_name", "Unknown")
        side = data.get("side", "YES").upper()
        amount = float(data.get("amount", 10))
        price = float(data.get("price", 0.5))
        edge = float(data.get("edge", 0))
        market_type = data.get("market_type", "general")

        if side not in ("YES", "NO"):
            return web.json_response({"error": "Side must be YES or NO"}, status=400)
        if amount <= 0:
            return web.json_response({"error": "Amount must be positive"}, status=400)
        if not (0 < price < 1):
            return web.json_response({"error": "Price must be between 0 and 1"}, status=400)

        result = sim_engine.place_bet(market_id, market_name, side, amount, price, edge, market_type)
        if "error" in result:
            return web.json_response(result, status=400)

        # Telegram handled by event detection in ws_broadcast_task

        return web.json_response({
            "success": True,
            "trade": result,
            "new_balance": sim_engine.balance,
        })
    except Exception as e:
        return web.json_response({"error": str(e)}, status=400)


async def handle_api_resolve(request):
    """Resolve position"""
    try:
        data = await request.json()
        position_id = data.get("position_id", "")
        outcome = data.get("outcome", "YES").upper()

        result = sim_engine.resolve_position(position_id, outcome)
        if "error" in result:
            return web.json_response(result, status=400)

        # Telegram handled by event detection in ws_broadcast_task

        return web.json_response({
            "success": True,
            "result": result,
            "new_balance": sim_engine.balance,
        })
    except Exception as e:
        return web.json_response({"error": str(e)}, status=400)


async def handle_api_reset_sim(request):
    """Reset simulation"""
    # Stop AutoTrader
    if auto_trader.running:
        auto_trader.stop()

    sim_engine.balance = CFG.initial_balance
    sim_engine.positions = []
    sim_engine.closed_trades = []
    sim_engine.trade_history = []
    sim_engine.save_state()

    # Reset AutoTrader stats
    auto_trader.scan_count = 0
    auto_trader.auto_bets_placed = 0
    auto_trader.auto_resolves = 0
    auto_trader.daily_spent = 0.0
    auto_trader.stats = {
        "total_scans": 0, "total_bets_placed": 0, "total_resolved": 0,
        "weather_bets": 0, "btc_bets": 0, "last_btc_prices": {},
        "last_weather_data": [], "last_opportunities": [], "cycle_log": [],
        "started_at": None, "errors": 0,
    }

    return web.json_response({
        "success": True,
        "balance": sim_engine.balance,
        "message": "Simulation reset"
    })


async def handle_api_trade_history(request):
    """Trade history"""
    limit = int(request.query.get("limit", "50"))
    return web.json_response({
        "trades": sim_engine.trade_history[-limit:],
        "total": len(sim_engine.trade_history),
    })


async def handle_api_analysis(request):
    """Probability analysis report"""
    try:
        scan_file = DATA_DIR / "last_scan.json"
        scan_data = {}
        if scan_file.exists():
            with open(scan_file, "r") as f:
                scan_data = json.load(f)

        portfolio = sim_engine.get_portfolio_summary()

        all_opps = scan_data.get("weather_opportunities", []) + scan_data.get("btc_opportunities", [])
        edges = [o.get("edge", 0) for o in all_opps]

        analysis = {
            "portfolio": portfolio,
            "opportunity_stats": {
                "total": len(all_opps),
                "avg_edge": round(sum(abs(e) for e in edges) / len(edges) * 100, 2) if edges else 0,
                "max_edge": round(max(abs(e) for e in edges) * 100, 2) if edges else 0,
                "weather_count": len(scan_data.get("weather_opportunities", [])),
                "btc_count": len(scan_data.get("btc_opportunities", [])),
            },
            "market_overview": {
                "total_markets": len(scan_data.get("general_markets", [])),
                "total_liquidity": sum(m.get("liquidity", 0) for m in scan_data.get("general_markets", [])),
            },
            "last_scan": scan_data.get("ts", "never"),
        }

        return web.json_response(analysis)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


# --- AUTO-TRADER API ENDPOINTS ------------------------------------------------

async def handle_auto_status(request):
    """AutoTrader status"""
    return web.json_response(auto_trader.get_status())


async def handle_auto_start(request):
    """Start AutoTrader"""
    result = auto_trader.start()
    log.info(f"AutoTrader start: {result}")
    return web.json_response(result)


async def handle_auto_stop(request):
    """Stop AutoTrader"""
    result = auto_trader.stop()
    log.info(f"AutoTrader stop: {result}")
    return web.json_response(result)


async def handle_auto_config(request):
    """Update AutoTrader configuration"""
    try:
        data = await request.json()
        result = auto_trader.update_config(**data)
        return web.json_response(result)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=400)


async def handle_auto_scan_now(request):
    """Run one-off scan (independent from AutoTrader loop)"""
    try:
        result = await auto_trader.run_scan_cycle()
        # Telegram handled by event detection in ws_broadcast_task
        return web.json_response(result)
    except Exception as e:
        log.error(f"Manual scan error: {e}")
        return web.json_response({"error": str(e)}, status=500)



async def handle_pnl_history(request):
    """PnL time series -- for charts"""
    limit = int(request.query.get("limit", "200"))
    period = request.query.get("period", "all")  # all, 1h, 6h, 24h

    data = pnl_history[-limit:]

    if period != "all" and data:
        hours = {"1h": 1, "6h": 6, "24h": 24}.get(period, 999)
        cutoff_ms = (datetime.now(timezone.utc).timestamp() - hours * 3600) * 1000
        data = [d for d in data if d.get("ts_ms", 0) >= cutoff_ms]

    # Initial balance
    initial = 10000.0
    try:
        initial = float(sim_engine.initial_balance)
    except Exception:
        pass

    return web.json_response({
        "history": data,
        "count": len(data),
        "initial_balance": initial,
        "current": pnl_history[-1] if pnl_history else {},
    })

async def handle_auto_logs(request):
    """Fetch AutoTrader logs"""
    limit = int(request.query.get("limit", "50"))
    logs = auto_trader.stats.get("cycle_log", [])[-limit:]
    return web.json_response({"logs": logs, "total": len(auto_trader.stats.get("cycle_log", []))})


async def handle_live_prices(request):
    """Live price data - lightweight endpoint optimized for 3s polling"""
    try:
        data = await live_cache.get_live_data()
        return web.json_response(data)
    except Exception as e:
        log.error(f"Live prices error: {e}")
        return web.json_response({"error": str(e)}, status=500)


# WebSocket connections
_ws_clients = set()

async def handle_websocket(request):
    """WebSocket endpoint - real-time live data stream"""
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)
    _ws_clients.add(ws)
    log.info(f"WS: New connection, total: {len(_ws_clients)}")

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    cmd = data.get("cmd", "")
                    if cmd == "ping":
                        await ws.send_json({"type": "pong", "ts": datetime.now(timezone.utc).isoformat()})
                    elif cmd == "get_portfolio":
                        summary = sim_engine.get_portfolio_summary()
                        summary["positions"] = sim_engine.positions
                        await ws.send_json({"type": "portfolio", "data": summary})
                    elif cmd == "scan_now":
                        await ws.send_json({"type": "scan_start", "ts": datetime.now(timezone.utc).isoformat()})
                        result = await auto_trader.run_scan_cycle()
                        await ws.send_json({"type": "scan_result", "data": result})
                except Exception as e:
                    await ws.send_json({"type": "error", "msg": str(e)})
            elif msg.type in (web.WSMsgType.ERROR, web.WSMsgType.CLOSE):
                break
    except Exception as e:
        log.error(f"WS error: {e}")
    finally:
        _ws_clients.discard(ws)
        log.info(f"WS: Connection closed, remaining: {len(_ws_clients)}")

    return ws


async def ws_broadcast_task(app):
    """WebSocket broadcast — every 3s send live data + detect trade events for Telegram"""
    # Event detection: track counts to catch new trades (both manual + auto)
    _prev_trade_count = len(sim_engine.trade_history)
    _prev_closed_count = len(sim_engine.closed_trades)
    tg = get_telegram_bot()

    while True:
        try:
            await asyncio.sleep(3)

            # ── EVENT DETECTION: new trades opened ──
            curr_trade_count = len(sim_engine.trade_history)
            if curr_trade_count > _prev_trade_count:
                for trade in sim_engine.trade_history[_prev_trade_count:]:
                    try:
                        asyncio.ensure_future(tg.notify_trade_opened(trade))
                    except Exception as te:
                        log.error(f"Telegram trade-open notify error: {te}")
                _prev_trade_count = curr_trade_count

            # ── EVENT DETECTION: positions closed ──
            curr_closed_count = len(sim_engine.closed_trades)
            if curr_closed_count > _prev_closed_count:
                for trade in sim_engine.closed_trades[_prev_closed_count:]:
                    try:
                        asyncio.ensure_future(tg.notify_trade_closed(trade))
                    except Exception as te:
                        log.error(f"Telegram trade-close notify error: {te}")
                _prev_closed_count = curr_closed_count

            # ── DAILY REPORT CHECK (midnight) ──
            try:
                await tg.check_daily_report()
            except Exception:
                pass

            # PnL snapshot every ~30s (with or without clients)
            if live_cache.tick_count % 10 == 0:
                try:
                    record_pnl_snapshot()
                except Exception:
                    pass

            # Skip data collection if no clients
            if not _ws_clients:
                continue

            data = await live_cache.get_live_data()
            summary = sim_engine.get_portfolio_summary()
            # Unrealized PnL from live cache
            try:
                unrealized = sum(
                    lp.get("unrealized_pnl", 0) for lp in live_cache.position_prices.values()
                )
                summary["unrealized_pnl"] = round(unrealized, 4)
                summary["total_pnl"] = round(summary["realized_pnl"] + unrealized, 4)
            except Exception:
                pass
            data["portfolio_full"] = summary
            data["positions"] = sim_engine.positions
            data["position_prices"] = live_cache.position_prices
            data["auto_status"] = {
                "running": auto_trader.running,
                "scan_count": auto_trader.scan_count,
                "auto_bets": auto_trader.auto_bets_placed,
                "auto_resolves": auto_trader.auto_resolves,
                "next_scan": auto_trader.next_scan_time,
                "scan_interval": auto_trader.scan_interval,
                "daily_spent": round(auto_trader.daily_spent, 2),
                "last_logs": auto_trader.stats.get("cycle_log", [])[-5:],
                "last_opportunities": auto_trader.stats.get("last_opportunities", [])[:10],
            }

            msg = json.dumps({"type": "live_update", "data": data}, default=str)
            # Use discard() instead of -= dead — avoids global scope error
            for client in list(_ws_clients):
                try:
                    await client.send_str(msg)
                except Exception:
                    _ws_clients.discard(client)

        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error(f"WS broadcast error: {e}")


async def start_ws_broadcast(app):
    load_pnl_history()
    record_pnl_snapshot()  # Initial snapshot
    app["ws_task"] = asyncio.ensure_future(ws_broadcast_task(app))

    # Start Telegram bot
    tg = get_telegram_bot()
    tg.set_engines(sim_engine, auto_trader, live_cache)
    tg.start_polling()
    log.info("Telegram bot polling started" if tg.enabled else "Telegram bot disabled")


async def stop_ws_broadcast(app):
    # Stop Telegram bot
    tg = get_telegram_bot()
    tg.stop_polling()

    task = app.get("ws_task")
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# --- CORS MIDDLEWARE ----------------------------------------------------------

@web.middleware
async def cors_middleware(request, handler):
    if request.method == "OPTIONS":
        response = web.Response()
    else:
        try:
            response = await handler(request)
        except web.HTTPException as ex:
            response = ex

    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


# --- APP SETUP ----------------------------------------------------------------

def create_app():
    app = web.Application(middlewares=[cors_middleware])

    # Static files
    static_dir = PROJECT_ROOT / "static"
    static_dir.mkdir(exist_ok=True)

    # Routes
    app.router.add_get("/", handle_index)
    app.router.add_get("/api/status", handle_api_status)
    app.router.add_get("/api/portfolio", handle_api_portfolio)
    app.router.add_get("/api/markets", handle_api_markets)
    app.router.add_get("/api/opportunities", handle_api_opportunities)
    app.router.add_get("/api/btc-prices", handle_api_btc_prices)
    app.router.add_get("/api/weather", handle_api_weather)
    app.router.add_get("/api/analysis", handle_api_analysis)
    app.router.add_get("/api/trade-history", handle_api_trade_history)
    app.router.add_post("/api/place-bet", handle_api_place_bet)
    app.router.add_post("/api/resolve", handle_api_resolve)
    app.router.add_post("/api/reset-sim", handle_api_reset_sim)

    # AutoTrader Routes
    app.router.add_get("/api/auto/status", handle_auto_status)
    app.router.add_post("/api/auto/start", handle_auto_start)
    app.router.add_post("/api/auto/stop", handle_auto_stop)
    app.router.add_post("/api/auto/config", handle_auto_config)
    app.router.add_post("/api/auto/scan-now", handle_auto_scan_now)
    app.router.add_get("/api/auto/logs", handle_auto_logs)
    app.router.add_get("/api/pnl-history", handle_pnl_history)

    # Live price endpoint (3s polling)
    app.router.add_get("/api/live-prices", handle_live_prices)

    # WebSocket endpoint
    app.router.add_get("/ws", handle_websocket)

    # Serve static files
    app.router.add_static("/static/", path=str(static_dir), name="static")

    # WebSocket broadcast task
    app.on_startup.append(start_ws_broadcast)
    app.on_shutdown.append(stop_ws_broadcast)

    return app


if __name__ == "__main__":
    port = int(os.getenv("WEB_PORT", "8080"))
    log.info(f"Web dashboard starting: http://localhost:{port}")
    app = create_app()
    web.run_app(app, host="0.0.0.0", port=port)
