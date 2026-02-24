"""
Polymarket Arbitrage Bot v2.0
- Weather (NOAA) -> Polymarket Weather Markets
- BTC Price Arbitrage -> Binance / OKX / Coinbase / Upbit vs Polymarket
- Simulation Mode support
- Scanning markets across all categories
"""

import asyncio
import aiohttp
import json
import logging
import os
import math
import re
import sys
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# Proje kok dizinini bul
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Log dizinini olustur
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "bot.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger("arb-bot")


# --- KONFIGURASYON -----------------------------------------------------------

@dataclass
class Config:
    # Polymarket
    poly_private_key: str = ""
    poly_address: str = ""

    # NOAA
    noaa_token: str = ""

    # Borsa API Keys
    binance_key: str = ""
    binance_secret: str = ""
    okx_key: str = ""
    okx_secret: str = ""
    okx_passphrase: str = ""
    coinbase_key: str = ""
    coinbase_secret: str = ""
    upbit_key: str = ""
    upbit_secret: str = ""

    # Risk Parametreleri
    max_order_usd: float = 20.0
    min_edge_pct: float = 15.0
    min_liquidity: float = 5000.0
    max_daily_usd: float = 100.0
    loop_interval_sec: int = 300

    # Simulasyon
    simulation_mode: bool = True
    initial_balance: float = 10000.0

    # Weather cities (US only - NOAA only supports US)
    cities: list = field(default_factory=list)

    def __post_init__(self):
        self.poly_private_key = os.getenv("POLY_PRIVATE_KEY", "")
        self.poly_address = os.getenv("POLY_ADDRESS", "")
        self.noaa_token = os.getenv("NOAA_TOKEN", "")
        self.binance_key = os.getenv("BINANCE_API_KEY", "")
        self.binance_secret = os.getenv("BINANCE_SECRET", "")
        self.okx_key = os.getenv("OKX_API_KEY", "")
        self.okx_secret = os.getenv("OKX_SECRET", "")
        self.okx_passphrase = os.getenv("OKX_PASSPHRASE", "")
        self.coinbase_key = os.getenv("COINBASE_API_KEY", "")
        self.coinbase_secret = os.getenv("COINBASE_SECRET", "")
        self.upbit_key = os.getenv("UPBIT_ACCESS_KEY", "")
        self.upbit_secret = os.getenv("UPBIT_SECRET_KEY", "")
        self.max_order_usd = float(os.getenv("MAX_ORDER_USD", "20"))
        self.min_edge_pct = float(os.getenv("MIN_EDGE_PCT", "15"))
        self.min_liquidity = float(os.getenv("MIN_LIQUIDITY", "5000"))
        self.max_daily_usd = float(os.getenv("MAX_DAILY_USD", "100"))
        self.loop_interval_sec = int(os.getenv("LOOP_INTERVAL", "300"))
        self.simulation_mode = os.getenv("SIMULATION_MODE", "true").lower() == "true"
        self.initial_balance = float(os.getenv("INITIAL_BALANCE", "10000"))
        # NOAA only supports US cities
        self.cities = [
            (40.7128, -74.0060, "NYC"),
            (47.6062, -122.3321, "Seattle"),
            (34.0522, -118.2437, "LA"),
            (41.8781, -87.6298, "Chicago"),
            (29.7604, -95.3698, "Houston"),
            (33.4484, -112.0740, "Phoenix"),
            (25.7617, -80.1918, "Miami"),
        ]


CFG = Config()


# --- TRADE LOGGER ------------------------------------------------------------

def log_trade(action: str, market: str, amount: float, price: float, reason: str,
              sim: bool = False):
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "market": market,
        "amount_usd": amount,
        "price": price,
        "reason": reason,
        "simulation": sim,
    }
    trades_file = DATA_DIR / ("sim_trades.json" if sim else "trades.json")
    with open(trades_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    prefix = "[SIM]" if sim else "[LIVE]"
    log.info(f"{prefix} TRADE | {action} | {market} | ${amount:.2f} @ {price:.3f} | {reason}")


# --- NOAA WEATHER ------------------------------------------------------------

class NOAAClient:
    """NOAA Weather API - works for US cities only"""
    BASE = "https://api.weather.gov"

    def _headers(self):
        h = {"User-Agent": "polymarket-arb-bot/2.0 (contact@example.com)"}
        if CFG.noaa_token:
            h["token"] = CFG.noaa_token
        return h

    async def get_hourly_forecast(self, lat: float, lon: float, city: str) -> Optional[dict]:
        try:
            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=timeout) as s:
                # Step 1: get grid point
                url = f"{self.BASE}/points/{lat:.4f},{lon:.4f}"
                async with s.get(url, headers=self._headers()) as r:
                    if r.status != 200:
                        log.warning(f"NOAA points failed for {city}: HTTP {r.status}")
                        return None
                    data = await r.json()
                    forecast_url = data["properties"]["forecastHourly"]

                # Step 2: get hourly forecast
                async with s.get(forecast_url, headers=self._headers()) as r:
                    if r.status != 200:
                        log.warning(f"NOAA forecast failed for {city}: HTTP {r.status}")
                        return None
                    data = await r.json()
                    periods = data["properties"]["periods"]

                    now = datetime.now(timezone.utc)
                    tomorrow_date = (now + timedelta(days=1)).date()

                    tomorrow_temps = []
                    for p in periods:
                        try:
                            dt = datetime.fromisoformat(p["startTime"])
                        except (ValueError, KeyError):
                            continue
                        if dt.date() == tomorrow_date:
                            tomorrow_temps.append(p["temperature"])

                    # Fallback: remaining hours of today
                    if not tomorrow_temps:
                        today_date = now.date()
                        tomorrow_temps = [
                            p["temperature"] for p in periods[:12]
                            if datetime.fromisoformat(p["startTime"]).date() == today_date
                        ]

                    if not tomorrow_temps:
                        log.info(f"NOAA: No temperature data for {city}")
                        return None

                    max_temp = max(tomorrow_temps)
                    min_temp = min(tomorrow_temps)
                    avg_temp = sum(tomorrow_temps) / len(tomorrow_temps)

                    return {
                        "city": city,
                        "max_temp_f": max_temp,
                        "min_temp_f": min_temp,
                        "avg_temp_f": round(avg_temp, 1),
                        "forecast_count": len(tomorrow_temps),
                        "confidence": min(95, 70 + len(tomorrow_temps) * 3),
                    }
        except asyncio.TimeoutError:
            log.warning(f"NOAA timeout for {city}")
            return None
        except Exception as e:
            log.error(f"NOAA error for {city}: {e}")
            return None


# --- BTC FIYAT TOPLAYICI -----------------------------------------------------

class PriceAggregator:
    async def _fetch_json(self, session, url, name):
        try:
            async with session.get(url) as r:
                if r.status != 200:
                    log.warning(f"{name}: HTTP {r.status}")
                    return None
                return await r.json()
        except Exception as e:
            log.warning(f"{name}: {e}")
            return None

    async def get_binance_btc(self, session) -> Optional[float]:
        d = await self._fetch_json(session,
            "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT", "Binance")
        return float(d["price"]) if d and "price" in d else None

    async def get_okx_btc(self, session) -> Optional[float]:
        d = await self._fetch_json(session,
            "https://www.okx.com/api/v5/market/ticker?instId=BTC-USDT", "OKX")
        try:
            return float(d["data"][0]["last"]) if d else None
        except (KeyError, IndexError):
            return None

    async def get_coinbase_btc(self, session) -> Optional[float]:
        d = await self._fetch_json(session,
            "https://api.coinbase.com/v2/prices/BTC-USD/spot", "Coinbase")
        try:
            return float(d["data"]["amount"]) if d else None
        except (KeyError, TypeError):
            return None

    async def get_upbit_btc(self, session) -> Optional[float]:
        try:
            d = await self._fetch_json(session,
                "https://api.upbit.com/v1/ticker?markets=KRW-BTC", "Upbit")
            if not d:
                return None
            btc_krw = float(d[0]["trade_price"])
            fx = await self._fetch_json(session,
                "https://api.exchangerate-api.com/v4/latest/KRW", "FX")
            if not fx:
                return None
            krw_usd = fx["rates"]["USD"]
            return btc_krw * krw_usd
        except Exception as e:
            log.warning(f"Upbit: {e}")
            return None

    async def get_kraken_btc(self, session) -> Optional[float]:
        """Kraken BTC/USD price - 5th exchange"""
        d = await self._fetch_json(session,
            "https://api.kraken.com/0/public/Ticker?pair=XBTUSD", "Kraken")
        try:
            if d and "result" in d:
                pair_key = list(d["result"].keys())[0]
                return float(d["result"][pair_key]["c"][0])
            return None
        except (KeyError, IndexError, TypeError):
            return None

    async def get_all_prices(self) -> dict:
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(
            headers={"User-Agent": "arb-bot/2.0"},
            timeout=timeout
        ) as session:
            results = await asyncio.gather(
                self.get_binance_btc(session),
                self.get_okx_btc(session),
                self.get_coinbase_btc(session),
                self.get_upbit_btc(session),
                self.get_kraken_btc(session),
                return_exceptions=True
            )
            prices = {}
            for name, val in zip(["binance", "okx", "coinbase", "upbit", "kraken"], results):
                if isinstance(val, (int, float)) and val > 0:
                    prices[name] = val
            return prices


# --- POLYMARKET CLIENT -------------------------------------------------------

class PolymarketClient:
    GAMMA_API = "https://gamma-api.polymarket.com"
    CLOB_API = "https://clob.polymarket.com"

    async def _fetch_clob_markets(self, pages=3) -> list:
        """CLOB API sampling-markets - fetches real active markets and token IDs"""
        all_markets = []
        next_cursor = ""
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as s:
                for _ in range(pages):
                    params = {"limit": "100"}
                    if next_cursor:
                        params["next_cursor"] = next_cursor
                    async with s.get(f"{self.CLOB_API}/sampling-markets", params=params) as r:
                        if r.status != 200:
                            break
                        data = await r.json()
                        markets = data.get("data", [])
                        next_cursor = data.get("next_cursor", "")
                        all_markets.extend(markets)
                        if not next_cursor or next_cursor == "LQ==":
                            break
        except Exception as e:
            log.error(f"CLOB markets error: {e}")
        return all_markets

    async def get_token_price(self, token_id: str) -> float:
        """CLOB price endpoint - reference price (last trade/midpoint)"""
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as s:
                async with s.get(f"{self.CLOB_API}/price",
                                 params={"token_id": token_id, "side": "buy"}) as r:
                    if r.status != 200:
                        return 0
                    data = await r.json()
                    return float(data.get("price", 0))
        except Exception:
            return 0

    async def get_orderbook_price(self, token_id: str) -> dict:
        """Fetches real bid/ask/spread data from orderbook"""
        result = {"best_bid": 0, "best_ask": 0, "midpoint": 0, "spread": 1.0, "liquid": False}
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as s:
                async with s.get(f"{self.CLOB_API}/book",
                                 params={"token_id": token_id}) as r:
                    if r.status != 200:
                        return result
                    data = await r.json()

                    bids = data.get("bids", [])
                    asks = data.get("asks", [])

                    if not bids or not asks:
                        return result

                    # En iyi bid/ask (fiyata gore)
                    best_bid = max(float(b["price"]) for b in bids)
                    best_ask = min(float(a["price"]) for a in asks)

                    result["best_bid"] = best_bid
                    result["best_ask"] = best_ask
                    result["midpoint"] = round((best_bid + best_ask) / 2, 4)
                    result["spread"] = round(best_ask - best_bid, 4)

                    # Spread < %30 ise likit kabul et
                    result["liquid"] = result["spread"] < 0.30

                    return result
        except Exception:
            return result

    async def get_smart_price(self, token_id: str) -> tuple:
        """Calculates best price by combining orderbook + reference price.
        Returns: (price, spread, is_liquid)
        - Liquid market (spread<30%): uses orderbook midpoint
        - Illiquid market (spread>30%): uses /price reference price
        """
        # Fetch in parallel: orderbook + reference price
        ob_task = self.get_orderbook_price(token_id)
        ref_task = self.get_token_price(token_id)
        ob, ref_price = await asyncio.gather(ob_task, ref_task)

        if ob["liquid"] and ob["midpoint"] > 0:
            return (ob["midpoint"], ob["spread"], True)
        elif ref_price > 0:
            return (ref_price, ob["spread"], False)
        else:
            return (0, 1.0, False)

    async def get_token_prices_batch(self, token_ids: list) -> dict:
        """Fetch multiple token prices in parallel"""
        results = await asyncio.gather(*[
            self.get_token_price(tid) for tid in token_ids
        ], return_exceptions=True)
        prices = {}
        for tid, price in zip(token_ids, results):
            if isinstance(price, (int, float)) and price > 0:
                prices[tid] = price
        return prices

    async def get_all_active_markets(self, limit=200) -> list:
        """Tum aktif marketleri CLOB'dan cek, fiyatlari dahil et"""
        pages = max(1, limit // 100)
        markets = await self._fetch_clob_markets(pages=pages)
        return markets[:limit]

    async def get_weather_markets(self) -> list:
        """Sicaklik/hava durumu marketlerini filtrele"""
        markets = await self._fetch_clob_markets(pages=3)
        weather_words = ["temperature", "degrees", "celsius", "fahrenheit",
                        "weather", "snow", "rain", "hurricane", "storm"]
        return [m for m in markets
                if any(w in m.get("question", "").lower() for w in weather_words)]

    async def get_btc_markets(self) -> list:
        """BTC/Crypto marketlerini filtrele"""
        markets = await self._fetch_clob_markets(pages=3)
        crypto_words = ["bitcoin", "btc", "ethereum", "eth", "crypto", "solana"]
        return [m for m in markets
                if any(w in m.get("question", "").lower() for w in crypto_words)]

    async def get_market_orderbook(self, token_id: str) -> Optional[dict]:
        """Fetch order book for token"""
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as s:
                async with s.get(f"{self.CLOB_API}/book",
                                 params={"token_id": token_id}) as r:
                    if r.status != 200:
                        return None
                    return await r.json()
        except Exception:
            return None

    async def place_order(self, token_id: str, side: str, amount_usd: float,
                          price: float) -> bool:
        if CFG.simulation_mode:
            log_trade("BUY" if side == "YES" else "SELL", token_id[:16],
                      amount_usd, price, f"sim-{side}", sim=True)
            return True

        if not CFG.poly_private_key:
            log.error("POLY_PRIVATE_KEY eksik!")
            return False

        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import MarketOrderArgs, OrderType
            from py_clob_client.constants import POLYGON

            client = ClobClient(
                host=self.CLOB_API,
                key=CFG.poly_private_key,
                chain_id=POLYGON,
                funder=CFG.poly_address,
            )
            order_args = MarketOrderArgs(token_id=token_id, amount=amount_usd)
            signed_order = client.create_market_order(order_args)
            resp = client.post_order(signed_order, OrderType.FOK)

            if resp and resp.get("status") == "matched":
                log_trade("BUY", token_id[:16], amount_usd, price, "arb-signal")
                return True
            else:
                log.warning(f"Order response: {resp}")
                return False
        except ImportError:
            log.error("py-clob-client kurulu degil: pip install py-clob-client")
            return False
        except Exception as e:
            log.error(f"Order error: {e}")
            return False


# --- WEATHER ARB ENGINE -------------------------------------------------------

class WeatherArbEngine:
    def __init__(self):
        self.noaa = NOAAClient()
        self.poly = PolymarketClient()

    def parse_temp_range(self, question: str) -> Optional[tuple]:
        match = re.search(r"(\d+)\s*[-\u2013]\s*(\d+)\s*[°]?[Ff]", question)
        if match:
            return int(match.group(1)), int(match.group(2))
        # Tek deger: "above 80°F"
        match = re.search(r"(?:above|over|exceed)\s*(\d+)\s*[°]?[Ff]", question, re.I)
        if match:
            return int(match.group(1)), 150  # ust sinir yok
        match = re.search(r"(?:below|under)\s*(\d+)\s*[°]?[Ff]", question, re.I)
        if match:
            return -50, int(match.group(1))  # alt sinir yok
        return None

    def noaa_probability_for_range(self, noaa_temp: float, low: float, high: float,
                                    confidence: int) -> float:
        std = 3.0
        mean = noaa_temp
        def phi(x):
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))
        prob = phi((high - mean) / std) - phi((low - mean) / std)
        adjusted = prob * (confidence / 100) + 0.5 * (1 - confidence / 100)
        return min(0.97, max(0.03, adjusted))

    def parse_global_temp_range(self, question: str) -> Optional[tuple]:
        """Global temperature marketleri: 'between 1.20C and 1.24C' -> (1.20, 1.24)"""
        # "between X and Y" pattern
        match = re.search(r"between\s+([\d.]+).*?and\s+([\d.]+)", question, re.I)
        if match:
            return float(match.group(1)), float(match.group(2))
        # "more than X" pattern
        match = re.search(r"more than\s+([\d.]+)", question, re.I)
        if match:
            return float(match.group(1)), 5.0
        # "less than X" pattern
        match = re.search(r"less than\s+([\d.]+)", question, re.I)
        if match:
            return -5.0, float(match.group(1))
        return None

    async def find_opportunities(self) -> list:
        opportunities = []

        # Fetch NOAA data (for US cities)
        forecasts = await asyncio.gather(*[
            self.noaa.get_hourly_forecast(lat, lon, city)
            for lat, lon, city in CFG.cities
        ])
        forecast_map = {f["city"]: f for f in forecasts if f}
        if forecast_map:
            log.info(f"NOAA: {len(forecast_map)} cities: " +
                     ", ".join(f"{c}={d['max_temp_f']}F" for c, d in forecast_map.items()))

        # Fetch Polymarket weather/temperature markets (CLOB API)
        markets = await self.poly.get_weather_markets()
        log.info(f"Weather: {len(markets)} markets found")

        for market in markets:
            question = market.get("question", "")
            tokens = market.get("tokens", [])
            if not tokens:
                continue

            yes_token = next((t for t in tokens if t.get("outcome", "").upper() == "YES"), None)
            if not yes_token:
                continue

            token_id = yes_token.get("token_id", "")
            if not token_id:
                continue

            # Fetch real price from CLOB
            market_price = await self.poly.get_token_price(token_id)
            if market_price <= 0:
                continue

            # Is this a global temperature market?
            if "global temperature" in question.lower() or "temperature increase" in question.lower():
                temp_range = self.parse_global_temp_range(question)
                if not temp_range:
                    continue
                low, high = temp_range
                # Simple estimate: based on current trend and other market prices
                # External data source needed for a more sophisticated model
                noaa_prob = market_price  # Use market price as starting point
                edge = 0  # External data needed for these market types
            else:
                # US city temperature market
                matched_city = None
                for city in forecast_map:
                    if city.lower() in question.lower():
                        matched_city = city
                        break
                if not matched_city:
                    continue

                forecast = forecast_map[matched_city]
                temp_range = self.parse_temp_range(question)
                if not temp_range:
                    continue

                low, high = temp_range
                noaa_prob = self.noaa_probability_for_range(
                    forecast["max_temp_f"], low, high, forecast["confidence"])
                edge = noaa_prob - market_price

            if abs(edge) >= (CFG.min_edge_pct / 100):
                no_token = next((t for t in tokens if t.get("outcome", "").upper() == "NO"), None)
                opportunities.append({
                    "type": "weather",
                    "market_id": market.get("condition_id", "")[:20],
                    "market": question[:80],
                    "noaa_prob": round(noaa_prob, 4),
                    "market_price": round(market_price, 4),
                    "edge": round(edge, 4),
                    "side": "YES" if edge > 0 else "NO",
                    "token_id": token_id,
                    "no_token_id": no_token.get("token_id", "") if no_token else "",
                    "liquidity": 0,
                    "condition_id": market.get("condition_id", ""),
                    "end_date": market.get("end_date_iso", ""),
                })

        opportunities.sort(key=lambda x: abs(x["edge"]), reverse=True)
        return opportunities


# --- BTC ARB ENGINE -----------------------------------------------------------

class BTCArbEngine:
    def __init__(self):
        self.aggregator = PriceAggregator()
        self.poly = PolymarketClient()

    def extract_btc_target(self, question: str) -> Optional[float]:
        # Million expressions like "$1m", "$1.5m", "$1M"
        match = re.search(r"\$\s*([\d.]+)\s*[mM]", question)
        if match:
            return float(match.group(1)) * 1_000_000
        # Billion expressions like "$1b", "$1B"
        match = re.search(r"\$\s*([\d.]+)\s*[bB]", question)
        if match:
            return float(match.group(1)) * 1_000_000_000
        # "100k", "100K", "$100k" gibi bin ifadeleri
        # \b ile kelime siniri: "2026 Kansas" gibi false-positive onlenir
        match = re.search(r"\$?\s*(\d+(?:\.\d+)?)\s*[kK]\b", question)
        if match:
            val = float(match.group(1)) * 1000
            if val < 1000:  # "$0.5k" gibi anlamsiz degerleri reddet
                return None
            return val
        # "$100,000" veya "$95000" gibi tam sayilar
        match = re.search(r"\$\s*([\d,]+(?:\.\d+)?)", question)
        if match:
            val = float(match.group(1).replace(",", ""))
            # $1, $2 gibi cok kucuk degerleri BTC hedefi olarak kabul etme
            if val < 100:
                return None
            return val
        return None

    async def find_opportunities(self) -> list:
        opportunities = []
        prices = await self.aggregator.get_all_prices()
        if not prices:
            log.warning("Borsa: Fiyat alinamadi")
            return []

        avg_btc = sum(prices.values()) / len(prices)
        log.info(f"BTC: " + " | ".join(f"{k}=${v:,.0f}" for k, v in prices.items()))
        log.info(f"BTC Average: ${avg_btc:,.0f}")

        markets = await self.poly.get_btc_markets()
        log.info(f"BTC: {len(markets)} markets found")

        for market in markets:
            question = market.get("question", "")

            target = self.extract_btc_target(question)
            if not target:
                log.info(f"BTC SKIP (target not found): {question[:60]}")
                continue

            # Is target in reasonable range? ($1000 - $10M)
            if target < 1000 or target > 10_000_000:
                log.info(f"BTC SKIP (target out of range ${target:,.0f}): {question[:60]}")
                continue

            tokens = market.get("tokens", [])
            yes_token = next((t for t in tokens if t.get("outcome", "").upper() == "YES"), None)
            if not yes_token:
                continue

            token_id = yes_token.get("token_id", "")
            if not token_id:
                continue

            # Fetch real price from CLOB (orderbook + reference)
            market_price, spread, is_liquid = await self.poly.get_smart_price(token_id)
            if market_price <= 0.01 or market_price >= 0.99:
                continue  # Extreme prices -> illiquid or already resolved

            liq_tag = "LIQ" if is_liquid else f"ILLIQ(spread={spread:.0%})"

            is_exceed = any(w in question.lower() for w in
                          ["exceed", "above", "over", "reach", "hit", "higher"])
            is_below = any(w in question.lower() for w in
                          ["below", "under", "drop", "fall", "lower"])

            # Time factor based on end date
            end_date_str = market.get("end_date_iso", "")
            time_factor = 1.0
            if end_date_str:
                try:
                    end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                    days_until = (end_dt - datetime.now(timezone.utc)).days
                    if days_until > 180:
                        time_factor = 0.7   # 6+ months: high uncertainty
                    elif days_until > 90:
                        time_factor = 0.8   # 3-6 months
                    elif days_until > 30:
                        time_factor = 0.9   # 1-3 months
                    elif days_until > 7:
                        time_factor = 0.95  # 1-4 weeks
                    # Within 7 days: time factor 1.0
                except (ValueError, TypeError):
                    pass

            if is_exceed:
                ratio = avg_btc / target
                if ratio > 1.05:
                    real_prob = 0.90
                elif ratio > 1.02:
                    real_prob = 0.78
                elif ratio > 1.00:
                    real_prob = 0.62
                elif ratio > 0.98:
                    real_prob = 0.45
                elif ratio > 0.95:
                    real_prob = 0.30
                elif ratio > 0.90:
                    real_prob = 0.18
                elif ratio > 0.80:
                    real_prob = 0.10
                elif ratio > 0.50:
                    real_prob = 0.05
                else:
                    real_prob = 0.02  # Hedef cok uzak (orn: $1M)
                real_prob *= time_factor
            elif is_below:
                ratio = target / avg_btc
                if ratio > 1.05:
                    real_prob = 0.90
                elif ratio > 1.02:
                    real_prob = 0.78
                elif ratio > 1.00:
                    real_prob = 0.62
                elif ratio > 0.98:
                    real_prob = 0.45
                elif ratio > 0.95:
                    real_prob = 0.30
                else:
                    real_prob = 0.12
                real_prob *= time_factor
            else:
                # Range markets like "Will BTC be between X and Y"
                # Skip for now
                continue

            # Clamp probability
            real_prob = min(0.95, max(0.03, real_prob))

            edge = real_prob - market_price

            if len(prices) >= 2:
                pv = list(prices.values())
                spread_pct = (max(pv) - min(pv)) / avg_btc
            else:
                spread_pct = 0

            log.info(f"BTC Analiz: {question[:50]} | "
                     f"Hedef: ${target:,.0f} | Oran: {avg_btc/target:.3f} | "
                     f"Prob: {real_prob:.2%} vs Market: {market_price:.2%} | "
                     f"Edge: {edge:.2%} | {liq_tag}")

            if abs(edge) >= (CFG.min_edge_pct / 100):
                opportunities.append({
                    "type": "btc",
                    "market_id": market.get("condition_id", "")[:20],
                    "market": question[:80],
                    "btc_avg": round(avg_btc, 0),
                    "btc_prices": {k: round(v, 0) for k, v in prices.items()},
                    "target": target,
                    "real_prob": round(real_prob, 4),
                    "market_price": round(market_price, 4),
                    "edge": round(edge, 4),
                    "spread_pct": round(spread_pct * 100, 2),
                    "side": "YES" if edge > 0 else "NO",
                    "token_id": yes_token.get("token_id"),
                    "no_token_id": next((t.get("token_id", "") for t in tokens if t.get("outcome", "").upper() == "NO"), ""),
                    "liquidity": 0,
                    "ob_spread": round(spread, 4),
                    "is_liquid": is_liquid,
                    "condition_id": market.get("condition_id", ""),
                    "end_date": market.get("end_date_iso", ""),
                })

        opportunities.sort(key=lambda x: abs(x["edge"]), reverse=True)
        return opportunities


# --- XAU/XAG METALS ARB ENGINE -----------------------------------------------

class MetalsArbEngine:
    """Real data analysis for XAU/XAG Polymarket bets"""

    def __init__(self):
        self.poly = PolymarketClient()
        self._xau_engine = None  # Lazy import

    def _get_engine(self):
        if self._xau_engine is None:
            try:
                from xau_xag_engine import XAUXAGEngine
                self._xau_engine = XAUXAGEngine()
            except ImportError:
                log.warning("xau_xag_engine module not found")
                return None
        return self._xau_engine

    def extract_metal_target(self, question: str):
        """Extract target price from market question (e.g., '$3,500' -> 3500.0)"""
        import re
        match = re.search(r"\$\s*([\d,]+(?:\.\d+)?)", question)
        if match:
            val = float(match.group(1).replace(",", ""))
            if val > 0:
                return val
        return None

    async def find_opportunities(self) -> list:
        engine = self._get_engine()
        if not engine:
            return []

        opportunities = []

        # Find XAU/XAG markets from Polymarket
        markets = await self.poly._fetch_clob_markets(pages=3)

        # More specific filter: gold/silver futures, XAU, XAG, ounce
        # Simple "gold" or "silver" alone creates too many false positives
        # NOTE: Don't use "ounce" alone! "announce" contains "ounce" -> false positive!
        metal_patterns = [
            "gold (gc)", "silver (si)", "xau", "xag",
            "gold price", "silver price", "gold futures", "silver futures",
            "precious metal", "per ounce", "troy ounce", "/oz",
            "gold spot", "silver spot",
            "gold hit", "gold reach", "gold above", "gold below",
            "silver hit", "silver reach", "silver above", "silver below",
        ]

        metal_markets = []
        for m in markets:
            q = m.get("question", "").lower()
            # Pattern matching
            if any(p in q for p in metal_patterns):
                metal_markets.append(m)
                continue
            # Markets containing "gold" or "silver" related to finance/price
            if ("gold" in q or "silver" in q) and any(w in q for w in
                ["price", "settle", "hit", "above", "below", "reach", "$", "futures"]):
                metal_markets.append(m)

        log.info(f"Metals: {len(metal_markets)} markets found (scanned {len(markets)} total)")
        if metal_markets:
            log.info(f"  First market: {metal_markets[0].get('question', '')[:70]}")

        # Analyze max 10 markets for performance
        _metal_price_cache = {}
        for market in metal_markets[:10]:
            question = market.get("question", "")
            tokens = market.get("tokens", [])
            if not tokens:
                continue

            yes_token = next((t for t in tokens if t.get("outcome", "").upper() == "YES"), None)
            no_token = next((t for t in tokens if t.get("outcome", "").upper() == "NO"), None)
            if not yes_token:
                continue

            token_id = yes_token.get("token_id", "")
            if not token_id:
                continue

            # Real price from CLOB (orderbook + reference)
            market_price, spread, is_liquid = await self.poly.get_smart_price(token_id)
            if market_price <= 0:
                continue

            # Determine metal type and direction
            q_lower = question.lower()
            metal = "XAU" if any(w in q_lower for w in ["gold", "xau"]) else "XAG"
            is_up = any(w in q_lower for w in ["above", "over", "exceed", "higher", "reach", "hit", "rise"])
            is_down = any(w in q_lower for w in ["below", "under", "fall", "drop", "lower", "decline"])
            direction = "UP" if is_up else "DOWN" if is_down else "UP"

            try:
                decision = await engine.analyze(metal, market_price, direction)

                # Distance-to-target discount: if move required is unrealistic,
                # dramatically reduce probability regardless of macro signals
                target_price = self.extract_metal_target(question)
                current_metal_price = _metal_price_cache.get(metal)
                if current_metal_price is None:
                    try:
                        pd = await engine.price_collector.get_price(metal)
                        _metal_price_cache[metal] = pd.price if pd else None
                        current_metal_price = _metal_price_cache[metal]
                    except Exception:
                        _metal_price_cache[metal] = None

                if target_price and current_metal_price and current_metal_price > 0:
                    if direction == "UP":
                        pct_move = (target_price - current_metal_price) / current_metal_price
                    else:
                        pct_move = (current_metal_price - target_price) / current_metal_price

                    # Apply discount based on move magnitude
                    if pct_move > 0.50:
                        dist_factor = 0.02
                    elif pct_move > 0.20:
                        dist_factor = max(0.02, 0.30 - (pct_move - 0.20) * 0.93)
                    elif pct_move > 0.05:
                        dist_factor = max(0.30, 1.0 - (pct_move - 0.05) * 4.67)
                    else:
                        dist_factor = 1.0  # Within 5%, trust the engine

                    adj_prob = decision.probability * dist_factor
                    adj_edge = adj_prob - market_price
                    adj_conf = decision.confidence * dist_factor
                    adj_side = "YES" if adj_edge > 0.10 else ("NO" if adj_edge < -0.10 else "SKIP")

                    log.info(
                        f"Metals distance: {metal} current=${current_metal_price:,.1f} "
                        f"target=${target_price:,.1f} move={pct_move*100:+.1f}% "
                        f"factor={dist_factor:.2f} | "
                        f"prob {decision.probability:.2%}->{adj_prob:.2%} "
                        f"edge {decision.edge:+.2%}->{adj_edge:+.2%}"
                    )

                    # Override decision values
                    from xau_xag_engine import MetalDecision
                    decision = MetalDecision(
                        metal=decision.metal, direction=decision.direction,
                        probability=adj_prob, polymarket_price=decision.polymarket_price,
                        edge=adj_edge, confidence=adj_conf,
                        order_side=adj_side,
                        kelly_size_pct=decision.kelly_size_pct * dist_factor,
                        factors=decision.factors,
                        summary=decision.summary + f" [dist_adj={dist_factor:.2f}]",
                    )

                if decision.order_side != "SKIP" and abs(decision.edge) >= (CFG.min_edge_pct / 100):
                    opportunities.append({
                        "type": "metals",
                        "market_id": market.get("condition_id", "")[:20],
                        "market": question[:80],
                        "metal": metal,
                        "direction": direction,
                        "real_prob": round(decision.probability, 4),
                        "market_price": round(market_price, 4),
                        "edge": round(decision.edge, 4),
                        "confidence": round(decision.confidence, 4),
                        "side": decision.order_side,
                        "token_id": token_id,
                        "no_token_id": no_token.get("token_id", "") if no_token else "",
                        "liquidity": 0,
                        "ob_spread": round(spread, 4),
                        "is_liquid": is_liquid,
                        "condition_id": market.get("condition_id", ""),
                        "end_date": market.get("end_date_iso", ""),
                        "factors": [{"name": f.name, "score": round(f.score, 3),
                                    "weight": f.weight, "desc": f.description}
                                   for f in decision.factors],
                    })
            except Exception as e:
                log.warning(f"Metals analyze error for {question[:40]}: {e}")

        opportunities.sort(key=lambda x: abs(x["edge"]), reverse=True)
        return opportunities


# --- GENEL MARKET TARAYICI ---------------------------------------------------

class GeneralMarketScanner:
    """Scans all active Polymarket markets, reports extremely high/low priced ones"""

    def __init__(self):
        self.poly = PolymarketClient()

    async def find_mispriced_markets(self) -> list:
        """Find high-liquidity markets priced below 0.05 or above 0.95"""
        opportunities = []
        markets = await self.poly.get_all_active_markets(limit=200)
        log.info(f"General Scan: {len(markets)} active markets found")

        for market in markets:
            question = market.get("question", "")
            liquidity = float(market.get("liquidity", 0) or 0)
            if liquidity < CFG.min_liquidity:
                continue

            tokens = market.get("tokens", [])
            if not tokens:
                continue

            yes_token = next((t for t in tokens if t.get("outcome", "").upper() == "YES"), None)
            if not yes_token:
                continue

            price = float(yes_token.get("price", 0.5) or 0.5)

            # Interesting categories
            tags = [t.get("slug", "") if isinstance(t, dict) else str(t)
                    for t in market.get("tags", [])]
            category = market.get("groupItemTitle", market.get("category", "other"))

            opportunities.append({
                "type": "general",
                "market_id": market.get("id", ""),
                "market": question[:80],
                "market_price": round(price, 4),
                "liquidity": liquidity,
                "volume": float(market.get("volume", 0) or 0),
                "category": str(category)[:30],
                "tags": tags[:5],
                "token_id": yes_token.get("token_id"),
                "end_date": market.get("endDate", ""),
                "condition_id": market.get("conditionId", ""),
            })

        opportunities.sort(key=lambda x: x["liquidity"], reverse=True)
        return opportunities


# --- SIMULATION ENGINE --------------------------------------------------------

class SimulationEngine:
    """Paper trading / simulation engine"""

    def __init__(self, initial_balance: float = 10000.0):
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.positions = []  # Open positions
        self.closed_trades = []  # Closed trades
        self.trade_history = []  # All trade history
        self._load_state()

    def _state_file(self):
        return DATA_DIR / "sim_state.json"

    def _load_state(self):
        sf = self._state_file()
        if sf.exists():
            try:
                with open(sf, "r") as f:
                    state = json.load(f)
                self.balance = state.get("balance", self.initial_balance)
                self.positions = state.get("positions", [])
                self.closed_trades = state.get("closed_trades", [])
                self.trade_history = state.get("trade_history", [])
                log.info(f"SIM: State loaded - Balance: ${self.balance:.2f}, "
                         f"{len(self.positions)} open positions")
            except Exception as e:
                log.warning(f"SIM state load error: {e}")

    def save_state(self):
        state = {
            "balance": self.balance,
            "positions": self.positions,
            "closed_trades": self.closed_trades,
            "trade_history": self.trade_history[-1000:],
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        with open(self._state_file(), "w") as f:
            json.dump(state, f, indent=2, default=str)

    def place_bet(self, market_id: str, market_name: str, side: str,
                  amount: float, price: float, edge: float, market_type: str,
                  end_date: str = "", token_id: str = "",
                  no_token_id: str = "") -> dict:
        if amount > self.balance:
            return {"error": f"Yetersiz bakiye: ${self.balance:.2f}"}

        position = {
            "id": f"sim_{len(self.trade_history)+1:04d}",
            "ts": datetime.now(timezone.utc).isoformat(),
            "market_id": market_id,
            "market": market_name,
            "side": side,
            "amount": round(amount, 2),
            "entry_price": round(price, 4),
            "edge_at_entry": round(edge, 4),
            "type": market_type,
            "end_date": end_date,
            "token_id": token_id,
            "no_token_id": no_token_id,
            "status": "open",
            "pnl": 0,
        }

        self.balance -= amount
        self.positions.append(position)
        self.trade_history.append(position)
        self.save_state()

        log_trade(f"SIM-{side}", market_name[:40], amount, price,
                  f"edge={edge*100:.1f}%", sim=True)

        return position

    def get_portfolio_summary(self) -> dict:
        total_invested = sum(p["amount"] for p in self.positions)
        realized_pnl = sum(t.get("pnl", 0) for t in self.closed_trades)
        win_count = sum(1 for t in self.closed_trades if t.get("pnl", 0) > 0)
        loss_count = sum(1 for t in self.closed_trades if t.get("pnl", 0) < 0)
        total_closed = len(self.closed_trades)

        return {
            "balance": round(self.balance, 2),
            "initial_balance": self.initial_balance,
            "total_invested": round(total_invested, 2),
            "portfolio_value": round(self.balance + total_invested, 2),
            "realized_pnl": round(realized_pnl, 2),
            "unrealized_pnl": 0.0,  # filled by web_server
            "total_pnl": round(realized_pnl, 2),
            "pnl_pct": round((realized_pnl / self.initial_balance) * 100, 2) if self.initial_balance > 0 else 0,
            "open_positions": len(self.positions),
            "closed_trades": total_closed,
            "win_count": win_count,
            "loss_count": loss_count,
            "win_rate": round((win_count / total_closed) * 100, 1) if total_closed > 0 else 0,
            "total_trades": len(self.trade_history),
        }

    def resolve_position(self, position_id: str, outcome: str) -> dict:
        """Resolve position (YES/NO)"""
        pos = None
        for i, p in enumerate(self.positions):
            if p["id"] == position_id:
                pos = self.positions.pop(i)
                break
        if not pos:
            return {"error": "Position not found"}

        won = (pos["side"] == outcome)
        if won:
            payout = pos["amount"] / pos["entry_price"]
            pnl = payout - pos["amount"]
        else:
            payout = 0
            pnl = -pos["amount"]

        pos["status"] = "won" if won else "lost"
        pos["pnl"] = round(pnl, 2)
        pos["payout"] = round(payout, 2)
        pos["resolved_at"] = datetime.now(timezone.utc).isoformat()
        pos["outcome"] = outcome

        self.balance += payout
        self.closed_trades.append(pos)
        self.save_state()
        return pos


# --- ANA BOT -----------------------------------------------------------------

class ArbBot:
    def __init__(self):
        self.weather_engine = WeatherArbEngine()
        self.btc_engine = BTCArbEngine()
        self.scanner = GeneralMarketScanner()
        self.poly = PolymarketClient()
        self.sim = SimulationEngine(CFG.initial_balance) if CFG.simulation_mode else None
        self.daily_spent = 0.0
        self.last_reset = datetime.now(timezone.utc).date()
        self.last_scan_results = {}  # Last scan results for Web API

    def reset_daily_if_needed(self):
        today = datetime.now(timezone.utc).date()
        if today != self.last_reset:
            self.daily_spent = 0.0
            self.last_reset = today
            log.info("Daily limit reset")

    def calculate_order_size(self, edge: float, liquidity: float) -> float:
        kelly_fraction = min(abs(edge), 0.3)
        size = CFG.max_order_usd * kelly_fraction
        size = min(size, liquidity * 0.02)
        size = min(size, CFG.max_order_usd)
        size = max(size, 1.0)
        return round(size, 2)

    async def execute_opportunity(self, opp: dict) -> bool:
        self.reset_daily_if_needed()
        if CFG.max_daily_usd > 0 and self.daily_spent >= CFG.max_daily_usd:
            log.warning(f"Gunluk limit dolu: ${self.daily_spent:.2f}/{CFG.max_daily_usd}")
            return False

        order_size = self.calculate_order_size(opp["edge"], opp["liquidity"])
        if CFG.max_daily_usd > 0:
            order_size = min(order_size, CFG.max_daily_usd - self.daily_spent)

        if self.sim:
            result = self.sim.place_bet(
                market_id=opp.get("market_id", ""),
                market_name=opp["market"],
                side=opp["side"],
                amount=order_size,
                price=opp["market_price"],
                edge=opp["edge"],
                market_type=opp["type"],
                end_date=opp.get("end_date", ""),
                token_id=opp.get("token_id", ""),
                no_token_id=opp.get("no_token_id", ""),
            )
            if "error" not in result:
                self.daily_spent += order_size
                return True
            else:
                log.warning(f"SIM hata: {result['error']}")
                return False
        else:
            price = opp["market_price"] if opp["side"] == "YES" else 1 - opp["market_price"]
            success = await self.poly.place_order(
                token_id=opp.get("token_id", ""),
                side=opp["side"],
                amount_usd=order_size,
                price=price,
            )
            if success:
                self.daily_spent += order_size
            return success

    async def run_cycle(self) -> dict:
        log.info("=" * 60)
        log.info(f"New cycle: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        weather_opps, btc_opps, general_markets = await asyncio.gather(
            self.weather_engine.find_opportunities(),
            self.btc_engine.find_opportunities(),
            self.scanner.find_mispriced_markets(),
        )

        all_opps = weather_opps + btc_opps
        all_opps.sort(key=lambda x: abs(x["edge"]), reverse=True)

        result = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "weather_opportunities": weather_opps,
            "btc_opportunities": btc_opps,
            "general_markets": general_markets[:50],
            "total_opportunities": len(all_opps),
            "portfolio": self.sim.get_portfolio_summary() if self.sim else {},
        }
        self.last_scan_results = result

        # Save results
        with open(DATA_DIR / "last_scan.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

        if not all_opps:
            log.info("No opportunities found this cycle")
            log.info(f"General scan listed {len(general_markets)} markets")
            return result

        log.info(f"{len(all_opps)} opportunities found "
                 f"({len(weather_opps)} weather, {len(btc_opps)} BTC)")

        for opp in all_opps[:10]:
            emoji = "W" if opp["type"] == "weather" else "B"
            log.info(f"[{emoji}] {opp['market'][:50]} | "
                     f"Edge: {opp['edge']*100:+.1f}% | "
                     f"{opp['side']} @ {opp['market_price']:.3f}")

        # Auto bet in simulation mode
        if self.sim and CFG.max_daily_usd > 0:
            executed = 0
            for opp in all_opps[:3]:
                if executed >= 3:
                    break
                success = await self.execute_opportunity(opp)
                if success:
                    executed += 1
                await asyncio.sleep(1)

        return result

    async def run(self):
        mode = "SIMULATION" if CFG.simulation_mode else "LIVE"
        log.info(f"Polymarket Arbitrage Bot v2.0 starting [{mode}]")
        log.info(f"Min edge: {CFG.min_edge_pct}% | Max order: ${CFG.max_order_usd} | "
                 f"Daily limit: ${CFG.max_daily_usd}")

        if not CFG.simulation_mode and not CFG.poly_private_key:
            log.error("POLY_PRIVATE_KEY missing! Check your .env file.")
            log.info("Switching to simulation mode...")
            CFG.simulation_mode = True
            self.sim = SimulationEngine(CFG.initial_balance)

        while True:
            try:
                await self.run_cycle()
            except KeyboardInterrupt:
                log.info("Bot stopped")
                break
            except Exception as e:
                log.error(f"Cycle error: {e}", exc_info=True)

            log.info(f"Next cycle in {CFG.loop_interval_sec}s...")
            await asyncio.sleep(CFG.loop_interval_sec)


# --- AUTO TRADING SYSTEM ------------------------------------------------------

class AutoTrader:
    """
    Fully automated trading system:
    - Periodically scans BTC (5 exchanges) and Weather (NOAA) opportunities
    - Opens auto simulation bets when edge is found
    - Auto-resolves open positions based on real data
    - Tracks all statistics
    """

    def __init__(self, sim_engine: SimulationEngine):
        self.sim = sim_engine
        self.weather_engine = WeatherArbEngine()
        self.btc_engine = BTCArbEngine()
        self.metals_engine = MetalsArbEngine()
        self.poly = PolymarketClient()
        self.price_agg = PriceAggregator()

        # Durum bilgileri
        self.running = False
        self.scan_count = 0
        self.auto_bets_placed = 0
        self.auto_resolves = 0
        self.last_scan_time = None
        self.next_scan_time = None
        self.last_scan_result = {}
        self.daily_spent = 0.0
        self.last_reset_date = datetime.now(timezone.utc).date()
        self.scan_interval = CFG.loop_interval_sec  # seconds
        self.max_bets_per_cycle = 3
        self.min_edge_auto = CFG.min_edge_pct / 100  # Minimum edge for auto betting
        self.auto_resolve_enabled = True

        # Statistics
        self.stats = {
            "total_scans": 0,
            "total_bets_placed": 0,
            "total_resolved": 0,
            "weather_bets": 0,
            "btc_bets": 0,
            "last_btc_prices": {},
            "last_weather_data": [],
            "last_opportunities": [],
            "cycle_log": [],  # Last 50 cycle logs
            "started_at": None,
            "errors": 0,
        }
        self._task = None

    def _reset_daily(self):
        today = datetime.now(timezone.utc).date()
        if today != self.last_reset_date:
            self.daily_spent = 0.0
            self.last_reset_date = today
            log.info("[AUTO] Daily limit reset")

    def get_status(self) -> dict:
        """Current status information"""
        return {
            "running": self.running,
            "scan_count": self.scan_count,
            "auto_bets_placed": self.auto_bets_placed,
            "auto_resolves": self.auto_resolves,
            "last_scan_time": self.last_scan_time,
            "next_scan_time": self.next_scan_time,
            "scan_interval": self.scan_interval,
            "min_edge_pct": round(self.min_edge_auto * 100, 1),
            "max_bets_per_cycle": self.max_bets_per_cycle,
            "daily_spent": round(self.daily_spent, 2),
            "daily_limit": CFG.max_daily_usd,
            "auto_resolve_enabled": self.auto_resolve_enabled,
            "stats": self.stats,
            "portfolio": self.sim.get_portfolio_summary(),
        }

    def _log_cycle(self, msg: str, level: str = "info"):
        """Save cycle log entry"""
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "msg": msg,
            "level": level,
        }
        self.stats["cycle_log"].append(entry)
        # Keep last 100 logs
        if len(self.stats["cycle_log"]) > 100:
            self.stats["cycle_log"] = self.stats["cycle_log"][-100:]

    def _get_tier_max(self, balance: float) -> float:
        """Tiered max bet limit based on balance"""
        if balance < 100:
            return 2.0       # $0-100: max $2
        elif balance < 200:
            return 5.0       # $100-200: max $5
        elif balance < 500:
            return 10.0      # $200-500: max $10
        elif balance < 1000:
            return 20.0      # $500-1000: max $20
        elif balance < 2000:
            return 50.0      # $1000-2000: max $50
        elif balance < 5000:
            return 100.0     # $2000-5000: max $100
        else:
            return 200.0     # $5000+: max $200

    def calculate_bet_size(self, edge: float, real_prob: float = 0.5) -> float:
        """
        Dynamic + tiered bet sizing based on balance:
        Tier limits:
          $0-100    -> max $2
          $100-200  -> max $5
          $200-500  -> max $10
          $500-1000 -> max $20
          $1000-2000-> max $50
          $2000+    -> max $100-200
        Higher edge and probability approach the upper limit within the tier.
        """
        balance = self.sim.balance
        abs_edge = abs(edge)
        tier_max = self._get_tier_max(balance)

        # Edge-based ratio: 5% edge -> 25% of tier, 30%+ edge -> 100% of tier
        edge_ratio = min(abs_edge * 3.3, 1.0)   # 0.05*3.3=0.165 -> 0.30*3.3=1.0
        edge_ratio = max(edge_ratio, 0.15)        # minimum 15% of tier

        # Probability multiplier: approach tier limit when certainty is high
        prob_mult = 1.0
        if real_prob >= 0.90:
            prob_mult = 1.5   # cok emin
        elif real_prob >= 0.80:
            prob_mult = 1.3   # emin
        elif real_prob >= 0.70:
            prob_mult = 1.15  # biraz emin

        size = tier_max * edge_ratio * prob_mult

        # Kademe limitini asma
        size = min(size, tier_max)

        # MIN: $0.50
        size = max(size, 0.50)

        # Gunluk limit (0 = limitsiz)
        if CFG.max_daily_usd > 0:
            remaining = CFG.max_daily_usd - self.daily_spent
            if remaining <= 0:
                return 0
            size = min(size, remaining)

        # Son guvenlik: bakiyeden fazla olamaz
        size = min(size, balance - 1.0)  # en az $1 bakiye kalsin

        if size < 0.50:
            return 0

        return round(size, 2)

    async def auto_resolve_positions(self):
        """
        Auto-resolve open positions based on real data.

        IMPORTANT RULES:
        - BTC positions: Only resolve when market_price updated AND
          real price has definitively reached/missed the target
        - Weather positions: Only resolve when weather event has occurred
        - Future prediction markets (far end_date) are NOT resolved immediately
        - Minimum wait time: 10 minutes (for short-term markets)
        """
        if not self.auto_resolve_enabled:
            return 0

        resolved = 0
        now = datetime.now(timezone.utc)
        positions_copy = list(self.sim.positions)

        for pos in positions_copy:
            try:
                pos_type = pos.get("type", "")
                market_name = pos.get("market", "").lower()
                entry_time = pos.get("ts", "")

                # Position must be open at least 10 minutes
                if entry_time:
                    try:
                        entry_dt = datetime.fromisoformat(entry_time)
                        age_minutes = (now - entry_dt).total_seconds() / 60
                        if age_minutes < 10:
                            continue  # Cok yeni, atla
                    except (ValueError, TypeError):
                        age_minutes = 0

                # Market end_date kontrolu - end_date cok uzaksa resolve etme
                end_date_str = pos.get("end_date", "")
                if end_date_str:
                    try:
                        end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                        days_until_end = (end_dt - now).days
                        # If end date is more than 7 days away, resolve by CLOB price change
                        if days_until_end > 7:
                            # Long-term market: resolve only by CLOB price change
                            await self._resolve_by_price_change(pos, age_minutes)
                            if pos["id"] not in [p["id"] for p in self.sim.positions]:
                                resolved += 1
                                self.auto_resolves += 1
                            continue
                    except (ValueError, TypeError):
                        pass

                outcome = None

                if pos_type == "btc":
                    # BTC: Real price target check
                    prices = self.stats.get("last_btc_prices", {})
                    if not prices:
                        prices = await self.price_agg.get_all_prices()

                    if prices:
                        avg_price = sum(prices.values()) / len(prices)
                        target = self.btc_engine.extract_btc_target(pos.get("market", ""))

                        if target and target > 100:  # Must be a valid target
                            is_exceed = any(w in market_name for w in
                                          ["exceed", "above", "over", "reach", "hit", "higher"])
                            is_below = any(w in market_name for w in
                                          ["below", "under", "drop", "fall", "lower"])

                            # Only resolve when target is definitively reached/missed
                            if is_exceed:
                                if avg_price >= target * 1.005:  # %0.5 margin
                                    outcome = "YES"
                                elif avg_price < target * 0.95 and age_minutes > 30:
                                    outcome = "NO"  # 5%+ away from target and 30min passed
                            elif is_below:
                                if avg_price <= target * 0.995:
                                    outcome = "YES"
                                elif avg_price > target * 1.05 and age_minutes > 30:
                                    outcome = "NO"

                elif pos_type == "weather":
                    # Weather: resolve with NOAA real data
                    if age_minutes < 15:
                        continue  # Wait at least 15min for weather

                    forecasts = self.stats.get("last_weather_data", [])
                    if forecasts:
                        for fc in forecasts:
                            if fc and fc.get("city", "").lower() in market_name:
                                temp_range = self.weather_engine.parse_temp_range(pos.get("market", ""))
                                if temp_range:
                                    low, high = temp_range
                                    actual_temp = fc["max_temp_f"]
                                    # Is it definitively within/outside the range?
                                    if actual_temp >= low + 2 and actual_temp <= high - 2:
                                        outcome = "YES"  # Definitively within range
                                    elif actual_temp < low - 3 or actual_temp > high + 3:
                                        outcome = "NO"  # Definitively outside range
                                break

                elif pos_type == "metals":
                    # Metals: resolve by price change
                    await self._resolve_by_price_change(pos, age_minutes)
                    if pos["id"] not in [p["id"] for p in self.sim.positions]:
                        resolved += 1
                        self.auto_resolves += 1
                    continue

                if outcome:
                    result = self.sim.resolve_position(pos["id"], outcome)
                    if "error" not in result:
                        resolved += 1
                        self.auto_resolves += 1
                        pnl = result.get("pnl", 0)
                        status = result.get("status", "unknown")
                        self._log_cycle(
                            f"AUTO-RESOLVE: {pos['market'][:40]} -> {outcome} | "
                            f"{status} | PnL: ${pnl:.2f}",
                            "success" if pnl > 0 else "warning"
                        )
                        log.info(f"[AUTO] Resolve: {pos['id']} -> {outcome} | PnL: ${pnl:.2f}")

            except Exception as e:
                log.error(f"[AUTO] Resolve error for {pos.get('id', '?')}: {e}")

        return resolved

    async def _resolve_by_price_change(self, pos: dict, age_minutes: float):
        """
        Fetch REAL current price from CLOB and compare with entry_price.
        - If price changed 30%+ in our favor: resolve (WIN)
        - If price changed 30%+ against us: resolve (LOSE)
        - If end_date passed: force resolve (market concluded)
        - Otherwise: wait, do NOT resolve randomly
        """
        if age_minutes < 15:
            return  # Too new, skip

        side = pos.get("side", "YES")
        entry_price = pos.get("entry_price", 0.5)

        # Has end date passed? (Market concluded)
        end_date_str = pos.get("end_date", "")
        now = datetime.now(timezone.utc)
        if end_date_str:
            try:
                end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                if now > end_dt:
                    # Market ended - resolve based on CLOB price
                    # On Polymarket closing markets go to 0.99/0.01
                    pass  # Price check below will handle it
            except (ValueError, TypeError):
                pass

        # Fetch real current price from CLOB
        token_id = pos.get("token_id", "") if side == "YES" else pos.get("no_token_id", "")
        if not token_id:
            # No token_id, cannot resolve
            return

        try:
            current_price, spread, is_liquid = await self.poly.get_smart_price(token_id)
        except Exception as e:
            log.warning(f"[AUTO] Price fetch error for {pos['id']}: {e}")
            return

        if current_price <= 0:
            return  # Could not get price, skip

        # Price change analysis
        if entry_price <= 0:
            return

        price_change = (current_price - entry_price) / entry_price  # Positive = in our favor

        # Is market nearly resolved? (0.95+ or 0.05- price)
        market_resolved_yes = current_price >= 0.95
        market_resolved_no = current_price <= 0.05

        outcome = None
        resolve_reason = ""

        if market_resolved_yes:
            # Token nearly $1 = YES won
            outcome = "YES"
            resolve_reason = f"Market approaching YES (price={current_price:.3f})"
        elif market_resolved_no:
            # Token nearly $0 = NO won
            outcome = "NO"
            resolve_reason = f"Market approaching NO (price={current_price:.3f})"
        elif abs(price_change) >= 0.40 and age_minutes >= 60:
            # 1hr+ passed and price changed 40%+ - strong signal
            if price_change > 0:
                outcome = "YES"  # Price up -> high YES win probability
                resolve_reason = f"Price up {price_change*100:.0f}% (entry={entry_price:.3f} -> now={current_price:.3f})"
            else:
                outcome = "NO"
                resolve_reason = f"Price down {abs(price_change)*100:.0f}% (entry={entry_price:.3f} -> now={current_price:.3f})"
        else:
            # Not enough signal - WAIT, do NOT resolve randomly
            log.debug(f"[AUTO] {pos['id']} waiting: price={current_price:.3f} change={price_change*100:.1f}% age={age_minutes:.0f}m")
            return

        if outcome:
            result = self.sim.resolve_position(pos["id"], outcome)
            if "error" not in result:
                pnl = result.get("pnl", 0)
                status = result.get("status", "unknown")
                self._log_cycle(
                    f"PRICE-RESOLVE: {pos['market'][:40]} -> {outcome} | "
                    f"{status} | PnL: ${pnl:.2f} | {resolve_reason}",
                    "success" if pnl > 0 else "warning"
                )
                log.info(f"[AUTO] Real-Resolve: {pos['id']} -> {outcome} | PnL: ${pnl:.2f} | {resolve_reason}")

    async def run_scan_cycle(self) -> dict:
        """Run a single scan cycle"""
        self._reset_daily()
        cycle_start = datetime.now(timezone.utc)
        self.scan_count += 1
        self.stats["total_scans"] = self.scan_count

        log.info(f"[AUTO] === Cycle #{self.scan_count} started ===")
        self._log_cycle(f"Cycle #{self.scan_count} started")

        # 1) Fetch data in parallel (BTC + Weather + Metals)
        try:
            weather_opps, btc_opps, metals_opps = await asyncio.gather(
                self.weather_engine.find_opportunities(),
                self.btc_engine.find_opportunities(),
                self.metals_engine.find_opportunities(),
            )
        except Exception as e:
            self.stats["errors"] += 1
            self._log_cycle(f"Data fetch error: {e}", "error")
            log.error(f"[AUTO] Scan error: {e}")
            # Metals error should not stop the entire system
            weather_opps = weather_opps if 'weather_opps' in dir() else []
            btc_opps = btc_opps if 'btc_opps' in dir() else []
            metals_opps = []

        # 2) Save BTC prices
        try:
            btc_prices = await self.price_agg.get_all_prices()
            self.stats["last_btc_prices"] = btc_prices
            if btc_prices:
                avg = sum(btc_prices.values()) / len(btc_prices)
                self._log_cycle(
                    f"BTC: {len(btc_prices)} exchanges | Avg: ${avg:,.0f} | " +
                    " ".join(f"{k}=${v:,.0f}" for k, v in btc_prices.items())
                )
        except Exception as e:
            log.warning(f"[AUTO] BTC price error: {e}")

        # 3) Save NOAA data
        try:
            noaa = NOAAClient()
            forecasts = await asyncio.gather(*[
                noaa.get_hourly_forecast(lat, lon, city)
                for lat, lon, city in CFG.cities
            ])
            valid_forecasts = [f for f in forecasts if f]
            self.stats["last_weather_data"] = valid_forecasts
            if valid_forecasts:
                self._log_cycle(
                    f"NOAA: {len(valid_forecasts)} cities | " +
                    " ".join(f"{f['city']}={f['max_temp_f']}°F" for f in valid_forecasts)
                )
        except Exception as e:
            log.warning(f"[AUTO] NOAA error: {e}")

        # 4) Combine and sort opportunities
        all_opps = weather_opps + btc_opps + metals_opps
        all_opps.sort(key=lambda x: abs(x["edge"]), reverse=True)
        self.stats["last_opportunities"] = all_opps
        self.stats["metals_bets"] = self.stats.get("metals_bets", 0)

        self._log_cycle(
            f"Opportunities: {len(weather_opps)} weather + {len(btc_opps)} BTC + "
            f"{len(metals_opps)} metals = {len(all_opps)} total"
        )

        # 5) Open auto bets
        bets_placed = 0
        # Check open market names for deduplication
        open_markets = set(p.get("market", "").lower() for p in self.sim.positions)

        # Max open positions cap — prevent over-diversification on small accounts
        if len(self.sim.positions) >= 6:
            self._log_cycle(
                f"Max positions reached ({len(self.sim.positions)}/6), skipping new bets",
                "warning"
            )
            bets_placed = self.max_bets_per_cycle  # Skip the betting loop

        for opp in all_opps:
            if bets_placed >= self.max_bets_per_cycle:
                break
            if CFG.max_daily_usd > 0 and self.daily_spent >= CFG.max_daily_usd:
                self._log_cycle("Daily limit reached, skipping bets", "warning")
                break

            # Skip duplicate markets
            if opp["market"].lower() in open_markets:
                self._log_cycle(f"SKIP (position exists): {opp['market'][:40]}")
                continue

            # Directional edge filter — YES needs positive edge, NO needs negative edge
            raw_edge = opp["edge"]
            side_from_engine = opp["side"]
            if side_from_engine == "YES" and raw_edge < self.min_edge_auto:
                continue
            elif side_from_engine == "NO" and raw_edge > -self.min_edge_auto:
                continue

            # Time-to-expiry filter: skip markets expiring within 72 hours
            end_date_str = opp.get("end_date", "")
            if end_date_str:
                try:
                    end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                    hours_left = (end_dt - datetime.now(timezone.utc)).total_seconds() / 3600
                    if hours_left < 72:
                        self._log_cycle(f"SKIP (expires in {hours_left:.0f}h): {opp['market'][:40]}")
                        continue
                except (ValueError, TypeError):
                    pass

            # Confidence floor: skip low-confidence analysis
            confidence = opp.get("confidence", 0)
            if confidence > 0 and confidence < 0.30:
                self._log_cycle(f"SKIP (low confidence {confidence:.0%}): {opp['market'][:40]}")
                continue

            real_prob = opp.get("noaa_prob") or opp.get("real_prob", 0.5)
            bet_size = self.calculate_bet_size(opp["edge"], real_prob)
            if bet_size < 0.50:
                continue

            # Bahis yap
            side = opp["side"]
            price = opp["market_price"] if side == "YES" else 1 - opp["market_price"]
            if price <= 0 or price >= 1:
                continue

            result = self.sim.place_bet(
                market_id=opp.get("market_id", ""),
                market_name=opp["market"],
                side=side,
                amount=bet_size,
                price=price,
                edge=opp["edge"],
                market_type=opp["type"],
                end_date=opp.get("end_date", ""),
                token_id=opp.get("token_id", ""),
                no_token_id=opp.get("no_token_id", ""),
            )

            if "error" not in result:
                bets_placed += 1
                self.auto_bets_placed += 1
                self.daily_spent += bet_size
                self.stats["total_bets_placed"] = self.auto_bets_placed

                if opp["type"] == "weather":
                    self.stats["weather_bets"] += 1
                elif opp["type"] == "btc":
                    self.stats["btc_bets"] += 1
                elif opp["type"] == "metals":
                    self.stats["metals_bets"] = self.stats.get("metals_bets", 0) + 1

                real_prob = opp.get("noaa_prob") or opp.get("real_prob", 0)
                self._log_cycle(
                    f">> AUTO-BET: {side} ${bet_size:.0f} @ {price:.3f} | "
                    f"Edge: {opp['edge']*100:+.1f}% | "
                    f"Real: {real_prob*100:.0f}% vs Market: {opp['market_price']*100:.0f}% | "
                    f"{opp['market'][:50]}",
                    "success"
                )
                log.info(f"[AUTO] BET: {side} ${bet_size} on {opp['market'][:40]} edge={opp['edge']*100:.1f}%")
            else:
                self._log_cycle(f"Bet error: {result.get('error', '')}", "error")

        # 6) Auto-resolve open positions
        resolved = await self.auto_resolve_positions()

        # 7) Save result
        cycle_duration = (datetime.now(timezone.utc) - cycle_start).total_seconds()
        self.last_scan_time = datetime.now(timezone.utc).isoformat()

        summary = self.sim.get_portfolio_summary()
        self._log_cycle(
            f"Cycle #{self.scan_count} completed ({cycle_duration:.1f}s) | "
            f"{bets_placed} bets, {resolved} resolved | "
            f"Balance: ${summary['balance']:.0f} | WR: {summary['win_rate']}%"
        )

        scan_result = {
            "ts": self.last_scan_time,
            "cycle": self.scan_count,
            "weather_opportunities": weather_opps,
            "btc_opportunities": btc_opps,
            "metals_opportunities": metals_opps,
            "total_opportunities": len(all_opps),
            "bets_placed": bets_placed,
            "resolved": resolved,
            "portfolio": summary,
            "duration": round(cycle_duration, 1),
        }
        self.last_scan_result = scan_result

        # Save to file
        try:
            with open(DATA_DIR / "last_scan.json", "w") as f:
                json.dump(scan_result, f, indent=2, default=str)
        except Exception:
            pass

        return scan_result

    async def _loop(self):
        """Main auto trading loop"""
        self.stats["started_at"] = datetime.now(timezone.utc).isoformat()
        log.info(f"[AUTO] Auto trading started | "
                 f"Interval: {self.scan_interval}s | Min Edge: {self.min_edge_auto*100:.1f}%")
        self._log_cycle("Auto trading system started [OK]")

        while self.running:
            try:
                await self.run_scan_cycle()
            except Exception as e:
                self.stats["errors"] += 1
                log.error(f"[AUTO] Cycle error: {e}", exc_info=True)
                self._log_cycle(f"Cycle error: {e}", "error")

            # Calculate next scan time
            self.next_scan_time = (
                datetime.now(timezone.utc) + timedelta(seconds=self.scan_interval)
            ).isoformat()

            # Wait
            log.info(f"[AUTO] Next cycle in {self.scan_interval}s...")
            for _ in range(self.scan_interval):
                if not self.running:
                    break
                await asyncio.sleep(1)

        self._log_cycle("Auto trading stopped [STOP]")
        log.info("[AUTO] Auto trading stopped")

    def start(self):
        """Start trading"""
        if self.running:
            return {"status": "already_running"}
        self.running = True
        self._task = asyncio.ensure_future(self._loop())
        return {"status": "started"}

    def stop(self):
        """Stop trading"""
        if not self.running:
            return {"status": "already_stopped"}
        self.running = False
        if self._task:
            self._task.cancel()
            self._task = None
        return {"status": "stopped"}

    def update_config(self, **kwargs):
        """Update configuration"""
        if "scan_interval" in kwargs:
            self.scan_interval = max(30, int(kwargs["scan_interval"]))
        if "min_edge_pct" in kwargs:
            self.min_edge_auto = max(1, float(kwargs["min_edge_pct"])) / 100
        if "max_bets_per_cycle" in kwargs:
            self.max_bets_per_cycle = max(1, min(10, int(kwargs["max_bets_per_cycle"])))
        if "auto_resolve" in kwargs:
            self.auto_resolve_enabled = bool(kwargs["auto_resolve"])
        return self.get_status()


# --- ENTRY POINT --------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(ArbBot().run())
