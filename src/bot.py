"""
Polymarket Arbitraj Botu v2.0
- Hava Durumu (NOAA) -> Polymarket Weather Markets
- BTC Fiyat Arbitraji -> Binance / OKX / Coinbase / Upbit vs Polymarket
- Simulasyon Modu destegi
- Tum kategorilerdeki marketleri tarama
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

    # Hava durumu sehirleri (sadece ABD - NOAA sadece ABD destekler)
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
        # NOAA sadece ABD sehirlerini destekler
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


# --- NOAA HAVA DURUMU --------------------------------------------------------

class NOAAClient:
    """NOAA Weather API - sadece ABD sehirleri icin calısır"""
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
                # Adim 1: grid point al
                url = f"{self.BASE}/points/{lat:.4f},{lon:.4f}"
                async with s.get(url, headers=self._headers()) as r:
                    if r.status != 200:
                        log.warning(f"NOAA points failed for {city}: HTTP {r.status}")
                        return None
                    data = await r.json()
                    forecast_url = data["properties"]["forecastHourly"]

                # Adim 2: saatlik tahmin al
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

                    # Fallback: bugunun kalan saatleri
                    if not tomorrow_temps:
                        today_date = now.date()
                        tomorrow_temps = [
                            p["temperature"] for p in periods[:12]
                            if datetime.fromisoformat(p["startTime"]).date() == today_date
                        ]

                    if not tomorrow_temps:
                        log.info(f"NOAA: {city} icin sicaklik verisi yok")
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
        """Kraken BTC/USD fiyati - 5. borsa"""
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
        """CLOB API sampling-markets - gercek aktif marketleri ve token ID'leri alir"""
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
        """CLOB price endpoint - referans fiyat (son islem/midpoint)"""
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
        """Orderbook'tan gercek bid/ask/spread bilgisi alir"""
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
        """Orderbook + referans fiyat birlestirerek en iyi fiyati hesaplar.
        Returns: (price, spread, is_liquid)
        - Likit market (spread<%30): orderbook midpoint kullanir
        - Illikit market (spread>%30): /price referans fiyatini kullanir
        """
        # Paralel cek: orderbook + referans fiyat
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
        """Birden fazla token fiyatini paralel cek"""
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
        """Token icin order book cek"""
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

        # NOAA verilerini cek (ABD sehirleri icin)
        forecasts = await asyncio.gather(*[
            self.noaa.get_hourly_forecast(lat, lon, city)
            for lat, lon, city in CFG.cities
        ])
        forecast_map = {f["city"]: f for f in forecasts if f}
        if forecast_map:
            log.info(f"NOAA: {len(forecast_map)} sehir: " +
                     ", ".join(f"{c}={d['max_temp_f']}F" for c, d in forecast_map.items()))

        # Polymarket weather/temperature marketlerini cek (CLOB API)
        markets = await self.poly.get_weather_markets()
        log.info(f"Weather: {len(markets)} market bulundu")

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

            # CLOB'dan gercek fiyat cek
            market_price = await self.poly.get_token_price(token_id)
            if market_price <= 0:
                continue

            # Global temperature market mi?
            if "global temperature" in question.lower() or "temperature increase" in question.lower():
                temp_range = self.parse_global_temp_range(question)
                if not temp_range:
                    continue
                low, high = temp_range
                # Basit tahmin: su anki trend ve mevcut diger marketlerin fiyatlarina gore
                # Daha sofistike model icin dis kaynak gerekir
                noaa_prob = market_price  # Baslangic olarak market fiyatini kullan
                edge = 0  # Bu turdeki marketler icin dis veri lazim
            else:
                # ABD sehir sicaklik marketi
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
        # "$1m", "$1.5m", "$1M" gibi milyon ifadeleri
        match = re.search(r"\$\s*([\d.]+)\s*[mM]", question)
        if match:
            return float(match.group(1)) * 1_000_000
        # "$1b", "$1B" gibi milyar ifadeleri
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
        log.info(f"BTC Ortalama: ${avg_btc:,.0f}")

        markets = await self.poly.get_btc_markets()
        log.info(f"BTC: {len(markets)} market bulundu")

        for market in markets:
            question = market.get("question", "")

            target = self.extract_btc_target(question)
            if not target:
                log.info(f"BTC SKIP (hedef bulunamadi): {question[:60]}")
                continue

            # Hedef makul aralikta mi? ($1000 - $10M arasi)
            if target < 1000 or target > 10_000_000:
                log.info(f"BTC SKIP (hedef aralik disi ${target:,.0f}): {question[:60]}")
                continue

            tokens = market.get("tokens", [])
            yes_token = next((t for t in tokens if t.get("outcome", "").upper() == "YES"), None)
            if not yes_token:
                continue

            token_id = yes_token.get("token_id", "")
            if not token_id:
                continue

            # CLOB'dan gercek fiyat cek (orderbook + referans)
            market_price, spread, is_liquid = await self.poly.get_smart_price(token_id)
            if market_price <= 0.01 or market_price >= 0.99:
                continue  # Cok asiri fiyatlar -> likit degil veya sonuclanmis

            liq_tag = "LIQ" if is_liquid else f"ILLIQ(spread={spread:.0%})"

            is_exceed = any(w in question.lower() for w in
                          ["exceed", "above", "over", "reach", "hit", "higher"])
            is_below = any(w in question.lower() for w in
                          ["below", "under", "drop", "fall", "lower"])

            # End date'e gore zaman faktoru
            end_date_str = market.get("end_date_iso", "")
            time_factor = 1.0
            if end_date_str:
                try:
                    end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                    days_until = (end_dt - datetime.now(timezone.utc)).days
                    if days_until > 180:
                        time_factor = 0.7   # 6+ ay: belirsizlik yuksek
                    elif days_until > 90:
                        time_factor = 0.8   # 3-6 ay
                    elif days_until > 30:
                        time_factor = 0.9   # 1-3 ay
                    elif days_until > 7:
                        time_factor = 0.95  # 1-4 hafta
                    # 7 gun icinde: zaman faktoru 1.0
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
                # "Will BTC be between X and Y" gibi aralik marketleri
                # Simdilik atla
                continue

            # Olasilik sinirla
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
    """XAU/XAG Polymarket bahisleri icin gercek veri analizi"""

    def __init__(self):
        self.poly = PolymarketClient()
        self._xau_engine = None  # Lazy import

    def _get_engine(self):
        if self._xau_engine is None:
            try:
                from xau_xag_engine import XAUXAGEngine
                self._xau_engine = XAUXAGEngine()
            except ImportError:
                log.warning("xau_xag_engine modulu bulunamadi")
                return None
        return self._xau_engine

    async def find_opportunities(self) -> list:
        engine = self._get_engine()
        if not engine:
            return []

        opportunities = []

        # Polymarket'ten XAU/XAG marketlerini bul
        markets = await self.poly._fetch_clob_markets(pages=3)

        # Daha spesifik filtre: gold/silver futures, XAU, XAG, ounce
        # Basit "gold" veya "silver" tek basina cok fazla false positive yaratir
        # NOT: "ounce" kullanma! "announce" icinde "ounce" var -> false positive!
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
            # Pattern eslesmesi
            if any(p in q for p in metal_patterns):
                metal_markets.append(m)
                continue
            # "gold" veya "silver" icerip finans/fiyat ile ilgili olanlar
            if ("gold" in q or "silver" in q) and any(w in q for w in
                ["price", "settle", "hit", "above", "below", "reach", "$", "futures"]):
                metal_markets.append(m)

        log.info(f"Metals: {len(metal_markets)} market bulundu (toplam {len(markets)} taranarak)")
        if metal_markets:
            log.info(f"  Ilk market: {metal_markets[0].get('question', '')[:70]}")

        # Performans icin max 10 market analiz et
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

            # CLOB'dan gercek fiyat (orderbook + referans)
            market_price, spread, is_liquid = await self.poly.get_smart_price(token_id)
            if market_price <= 0:
                continue

            # Metal turu ve yon belirle
            q_lower = question.lower()
            metal = "XAU" if any(w in q_lower for w in ["gold", "xau"]) else "XAG"
            is_up = any(w in q_lower for w in ["above", "over", "exceed", "higher", "reach", "hit", "rise"])
            is_down = any(w in q_lower for w in ["below", "under", "fall", "drop", "lower", "decline"])
            direction = "UP" if is_up else "DOWN" if is_down else "UP"

            try:
                decision = await engine.analyze(metal, market_price, direction)

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
    """Tum aktif Polymarket marketlerini tarar, asiri yuksek/dusuk fiyatli olanlari raporlar"""

    def __init__(self):
        self.poly = PolymarketClient()

    async def find_mispriced_markets(self) -> list:
        """Fiyatı 0.05 altında veya 0.95 üstünde olan yüksek likidite marketleri bul"""
        opportunities = []
        markets = await self.poly.get_all_active_markets(limit=200)
        log.info(f"Genel Tarama: {len(markets)} aktif market bulundu")

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

            # Ilginc kategoriler
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


# --- SIMULASYON MOTORU --------------------------------------------------------

class SimulationEngine:
    """Paper trading / simulasyon motoru"""

    def __init__(self, initial_balance: float = 10000.0):
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.positions = []  # Acik pozisyonlar
        self.closed_trades = []  # Kapanmis islemler
        self.trade_history = []  # Tum islem gecmisi
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
                log.info(f"SIM: State yuklendi - Bakiye: ${self.balance:.2f}, "
                         f"{len(self.positions)} acik pozisyon")
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
        total_pnl = sum(t.get("pnl", 0) for t in self.closed_trades)
        win_count = sum(1 for t in self.closed_trades if t.get("pnl", 0) > 0)
        total_closed = len(self.closed_trades)

        return {
            "balance": round(self.balance, 2),
            "initial_balance": self.initial_balance,
            "total_invested": round(total_invested, 2),
            "portfolio_value": round(self.balance + total_invested, 2),
            "total_pnl": round(total_pnl, 2),
            "pnl_pct": round((total_pnl / self.initial_balance) * 100, 2) if self.initial_balance > 0 else 0,
            "open_positions": len(self.positions),
            "closed_trades": total_closed,
            "win_rate": round((win_count / total_closed) * 100, 1) if total_closed > 0 else 0,
            "total_trades": len(self.trade_history),
        }

    def resolve_position(self, position_id: str, outcome: str) -> dict:
        """Pozisyonu sonuclandir (YES/NO) - binary (tam kazanc/tam kayip)"""
        pos = None
        for i, p in enumerate(self.positions):
            if p["id"] == position_id:
                pos = self.positions.pop(i)
                break
        if not pos:
            return {"error": "Pozisyon bulunamadi"}

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

    def sell_position(self, position_id: str, current_price: float) -> dict:
        """
        Pozisyonu mevcut piyasa fiyatindan sat (gercekci PnL).
        Binary resolve yerine, gercek fiyat degisimine gore PnL hesaplar.
        shares = amount / entry_price
        current_value = shares * current_price
        pnl = current_value - amount
        """
        pos = None
        for i, p in enumerate(self.positions):
            if p["id"] == position_id:
                pos = self.positions.pop(i)
                break
        if not pos:
            return {"error": "Pozisyon bulunamadi"}

        entry_price = pos.get("entry_price", 0.5)
        amount = pos.get("amount", 0)

        if entry_price > 0:
            shares = amount / entry_price
            payout = shares * current_price
        else:
            payout = amount  # entry=0 ise orjinal miktari geri ver

        pnl = payout - amount

        pos["status"] = "won" if pnl >= 0 else "lost"
        pos["pnl"] = round(pnl, 4)
        pos["payout"] = round(payout, 4)
        pos["sell_price"] = round(current_price, 4)
        pos["resolved_at"] = datetime.now(timezone.utc).isoformat()
        pos["outcome"] = f"SOLD@{current_price:.3f}"

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
        self.last_scan_results = {}  # Web API icin son tarama sonuclari

    def reset_daily_if_needed(self):
        today = datetime.now(timezone.utc).date()
        if today != self.last_reset:
            self.daily_spent = 0.0
            self.last_reset = today
            log.info("Gunluk limit sifirlandi")

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
        log.info(f"Yeni dongü: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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

        # Sonuclari kaydet
        with open(DATA_DIR / "last_scan.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

        if not all_opps:
            log.info("Bu dongude firsat bulunamadi")
            log.info(f"Genel taramada {len(general_markets)} market listelendi")
            return result

        log.info(f"{len(all_opps)} firsat bulundu "
                 f"({len(weather_opps)} hava, {len(btc_opps)} BTC)")

        for opp in all_opps[:10]:
            emoji = "W" if opp["type"] == "weather" else "B"
            log.info(f"[{emoji}] {opp['market'][:50]} | "
                     f"Edge: {opp['edge']*100:+.1f}% | "
                     f"{opp['side']} @ {opp['market_price']:.3f}")

        # Simulasyon modunda otomatik bet
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
        mode = "SIMULASYON" if CFG.simulation_mode else "CANLI"
        log.info(f"Polymarket Arbitraj Botu v2.0 baslatiliyor [{mode}]")
        log.info(f"Min edge: {CFG.min_edge_pct}% | Max order: ${CFG.max_order_usd} | "
                 f"Gunluk limit: ${CFG.max_daily_usd}")

        if not CFG.simulation_mode and not CFG.poly_private_key:
            log.error("POLY_PRIVATE_KEY eksik! .env dosyasini kontrol et.")
            log.info("Simulasyon moduna geciliyor...")
            CFG.simulation_mode = True
            self.sim = SimulationEngine(CFG.initial_balance)

        while True:
            try:
                await self.run_cycle()
            except KeyboardInterrupt:
                log.info("Bot durduruldu")
                break
            except Exception as e:
                log.error(f"Dongu hatasi: {e}", exc_info=True)

            log.info(f"Sonraki dongu {CFG.loop_interval_sec}s sonra...")
            await asyncio.sleep(CFG.loop_interval_sec)


# --- OTOMATİK TRADİNG SİSTEMİ -------------------------------------------------

class AutoTrader:
    """
    Tam otomatik trading sistemi:
    - Periyodik olarak BTC (5 borsa) ve Weather (NOAA) firsatlarini tarar
    - Edge bulunursa otomatik simülasyon bahisi açar
    - Açık pozisyonları gerçek verilere göre otomatik resolve eder
    - Tüm istatistikleri takip eder
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
        self.scan_interval = CFG.loop_interval_sec  # saniye
        self.max_bets_per_cycle = 3
        self.min_edge_auto = CFG.min_edge_pct / 100  # Otomatik bahis icin minimum edge
        self.auto_resolve_enabled = True

        # Istatistikler
        self.stats = {
            "total_scans": 0,
            "total_bets_placed": 0,
            "total_resolved": 0,
            "weather_bets": 0,
            "btc_bets": 0,
            "last_btc_prices": {},
            "last_weather_data": [],
            "last_opportunities": [],
            "cycle_log": [],  # Son 50 döngü logu
            "started_at": None,
            "errors": 0,
        }
        self._task = None

    def _reset_daily(self):
        today = datetime.now(timezone.utc).date()
        if today != self.last_reset_date:
            self.daily_spent = 0.0
            self.last_reset_date = today
            log.info("[AUTO] Gunluk limit sifirlandi")

    def get_status(self) -> dict:
        """Mevcut durum bilgisi"""
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
        """Döngü logunu kaydet"""
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "msg": msg,
            "level": level,
        }
        self.stats["cycle_log"].append(entry)
        # Son 100 log tut
        if len(self.stats["cycle_log"]) > 100:
            self.stats["cycle_log"] = self.stats["cycle_log"][-100:]

    def _get_tier_max(self, balance: float) -> float:
        """Bakiyeye gore kademeli max bahis limiti"""
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
        Bakiyeye gore dinamik + kademeli bahis boyutu:
        Kademe limitleri:
          $0-100    -> max $2
          $100-200  -> max $5
          $200-500  -> max $10
          $500-1000 -> max $20
          $1000-2000-> max $50
          $2000+    -> max $100-200
        Edge ve olasilik yuksekse kademe icinde ust sinira yaklasir.
        """
        balance = self.sim.balance
        abs_edge = abs(edge)
        tier_max = self._get_tier_max(balance)

        # Edge bazli oran: %5 edge -> kademenin %25'i, %30+ edge -> kademenin %100'u
        edge_ratio = min(abs_edge * 3.3, 1.0)   # 0.05*3.3=0.165 -> 0.30*3.3=1.0
        edge_ratio = max(edge_ratio, 0.15)        # minimum kademenin %15'i

        # Olasilik carpani: kesinlik yuksekse kademe limitine yaklastir
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
        Acik pozisyonlari CLOB fiyat degisimine gore otomatik resolve et.
        TUM pozisyon tipleri (btc, metals, weather) ayni sistemi kullanir.
        ZORUNLU KURAL: Maksimum 60 dakika (1 saat) sonra zorla kapatilir.
        """
        if not self.auto_resolve_enabled:
            return 0

        resolved = 0
        now = datetime.now(timezone.utc)
        positions_copy = list(self.sim.positions)

        for pos in positions_copy:
            try:
                entry_time = pos.get("ts", "")
                age_minutes = 0

                # Pozisyon yasi hesapla
                if entry_time:
                    try:
                        entry_dt = datetime.fromisoformat(entry_time)
                        age_minutes = (now - entry_dt).total_seconds() / 60
                        if age_minutes < 5:
                            continue  # 5 dakikadan yeni, atla
                    except (ValueError, TypeError):
                        age_minutes = 0

                # TUM pozisyonlar icin CLOB fiyat degisimine gore resolve
                await self._resolve_by_price_change(pos, age_minutes)
                if pos["id"] not in [p["id"] for p in self.sim.positions]:
                    resolved += 1
                    self.auto_resolves += 1

            except Exception as e:
                log.error(f"[AUTO] Resolve error for {pos.get('id', '?')}: {e}")

        return resolved

    async def _resolve_by_price_change(self, pos: dict, age_minutes: float):
        """
        CLOB'dan GERCEK guncel fiyat cekip entry_price ile karsilastir.
        Kademeliesiksistemi:
          - Market settled: token fiyati >=0.90 veya <=0.10 -> hemen resolve
          - Take profit: %20+ lehimize ve 20dk+ -> WIN
          - Stop loss: %25+ aleyhimize ve 20dk+ -> LOSE
          - Orta sinyal: %15+ degisim ve 2 saat+ -> resolve
          - Zaman asimi: 6 saat+ -> esik %10'a duser, 24 saat+ -> zorla resolve
          - End_date gecmisse -> zorla resolve
        """
        if age_minutes < 10:
            return  # Cok yeni, atla

        side = pos.get("side", "YES")
        entry_price = pos.get("entry_price", 0.5)

        # End date gecmis mi? (Market sonuclandi)
        end_date_str = pos.get("end_date", "")
        now = datetime.now(timezone.utc)
        end_date_passed = False
        if end_date_str:
            try:
                end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                if now > end_dt:
                    end_date_passed = True
            except (ValueError, TypeError):
                pass

        # CLOB'dan gercek guncel fiyati cek
        token_id = pos.get("token_id", "") if side == "YES" else pos.get("no_token_id", "")
        if not token_id:
            # token_id yok, resolve yapamayiz
            log.info(f"[AUTO] {pos['id']} token_id yok, resolve atilaniliyor")
            return

        try:
            current_price, spread, is_liquid = await self.poly.get_smart_price(token_id)
        except Exception as e:
            log.warning(f"[AUTO] Price fetch error for {pos['id']}: {e}")
            return

        if current_price <= 0:
            log.info(f"[AUTO] {pos['id']} CLOB fiyat alinamadi (price={current_price}), atlaniyor")
            return  # Fiyat alinamadi, atla

        # Fiyat degisim analizi
        if entry_price <= 0:
            log.info(f"[AUTO] {pos['id']} entry_price=0, atlaniyor")
            return

        # price_change: pozitif = token fiyati artti (lehimize)
        price_change = (current_price - entry_price) / entry_price

        # Market neredeyse sonuclandi mi?
        # ONEMLI: Dusuk fiyatli long-shot bahisler (entry<0.15) icin
        # market_resolved_no esigi uygulanmaz - cunku zaten dusuk fiyatla girilmis
        market_resolved_yes = current_price >= 0.90
        # Sadece entry_price yuksekse (>0.15) market_resolved_no uygula
        # Yoksa dusuk fiyatli tokenleri yanlis kapatir
        market_resolved_no = current_price <= 0.05 and entry_price > 0.15

        outcome = None
        resolve_reason = ""

        if market_resolved_yes:
            # Token neredeyse $1 = bu token'in side'i kazandi
            outcome = side  # Bizim token yukseldiyse BIZ kazandik
            resolve_reason = f"Market {side}'a yaklasti (price={current_price:.3f})"
        elif market_resolved_no:
            # Token neredeyse $0 VE giris fiyati yuksekti = gercekten deger kaybetti
            outcome = "NO" if side == "YES" else "YES"  # Karsitaraf kazandi
            resolve_reason = f"Token degerini kaybetti (price={current_price:.3f}, entry={entry_price:.3f})"
        elif end_date_passed:
            # Market bitmis - mevcut fiyattan sat
            result = self.sim.sell_position(pos["id"], current_price)
            if "error" not in result:
                pnl = result.get("pnl", 0)
                self._log_cycle(
                    f"END-DATE SELL: {pos['market'][:40]} | PnL: ${pnl:.4f} | "
                    f"(entry={entry_price:.3f} -> sell={current_price:.3f})",
                    "success" if pnl >= 0 else "warning"
                )
                log.info(f"[AUTO] END-DATE SELL: {pos['id']} | PnL: ${pnl:.4f}")
            return
        else:
            # ZORUNLU KAPANIŞ: 60 dakika (1 saat) sonra mevcut fiyattan sat
            if age_minutes >= 60:
                # sell_position ile gercekci PnL hesapla (binary degil)
                result = self.sim.sell_position(pos["id"], current_price)
                if "error" not in result:
                    pnl = result.get("pnl", 0)
                    status = result.get("status", "unknown")
                    self._log_cycle(
                        f"1H-SELL: {pos['market'][:40]} | {status} | PnL: ${pnl:.4f} | "
                        f"change={price_change*100:+.1f}% (entry={entry_price:.3f} -> sell={current_price:.3f})",
                        "success" if pnl >= 0 else "warning"
                    )
                    log.info(
                        f"[AUTO] 1H-SELL: {pos['id']} | PnL: ${pnl:.4f} | "
                        f"entry={entry_price:.3f} sell={current_price:.3f} change={price_change*100:+.1f}%"
                    )
                return  # sell_position zaten islem yapti, asagiya gerek yok
            # Kademeli esik sistemi (60 dk altı)
            elif age_minutes >= 40:  # 40-60 dk
                tp_threshold = 0.05   # %5 kar al
                sl_threshold = 0.08   # %8 zarar kes
                resolve_reason_prefix = "40m+"
            elif age_minutes >= 20:   # 20-40 dk
                tp_threshold = 0.10   # %10 kar al
                sl_threshold = 0.15   # %15 zarar kes
                resolve_reason_prefix = "20m+"
            else:
                # 10-20 dk arasi -> sadece büyük hareketlerde
                tp_threshold = 0.20
                sl_threshold = 0.25
                resolve_reason_prefix = "10m+"

            # outcome henuz set edilmediyse (60dk force-close degilse) esik kontrol et
            if outcome is None:
                if price_change >= tp_threshold:
                    # Fiyat lehimize gitti -> BIZ kazandik
                    outcome = side
                    resolve_reason = (
                        f"{resolve_reason_prefix} Take Profit: fiyat %{price_change*100:.0f} artti "
                        f"(entry={entry_price:.3f} -> now={current_price:.3f})"
                    )
                elif price_change <= -sl_threshold:
                    # Fiyat aleyhimize gitti -> BIZ kaybettik
                    outcome = "NO" if side == "YES" else "YES"
                    resolve_reason = (
                        f"{resolve_reason_prefix} Stop Loss: fiyat %{abs(price_change)*100:.0f} dustu "
                        f"(entry={entry_price:.3f} -> now={current_price:.3f})"
                    )
                else:
                    # Yeterli sinyal yok - BEKLE
                    log.info(
                        f"[AUTO] {pos['id']} bekleniyor: {side} price={current_price:.3f} "
                        f"change={price_change*100:+.1f}% age={age_minutes:.0f}m "
                        f"(tp={tp_threshold*100:.0f}% sl={sl_threshold*100:.0f}%)"
                    )
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
        """Tek bir tarama döngüsü çalıştır"""
        self._reset_daily()
        cycle_start = datetime.now(timezone.utc)
        self.scan_count += 1
        self.stats["total_scans"] = self.scan_count

        log.info(f"[AUTO] === Dongu #{self.scan_count} basladi ===")
        self._log_cycle(f"Döngü #{self.scan_count} başladı")

        # 1) Verileri paralel cek (BTC + Weather + Metals)
        try:
            weather_opps, btc_opps, metals_opps = await asyncio.gather(
                self.weather_engine.find_opportunities(),
                self.btc_engine.find_opportunities(),
                self.metals_engine.find_opportunities(),
            )
        except Exception as e:
            self.stats["errors"] += 1
            self._log_cycle(f"Veri cekme hatasi: {e}", "error")
            log.error(f"[AUTO] Scan error: {e}")
            # Metals hatasi tum sistemi durdurmasin
            weather_opps = weather_opps if 'weather_opps' in dir() else []
            btc_opps = btc_opps if 'btc_opps' in dir() else []
            metals_opps = []

        # 2) BTC fiyatlarını kaydet
        try:
            btc_prices = await self.price_agg.get_all_prices()
            self.stats["last_btc_prices"] = btc_prices
            if btc_prices:
                avg = sum(btc_prices.values()) / len(btc_prices)
                self._log_cycle(
                    f"BTC: {len(btc_prices)} borsa | Ort: ${avg:,.0f} | " +
                    " ".join(f"{k}=${v:,.0f}" for k, v in btc_prices.items())
                )
        except Exception as e:
            log.warning(f"[AUTO] BTC price error: {e}")

        # 3) NOAA verilerini kaydet
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
                    f"NOAA: {len(valid_forecasts)} şehir | " +
                    " ".join(f"{f['city']}={f['max_temp_f']}°F" for f in valid_forecasts)
                )
        except Exception as e:
            log.warning(f"[AUTO] NOAA error: {e}")

        # 4) Firsatlari birlestir ve EV (Expected Value) bazli sirala
        all_opps = weather_opps + btc_opps + metals_opps
        # EV = edge * confidence * (1 - spread_penalty) * liquidity_bonus
        def ev_score(opp):
            e = abs(opp["edge"])
            conf = opp.get("confidence", 0.7)
            spread = opp.get("ob_spread", 0.05)
            spread_penalty = min(spread * 5, 0.5)  # max %50 ceza
            liquid_bonus = 1.1 if opp.get("is_liquid", False) else 0.8
            return e * conf * (1 - spread_penalty) * liquid_bonus
        all_opps.sort(key=ev_score, reverse=True)
        self.stats["last_opportunities"] = all_opps
        self.stats["metals_bets"] = self.stats.get("metals_bets", 0)

        self._log_cycle(
            f"Firsatlar: {len(weather_opps)} hava + {len(btc_opps)} BTC + "
            f"{len(metals_opps)} metals = {len(all_opps)} toplam"
        )

        # 5) Otomatik bahis aç
        bets_placed = 0
        # Acik pozisyonlardaki market isimlerini kontrol et (duplicate onleme)
        open_markets = set(p.get("market", "").lower() for p in self.sim.positions)

        for opp in all_opps:
            if bets_placed >= self.max_bets_per_cycle:
                break
            if CFG.max_daily_usd > 0 and self.daily_spent >= CFG.max_daily_usd:
                self._log_cycle("Gunluk limit doldu, bahis atlanadi", "warning")
                break

            # Ayni markete tekrar bahis acma
            if opp["market"].lower() in open_markets:
                self._log_cycle(f"SKIP (acik pozisyon var): {opp['market'][:40]}")
                continue

            edge = abs(opp["edge"])
            if edge < self.min_edge_auto:
                continue

            # --- GÜÇLÜ BAHİS FİLTRESİ ---
            # 1) Likidite kontrolü: market likit olmalı
            if not opp.get("is_liquid", True):
                self._log_cycle(f"SKIP (likit degil): {opp['market'][:40]}")
                continue

            # 2) Spread kontrolü: orderbook spread çok geniş olmamalı
            ob_spread = opp.get("ob_spread", 0)
            if ob_spread > 0.10:  # %10'dan fazla spread riskli
                self._log_cycle(f"SKIP (spread cok genis {ob_spread:.1%}): {opp['market'][:40]}")
                continue

            # 3) Confidence kontrolü (metals için): en az %30 güven
            confidence = opp.get("confidence", 1.0)
            if opp["type"] == "metals" and confidence < 0.30:
                self._log_cycle(f"SKIP (dusuk confidence {confidence:.0%}): {opp['market'][:40]}")
                continue

            # 4) Çok düşük fiyatlı tokenlarda dikkatli ol (penny bet riski)
            market_price = opp.get("market_price", 0.5)
            if market_price < 0.02 or market_price > 0.98:
                # Çok uç fiyatlar - bunlar genelde "neredeyse kesin" marketler
                # Edge yüksek görünür ama gerçekte çok riskli
                if edge < 0.25:  # Edge %25'ten az ise atla
                    self._log_cycle(f"SKIP (uc fiyat {market_price:.3f}, edge yetersiz): {opp['market'][:40]}")
                    continue

            # 5) Kısa vadeli market filtresi - 30 günden uzak marketlere bahis açma
            opp_end = opp.get("end_date", "")
            if opp_end:
                try:
                    opp_end_dt = datetime.fromisoformat(opp_end.replace("Z", "+00:00"))
                    opp_days_left = (opp_end_dt - datetime.now(timezone.utc)).days
                    if opp_days_left > 30:
                        continue  # 30 günden uzak market, sessizce atla
                except (ValueError, TypeError):
                    pass

            real_prob = opp.get("noaa_prob") or opp.get("real_prob", 0.5)

            # 6) Güçlü olasılık filtresi: gerçek olasılık çok belirsiz olmamalı
            # %35-%65 arası "belirsiz bölge" - burada bahis riskli
            if 0.35 <= real_prob <= 0.65 and edge < 0.20:
                self._log_cycle(f"SKIP (belirsiz olasilik {real_prob:.0%}): {opp['market'][:40]}")
                continue

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
                    f"Gerçek: {real_prob*100:.0f}% vs Market: {opp['market_price']*100:.0f}% | "
                    f"{opp['market'][:50]}",
                    "success"
                )
                log.info(f"[AUTO] BET: {side} ${bet_size} on {opp['market'][:40]} edge={opp['edge']*100:.1f}%")
            else:
                self._log_cycle(f"Bahis hatası: {result.get('error', '')}", "error")

        # 6) Açık pozisyonları otomatik resolve et
        resolved = await self.auto_resolve_positions()

        # 7) Sonucu kaydet
        cycle_duration = (datetime.now(timezone.utc) - cycle_start).total_seconds()
        self.last_scan_time = datetime.now(timezone.utc).isoformat()

        summary = self.sim.get_portfolio_summary()
        self._log_cycle(
            f"Döngü #{self.scan_count} tamamlandı ({cycle_duration:.1f}s) | "
            f"{bets_placed} bahis, {resolved} resolve | "
            f"Bakiye: ${summary['balance']:.0f} | WR: {summary['win_rate']}%"
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

        # Dosyaya kaydet
        try:
            with open(DATA_DIR / "last_scan.json", "w") as f:
                json.dump(scan_result, f, indent=2, default=str)
        except Exception:
            pass

        return scan_result

    async def _loop(self):
        """Ana otomatik trading döngüsü"""
        self.stats["started_at"] = datetime.now(timezone.utc).isoformat()
        log.info(f"[AUTO] Otomatik trading baslatildi | "
                 f"Interval: {self.scan_interval}s | Min Edge: {self.min_edge_auto*100:.1f}%")
        self._log_cycle("Otomatik trading sistemi baslatildi [OK]")

        while self.running:
            try:
                await self.run_scan_cycle()
            except Exception as e:
                self.stats["errors"] += 1
                log.error(f"[AUTO] Dongu hatasi: {e}", exc_info=True)
                self._log_cycle(f"Döngü hatası: {e}", "error")

            # Sonraki tarama zamanını hesapla
            self.next_scan_time = (
                datetime.now(timezone.utc) + timedelta(seconds=self.scan_interval)
            ).isoformat()

            # Bekleme
            log.info(f"[AUTO] Sonraki dongu {self.scan_interval}s sonra...")
            for _ in range(self.scan_interval):
                if not self.running:
                    break
                await asyncio.sleep(1)

        self._log_cycle("Otomatik trading durduruldu [STOP]")
        log.info("[AUTO] Otomatik trading durduruldu")

    def start(self):
        """Trading'i başlat"""
        if self.running:
            return {"status": "already_running"}
        self.running = True
        self._task = asyncio.ensure_future(self._loop())
        return {"status": "started"}

    def stop(self):
        """Trading'i durdur"""
        if not self.running:
            return {"status": "already_stopped"}
        self.running = False
        if self._task:
            self._task.cancel()
            self._task = None
        return {"status": "stopped"}

    def update_config(self, **kwargs):
        """Konfigürasyonu güncelle"""
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
