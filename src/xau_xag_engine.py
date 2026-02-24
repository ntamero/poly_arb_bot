"""
XAU / XAG Olasilik ve Karar Motoru
====================================
Veri Katmanlari:
  1. Fiyat        -> GoldAPI.io + MetalpriceAPI (cift kaynak, capraz dogrulama)
  2. Makro        -> FRED API  (DXY, reel getiri, CPI, M2, Fed Funds)
  3. Pozisyonlama -> CFTC COT (Haftalik: Commercial short, Managed Money long)
  4. Momentum     -> Hesaplanan teknik indikatorler (RSI, ATR, BB, EMA)
  5. Polymarket   -> Implied probability (piyasa ne soyluyor?)

Karar Motoru:
  - Her faktor icin normalize edilmis skor (-1 ile +1)
  - Agirlikli olasilik birlestirme (Bayes-benzeri)
  - Edge hesaplama: (NOAA/gercek prob) - (Polymarket implied prob)
  - Kelly criterion ile pozisyon buyuklugu
"""

import asyncio
import aiohttp
import json
import math
import logging
import os
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("xau-xag-engine")


# --- KONFIGURASYON ------------------------------------------------------------

GOLDAPI_KEY      = os.getenv("GOLDAPI_KEY", "")        # goldapi.io (ucretsiz 100/ay)
METALPRICE_KEY   = os.getenv("METALPRICE_KEY", "")     # metalpriceapi.com
FRED_KEY         = os.getenv("FRED_KEY", "")           # fred.stlouisfed.org (ucretsiz)

# FRED Seri ID'leri
FRED_SERIES = {
    "DXY":         "DTWEXBGS",   # Dolar endeksi (genis, gunluk)
    "REAL_YIELD":  "DFII10",     # 10Y reel getiri (TIPS)
    "CPI_YOY":     "CPIAUCSL",   # CPI (aylik)
    "FED_RATE":    "FEDFUNDS",   # Fed Funds Rate
    "M2":          "M2SL",       # M2 para arzi
    "VIX":         "VIXCLS",     # VIX (risk istahi proxy)
    "GOLD_ETF":    "GOLDAMGBD228NLBM",  # LBMA altin fiyati (referans)
}

# COT: CFTC Altin (Gold) -> code 088691, Gumus -> code 084691
CFTC_URL = "https://publicreporting.cftc.gov/api/explore/dataset/commitments_of_traders/records/"
GOLD_CFTC_CODE  = "088691"
SILVER_CFTC_CODE = "084691"


# --- VERI YAPILARI ------------------------------------------------------------

@dataclass
class MetalPrice:
    metal: str          # XAU veya XAG
    price: float
    bid: float
    ask: float
    change_pct: float
    source: str
    timestamp: datetime


@dataclass
class MacroSnapshot:
    dxy: Optional[float]        = None   # Dolar endeksi
    real_yield_10y: Optional[float] = None  # 10Y reel getiri %
    cpi_yoy: Optional[float]    = None   # CPI yillik degisim %
    fed_rate: Optional[float]   = None   # Fed Funds Rate %
    m2_growth: Optional[float]  = None   # M2 buyume %
    vix: Optional[float]        = None   # VIX


@dataclass
class COTSnapshot:
    metal: str
    commercial_net: int         # Commercial net pozisyon (negatif = net short)
    managed_money_net: int      # Managed money net pozisyon
    commercial_net_pct: float   # Son 52 haftada % yuzdelik (0-100)
    managed_money_pct: float    # Son 52 haftada % yuzdelik (0-100)
    report_date: str


@dataclass
class TechnicalSnapshot:
    metal: str
    rsi_14: float               # 0-100
    atr_pct: float              # ATR % fiyatin yuzdesi olarak
    bb_position: float          # Bollinger Band pozisyonu (-1=alt, 0=orta, 1=ust)
    ema_20_vs_50: float         # EMA20/EMA50 - 1 (pozitif = bullish)
    momentum_5d: float          # 5 gunluk getiri %


@dataclass
class FactorScore:
    name: str
    raw_value: float
    score: float        # -1 (cok bearish) ile +1 (cok bullish) arasi
    weight: float
    description: str


@dataclass
class MetalDecision:
    metal: str
    direction: str              # "UP" / "DOWN" / "NEUTRAL"
    probability: float          # 0-1 (hedefe ulasma olasiligi)
    polymarket_price: float     # Polymarket'in implied probability
    edge: float                 # probability - polymarket_price
    confidence: float           # Karar guven skoru 0-1
    order_side: str             # "YES" / "NO" / "SKIP"
    kelly_size_pct: float       # Portfoyun yuzdesi olarak onerilen pozisyon
    factors: list               # FactorScore listesi
    summary: str


# --- ALTIN / GUMUS FIYAT TOPLAYICI -------------------------------------------

class MetalPriceCollector:

    async def _fetch_goldapi(self, metal: str, session: aiohttp.ClientSession) -> Optional[MetalPrice]:
        if not GOLDAPI_KEY:
            return None
        try:
            url = f"https://www.goldapi.io/api/{metal}/USD"
            async with session.get(url, headers={"x-access-token": GOLDAPI_KEY}) as r:
                d = await r.json()
                return MetalPrice(
                    metal=metal,
                    price=float(d.get("price", 0)),
                    bid=float(d.get("bid", 0)),
                    ask=float(d.get("ask", 0)),
                    change_pct=float(d.get("chp", 0)),
                    source="goldapi.io",
                    timestamp=datetime.now(timezone.utc),
                )
        except Exception as e:
            log.warning(f"GoldAPI {metal}: {e}")
            return None

    async def _fetch_metalprice(self, metal: str, session: aiohttp.ClientSession) -> Optional[MetalPrice]:
        if not METALPRICE_KEY:
            return None
        try:
            url = f"https://api.metalpriceapi.com/v1/latest?api_key={METALPRICE_KEY}&base=USD&currencies={metal}"
            async with session.get(url) as r:
                d = await r.json()
                if not d.get("success"):
                    return None
                rate = d["rates"].get(metal, 0)
                price = 1.0 / rate if rate else 0
                return MetalPrice(
                    metal=metal,
                    price=price,
                    bid=price * 0.9995,
                    ask=price * 1.0005,
                    change_pct=0,
                    source="metalpriceapi.com",
                    timestamp=datetime.now(timezone.utc),
                )
        except Exception as e:
            log.warning(f"MetalPriceAPI {metal}: {e}")
            return None

    async def _fetch_fallback_yahoo(self, metal: str, session: aiohttp.ClientSession) -> Optional[MetalPrice]:
        """API key yoksa Yahoo Finance fallback"""
        symbol = "GC=F" if metal == "XAU" else "SI=F"
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=2d"
            async with session.get(url, headers={"User-Agent": "Mozilla/5.0"}) as r:
                d = await r.json()
                closes = d["chart"]["result"][0]["indicators"]["quote"][0]["close"]
                price = closes[-1]
                prev = closes[-2] if len(closes) > 1 else price
                chg_pct = ((price - prev) / prev * 100) if prev else 0
                return MetalPrice(
                    metal=metal,
                    price=float(price),
                    bid=float(price) * 0.9995,
                    ask=float(price) * 1.0005,
                    change_pct=float(chg_pct),
                    source="yahoo_finance",
                    timestamp=datetime.now(timezone.utc),
                )
        except Exception as e:
            log.warning(f"Yahoo fallback {metal}: {e}")
            return None

    async def get_price(self, metal: str) -> Optional[MetalPrice]:
        async with aiohttp.ClientSession(headers={"User-Agent": "arb-bot/2.0"}) as s:
            # Iki kaynaktan al, capraz dogrula
            p1, p2 = await asyncio.gather(
                self._fetch_goldapi(metal, s),
                self._fetch_metalprice(metal, s),
            )

            prices = [p for p in [p1, p2] if p and p.price > 0]

            if not prices:
                fallback = await self._fetch_fallback_yahoo(metal, s)
                if fallback:
                    return fallback
                return None

            if len(prices) == 2:
                # Capraz dogrulama: %1'den fazla fark varsa uyar
                diff_pct = abs(prices[0].price - prices[1].price) / prices[0].price * 100
                if diff_pct > 1.0:
                    log.warning(f"[WARN] {metal} fiyat tutarsizligi: {diff_pct:.2f}% fark")
                # Ortalama al
                avg_price = (prices[0].price + prices[1].price) / 2
                prices[0].price = avg_price
                prices[0].source = "goldapi+metalpriceapi (avg)"

            return prices[0]


# --- FRED MAKRO VERI TOPLAYICI ------------------------------------------------

class FREDCollector:

    async def _get_series_latest(self, series_id: str, session: aiohttp.ClientSession) -> Optional[float]:
        if not FRED_KEY:
            return None
        try:
            url = (
                f"https://api.stlouisfed.org/fred/series/observations"
                f"?series_id={series_id}&api_key={FRED_KEY}"
                f"&file_type=json&sort_order=desc&limit=2"
            )
            async with session.get(url) as r:
                d = await r.json()
                obs = d.get("observations", [])
                for o in obs:
                    if o["value"] != ".":
                        return float(o["value"])
        except Exception as e:
            log.warning(f"FRED {series_id}: {e}")
        return None

    async def get_snapshot(self) -> MacroSnapshot:
        snap = MacroSnapshot()
        if not FRED_KEY:
            log.warning("FRED_KEY eksik -- makro veriler atlaniyor")
            return snap

        async with aiohttp.ClientSession() as s:
            vals = await asyncio.gather(*[
                self._get_series_latest(sid, s)
                for sid in FRED_SERIES.values()
            ], return_exceptions=True)

        keys = list(FRED_SERIES.keys())
        result = {keys[i]: (vals[i] if not isinstance(vals[i], Exception) else None)
                  for i in range(len(keys))}

        snap.dxy            = result.get("DXY")
        snap.real_yield_10y = result.get("REAL_YIELD")
        snap.cpi_yoy        = result.get("CPI_YOY")
        snap.fed_rate       = result.get("FED_RATE")
        snap.m2_growth      = result.get("M2")
        snap.vix            = result.get("VIX")

        log.info(f"[MACRO] Makro: DXY={snap.dxy} | RealYield={snap.real_yield_10y}% | "
                 f"CPI={snap.cpi_yoy} | FedRate={snap.fed_rate}% | VIX={snap.vix}")
        return snap


# --- CFTC COT VERI TOPLAYICI --------------------------------------------------

class COTCollector:

    async def _fetch_cot(self, code: str, session: aiohttp.ClientSession, limit=53) -> list:
        """CFTC Public Reporting Environment API"""
        try:
            params = {
                "where": f"cftc_commodity_code='{code}'",
                "order_by": "report_date_as_yyyy_mm_dd DESC",
                "limit": limit,
                "dataset": "commitments_of_traders",
            }
            async with session.get(CFTC_URL, params=params) as r:
                if r.status != 200:
                    return []
                d = await r.json()
                return d.get("results", []) or d.get("records", [])
        except Exception as e:
            log.warning(f"CFTC COT {code}: {e}")
            return []

    def _calc_percentile(self, current: float, history: list) -> float:
        if not history:
            return 50.0
        below = sum(1 for v in history if v <= current)
        return (below / len(history)) * 100

    async def get_snapshot(self, metal: str) -> Optional[COTSnapshot]:
        code = GOLD_CFTC_CODE if metal == "XAU" else SILVER_CFTC_CODE
        async with aiohttp.ClientSession(headers={"User-Agent": "arb-bot/2.0"}) as s:
            records = await self._fetch_cot(code, s, limit=53)

        if not records:
            log.warning(f"COT verisi alinamadi: {metal}")
            return None

        latest = records[0]
        history = records[1:]

        try:
            comm_net = int(latest.get("comm_positions_long_all", 0)) - \
                       int(latest.get("comm_positions_short_all", 0))
            mm_net   = int(latest.get("noncomm_positions_long_all", 0)) - \
                       int(latest.get("noncomm_positions_short_all", 0))

            hist_comm = [
                int(r.get("comm_positions_long_all", 0)) -
                int(r.get("comm_positions_short_all", 0))
                for r in history
            ]
            hist_mm = [
                int(r.get("noncomm_positions_long_all", 0)) -
                int(r.get("noncomm_positions_short_all", 0))
                for r in history
            ]

            comm_pct = self._calc_percentile(comm_net, hist_comm)
            mm_pct   = self._calc_percentile(mm_net, hist_mm)

            snap = COTSnapshot(
                metal=metal,
                commercial_net=comm_net,
                managed_money_net=mm_net,
                commercial_net_pct=comm_pct,
                managed_money_pct=mm_pct,
                report_date=latest.get("report_date_as_yyyy_mm_dd", "unknown"),
            )
            log.info(f"[COT] COT {metal}: Comm={comm_net:+,} (%{comm_pct:.0f}) | "
                     f"MM={mm_net:+,} (%{mm_pct:.0f}) | Tarih={snap.report_date}")
            return snap
        except Exception as e:
            log.error(f"COT parse hatasi: {e}")
            return None


# --- TEKNIK INDIKATORLER -----------------------------------------------------

class TechnicalEngine:

    async def _get_history(self, metal: str, days=60) -> list:
        """Yahoo Finance'den gecmis fiyat verisi"""
        symbol = "GC=F" if metal == "XAU" else "SI=F"
        try:
            end = int(datetime.now(timezone.utc).timestamp())
            start = end - days * 86400
            url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
                   f"?interval=1d&period1={start}&period2={end}")
            async with aiohttp.ClientSession(headers={"User-Agent": "Mozilla/5.0"}) as s:
                async with s.get(url) as r:
                    d = await r.json()
                    closes = d["chart"]["result"][0]["indicators"]["quote"][0]["close"]
                    return [c for c in closes if c is not None]
        except Exception as e:
            log.warning(f"Teknik veri {metal}: {e}")
            return []

    def _rsi(self, prices: list, period=14) -> float:
        if len(prices) < period + 1:
            return 50.0
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains   = [max(c, 0) for c in changes[-period:]]
        losses  = [abs(min(c, 0)) for c in changes[-period:]]
        avg_g   = sum(gains) / period
        avg_l   = sum(losses) / period
        if avg_l == 0:
            return 100.0
        rs = avg_g / avg_l
        return 100 - (100 / (1 + rs))

    def _ema(self, prices: list, period: int) -> float:
        if len(prices) < period:
            return prices[-1] if prices else 0
        k = 2 / (period + 1)
        ema = sum(prices[:period]) / period
        for p in prices[period:]:
            ema = p * k + ema * (1 - k)
        return ema

    def _bollinger(self, prices: list, period=20) -> float:
        if len(prices) < period:
            return 0.0
        recent = prices[-period:]
        mid = sum(recent) / period
        std = math.sqrt(sum((p - mid)**2 for p in recent) / period)
        if std == 0:
            return 0.0
        upper = mid + 2 * std
        lower = mid - 2 * std
        current = prices[-1]
        # -1 (alt bant) ile +1 (ust bant) arasi normalize
        pos = (current - lower) / (upper - lower) * 2 - 1
        return max(-1.5, min(1.5, pos))

    def _atr_pct(self, prices: list, period=14) -> float:
        if len(prices) < 2:
            return 0.0
        trs = [abs(prices[i] - prices[i-1]) for i in range(1, min(period+1, len(prices)))]
        atr = sum(trs) / len(trs)
        return (atr / prices[-1]) * 100

    async def get_snapshot(self, metal: str) -> Optional[TechnicalSnapshot]:
        prices = await self._get_history(metal)
        if len(prices) < 20:
            log.warning(f"Teknik icin yeterli fiyat verisi yok: {metal}")
            return None

        rsi   = self._rsi(prices)
        ema20 = self._ema(prices, 20)
        ema50 = self._ema(prices, 50)
        bb    = self._bollinger(prices)
        atr   = self._atr_pct(prices)
        mom5  = ((prices[-1] - prices[-6]) / prices[-6] * 100) if len(prices) >= 6 else 0

        snap = TechnicalSnapshot(
            metal=metal,
            rsi_14=rsi,
            atr_pct=atr,
            bb_position=bb,
            ema_20_vs_50=(ema20 / ema50 - 1) * 100 if ema50 > 0 else 0,
            momentum_5d=mom5,
        )
        log.info(f"[TECH] Teknik {metal}: RSI={rsi:.1f} | BB={bb:.2f} | "
                 f"EMA20>50={snap.ema_20_vs_50:+.2f}% | Mom5d={mom5:+.2f}%")
        return snap


# --- OLASALIK MOTORU (KARAR KATMANI) ------------------------------------------

class ProbabilityEngine:
    """
    Her faktoru normalize edilmis skora (-1..+1) donusturur.
    Agirlikli ortalama ile birlesik yonlu skor hesaplar.
    Skor -> olasilik donusumu sigmoid ile yapilir.
    """

    # Faktor agirliklari (toplam = 1.0 olmali)
    WEIGHTS = {
        "macro_dxy":          0.20,  # DXY (en guclu negatif korelasyon)
        "macro_real_yield":   0.18,  # Reel getiri (firsat maliyeti)
        "cot_commercial":     0.16,  # "Akilli para" pozisyonlama
        "cot_managed_money":  0.10,  # Spekulatif pozisyonlama
        "technical_rsi":      0.10,  # Momentum asiriligi
        "technical_trend":    0.12,  # EMA trend
        "technical_bb":       0.08,  # Bollinger band konumu
        "macro_vix":          0.06,  # Risk istahi / safe-haven talebi
    }

    def _score_dxy(self, dxy: Optional[float]) -> FactorScore:
        """DXY yukselirse XAU/XAG duser (negatif korelasyon ~-0.7)"""
        if dxy is None:
            return FactorScore("macro_dxy", 0, 0, self.WEIGHTS["macro_dxy"], "Veri yok")
        # DXY 100 = notr, >105 cok guclu dolar, <95 zayif dolar
        score = -(dxy - 100) / 10  # DXY=110 -> score=-1, DXY=90 -> score=+1
        score = max(-1, min(1, score))
        desc = f"DXY={dxy:.1f} -> {'Zayif $ (altin +)' if score > 0 else 'Guclu $ (altin -)'}"
        return FactorScore("macro_dxy", dxy, score, self.WEIGHTS["macro_dxy"], desc)

    def _score_real_yield(self, ry: Optional[float]) -> FactorScore:
        """Reel getiri dusukse/negatifse altin cazibe kazanir"""
        if ry is None:
            return FactorScore("macro_real_yield", 0, 0, self.WEIGHTS["macro_real_yield"], "Veri yok")
        # Reel getiri: -2% -> score=+1, 0% -> score=0, +2% -> score=-1
        score = -ry / 2
        score = max(-1, min(1, score))
        desc = f"10Y Reel={ry:.2f}% -> {'Negatif reel getiri (altin +)' if score > 0 else 'Pozitif reel getiri (altin -)'}"
        return FactorScore("macro_real_yield", ry, score, self.WEIGHTS["macro_real_yield"], desc)

    def _score_cot_commercial(self, cot: Optional[COTSnapshot]) -> FactorScore:
        """Commercials extreme short = contrarian bullish sinyal"""
        if cot is None:
            return FactorScore("cot_commercial", 0, 0, self.WEIGHTS["cot_commercial"], "Veri yok")
        # %0-20 -> extreme short -> bullish (score=+1)
        # %80-100 -> extreme long -> bearish (score=-1)
        pct = cot.commercial_net_pct
        score = 1 - (pct / 50)  # %0->+1, %50->0, %100->-1
        score = max(-1, min(1, score))
        desc = (f"COT Comm %{pct:.0f} -> "
                f"{'Extreme short (contrarian bull!)' if pct < 20 else 'Extreme long (contrarian bear!)' if pct > 80 else 'Notr'}")
        return FactorScore("cot_commercial", pct, score, self.WEIGHTS["cot_commercial"], desc)

    def _score_cot_mm(self, cot: Optional[COTSnapshot]) -> FactorScore:
        """Managed money extreme long = asiri alim (negatif sinyal)"""
        if cot is None:
            return FactorScore("cot_managed_money", 0, 0, self.WEIGHTS["cot_managed_money"], "Veri yok")
        pct = cot.managed_money_pct
        # MM extreme long (>80) = asiri alim, duzeltme riski -> bearish
        score = -(pct / 50 - 1)
        score = max(-1, min(1, score))
        desc = f"COT MM %{pct:.0f} -> {'Asiri alim (bearish risk)' if pct > 80 else 'Asiri satim (bullish)' if pct < 20 else 'Notr'}"
        return FactorScore("cot_managed_money", pct, score, self.WEIGHTS["cot_managed_money"], desc)

    def _score_rsi(self, tech: Optional[TechnicalSnapshot]) -> FactorScore:
        if tech is None:
            return FactorScore("technical_rsi", 50, 0, self.WEIGHTS["technical_rsi"], "Veri yok")
        rsi = tech.rsi_14
        # RSI 30 = oversold bullish, RSI 70 = overbought bearish
        score = -(rsi - 50) / 30
        score = max(-1, min(1, score))
        desc = f"RSI={rsi:.1f} -> {'Asiri satim (bullish)' if rsi < 35 else 'Asiri alim (bearish)' if rsi > 65 else 'Notr'}"
        return FactorScore("technical_rsi", rsi, score, self.WEIGHTS["technical_rsi"], desc)

    def _score_trend(self, tech: Optional[TechnicalSnapshot]) -> FactorScore:
        if tech is None:
            return FactorScore("technical_trend", 0, 0, self.WEIGHTS["technical_trend"], "Veri yok")
        ema_diff = tech.ema_20_vs_50
        # EMA20 > EMA50: uptrend -> bullish
        score = ema_diff / 3   # 3% fark = maksimum skor
        score = max(-1, min(1, score))
        desc = f"EMA20 vs EMA50: {ema_diff:+.2f}% -> {'Uptrend' if ema_diff > 0 else 'Downtrend'}"
        return FactorScore("technical_trend", ema_diff, score, self.WEIGHTS["technical_trend"], desc)

    def _score_bb(self, tech: Optional[TechnicalSnapshot]) -> FactorScore:
        if tech is None:
            return FactorScore("technical_bb", 0, 0, self.WEIGHTS["technical_bb"], "Veri yok")
        bb = tech.bb_position
        # Alt banta yakin = oversold = bullish
        score = -bb / 1.5
        score = max(-1, min(1, score))
        desc = f"BB konum={bb:.2f} -> {'Alt bant (bullish)' if bb < -0.5 else 'Ust bant (bearish)' if bb > 0.5 else 'Orta'}"
        return FactorScore("technical_bb", bb, score, self.WEIGHTS["technical_bb"], desc)

    def _score_vix(self, macro: Optional[MacroSnapshot]) -> FactorScore:
        if macro is None or macro.vix is None:
            return FactorScore("macro_vix", 0, 0, self.WEIGHTS["macro_vix"], "Veri yok")
        vix = macro.vix
        # VIX yukselince safe-haven talebi artar -> altin bullish
        score = (vix - 20) / 20   # VIX=40 -> score=+1, VIX=20 -> score=0, VIX=10 -> score=-0.5
        score = max(-1, min(1, score))
        desc = f"VIX={vix:.1f} -> {'Yuksek korku (safe-haven talebi)' if vix > 25 else 'Dusuk korku'}"
        return FactorScore("macro_vix", vix, score, self.WEIGHTS["macro_vix"], desc)

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Skor -> olasilik (-1..+1 -> 0..1)"""
        return 1 / (1 + math.exp(-3 * x))

    def calculate(
        self,
        metal: str,
        macro: Optional[MacroSnapshot],
        cot: Optional[COTSnapshot],
        tech: Optional[TechnicalSnapshot],
        polymarket_price: float,
        direction: str = "UP",  # "UP" veya "DOWN"
    ) -> MetalDecision:

        factors = [
            self._score_dxy(macro.dxy if macro else None),
            self._score_real_yield(macro.real_yield_10y if macro else None),
            self._score_cot_commercial(cot),
            self._score_cot_mm(cot),
            self._score_rsi(tech),
            self._score_trend(tech),
            self._score_bb(tech),
            self._score_vix(macro),
        ]

        # Agirlikli bilesik skor - SADECE verisi olan faktorleri kullan
        # (veri yoksa score=0 olan faktorler composite'i sulandirmasin)
        active_factors = [f for f in factors if f.raw_value != 0]
        if active_factors:
            active_weight = sum(f.weight for f in active_factors)
            composite = sum(f.score * f.weight for f in active_factors) / active_weight
        else:
            composite = 0.0

        # "UP" yonu icin olasilik; "DOWN" icin 1 - prob
        raw_prob = self._sigmoid(composite)
        probability = raw_prob if direction == "UP" else (1 - raw_prob)

        # Kac faktorun verisi var? (guven skoru)
        data_coverage = len(active_factors) / len(factors)
        # Sinyal gucu ile guven carpani
        confidence = data_coverage * min(1.0, abs(composite) * 2)
        # Minimum guven: en az 3 faktor varsa ve sinyal gucluyse islem yap
        if len(active_factors) >= 3 and abs(composite) > 0.3:
            confidence = max(confidence, 0.20)

        # Edge hesapla
        edge = probability - polymarket_price
        min_edge = 0.10  # minimum %10 edge

        if abs(edge) < min_edge or confidence < 0.15:
            order_side = "SKIP"
            kelly_pct = 0.0
        elif edge > 0:
            order_side = "YES"
            # Yarim Kelly
            kelly_pct = min(0.10, (edge / (1 - polymarket_price)) * 0.5)
        else:
            order_side = "NO"
            kelly_pct = min(0.10, (abs(edge) / polymarket_price) * 0.5)

        # Ozet
        dir_tag = "[UP]" if direction == "UP" else "[DOWN]"
        summary = (
            f"{dir_tag} {metal} | Bilesik skor={composite:+.3f} | "
            f"Olasilik={probability:.1%} | Market={polymarket_price:.1%} | "
            f"Edge={edge:+.1%} | Guven={confidence:.1%} | "
            f"-> {order_side} (Kelly={kelly_pct:.1%})"
        )

        return MetalDecision(
            metal=metal,
            direction=direction,
            probability=probability,
            polymarket_price=polymarket_price,
            edge=edge,
            confidence=confidence,
            order_side=order_side,
            kelly_size_pct=kelly_pct,
            factors=factors,
            summary=summary,
        )


# --- ANA KULLANIM SINIFI ------------------------------------------------------

class XAUXAGEngine:
    def __init__(self):
        self.price_collector = MetalPriceCollector()
        self.fred            = FREDCollector()
        self.cot             = COTCollector()
        self.technical       = TechnicalEngine()
        self.prob_engine     = ProbabilityEngine()

    async def analyze(self, metal: str, polymarket_price: float, direction: str = "UP") -> MetalDecision:
        log.info(f"\n{'='*55}")
        log.info(f"[SCAN] {metal} Analizi baslatiliyor... Yon: {direction}")

        # Tum veri katmanlarini paralel cek
        price_data, macro, cot_snap, tech = await asyncio.gather(
            self.price_collector.get_price(metal),
            self.fred.get_snapshot(),
            self.cot.get_snapshot(metal),
            self.technical.get_snapshot(metal),
            return_exceptions=True
        )

        # Hata kontrolu
        for name, val in [("Price", price_data), ("Macro", macro),
                           ("COT", cot_snap), ("Technical", tech)]:
            if isinstance(val, Exception):
                log.error(f"{name} veri hatasi: {val}")

        price_data = None if isinstance(price_data, Exception) else price_data
        macro      = None if isinstance(macro, Exception) else macro
        cot_snap   = None if isinstance(cot_snap, Exception) else cot_snap
        tech       = None if isinstance(tech, Exception) else tech

        if price_data:
            log.info(f"[PRICE] {metal} Fiyat: ${price_data.price:,.2f} "
                     f"({price_data.change_pct:+.2f}%) [{price_data.source}]")

        decision = self.prob_engine.calculate(
            metal=metal,
            macro=macro,
            cot=cot_snap,
            tech=tech,
            polymarket_price=polymarket_price,
            direction=direction,
        )

        log.info(decision.summary)
        log.info(f"\n{'-'*55}")
        log.info("Faktor Detaylari:")
        for f in decision.factors:
            bar = "#" * int(abs(f.score) * 10)
            sign = "+" if f.score >= 0 else "-"
            log.info(f"  {f.name:<25} {sign}{bar:<10} ({f.score:+.2f}) | {f.description}")

        return decision


# --- STANDALONE TEST ----------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
    )

    async def test():
        engine = XAUXAGEngine()

        print("\n" + "="*60)
        print("XAU/XAG Olasilik Motoru -- Test Calismasi")
        print("="*60)

        # Ornek: Polymarket'te "XAU $3000 uzerinde kalir mi?" -> 0.65 fiyatli
        xau = await engine.analyze("XAU", polymarket_price=0.65, direction="UP")
        print(f"\n[RESULT] XAU Karar: {xau.order_side} | Edge: {xau.edge:+.1%}")

        # Ornek: XAG icin
        xag = await engine.analyze("XAG", polymarket_price=0.40, direction="UP")
        print(f"\n[RESULT] XAG Karar: {xag.order_side} | Edge: {xag.edge:+.1%}")

    asyncio.run(test())
