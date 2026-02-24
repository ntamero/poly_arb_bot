"""Hizli API testi - tum servisleri kontrol eder"""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from bot import PriceAggregator, PolymarketClient, NOAAClient, CFG

async def test_all():
    print("=" * 60)
    print("POLYMARKET ARB BOT - API TEST")
    print("=" * 60)

    # 1. BTC Fiyatlari
    print("\n[1] BTC FIYATLARI")
    print("-" * 40)
    agg = PriceAggregator()
    prices = await agg.get_all_prices()
    if prices:
        for name, price in prices.items():
            print(f"  {name:>10}: ${price:,.2f}")
        avg = sum(prices.values()) / len(prices)
        print(f"  {'ORTALAMA':>10}: ${avg:,.2f}")
        print(f"  STATUS: OK ({len(prices)} borsa)")
    else:
        print("  STATUS: FAIL - Hicbir borsadan veri alinamadi")

    # 2. Polymarket Marketleri
    print("\n[2] POLYMARKET MARKETLERI")
    print("-" * 40)
    poly = PolymarketClient()

    weather = await poly.get_weather_markets()
    print(f"  Hava marketleri: {len(weather)}")
    for m in weather[:3]:
        q = m.get("question", "?")[:60]
        liq = float(m.get("liquidity", 0) or 0)
        print(f"    - {q} [${liq:,.0f}]")

    btc = await poly.get_btc_markets()
    print(f"  BTC marketleri: {len(btc)}")
    for m in btc[:3]:
        q = m.get("question", "?")[:60]
        liq = float(m.get("liquidity", 0) or 0)
        tokens = m.get("tokens", [])
        yes_t = next((t for t in tokens if t.get("outcome", "").upper() == "YES"), None)
        price = float(yes_t.get("price", 0.5) or 0.5) if yes_t else 0.5
        print(f"    - {q} [YES={price:.2f} | ${liq:,.0f}]")

    all_markets = await poly.get_all_active_markets(limit=20)
    print(f"  Tum aktif (top 20): {len(all_markets)}")
    for m in all_markets[:5]:
        q = m.get("question", "?")[:55]
        liq = float(m.get("liquidity", 0) or 0)
        print(f"    - {q} [${liq:,.0f}]")

    # 3. NOAA Hava Durumu
    print("\n[3] NOAA HAVA DURUMU")
    print("-" * 40)
    noaa = NOAAClient()
    print(f"  NOAA Token: {'VAR' if CFG.noaa_token else 'YOK'}")
    # Sadece NYC test et (hizli olsun)
    forecast = await noaa.get_hourly_forecast(40.7128, -74.0060, "NYC")
    if forecast:
        print(f"  NYC: Max={forecast['max_temp_f']}F ({(forecast['max_temp_f']-32)*5/9:.1f}C)")
        print(f"        Min={forecast['min_temp_f']}F | Avg={forecast['avg_temp_f']}F")
        print(f"        Guven={forecast['confidence']}% | Veri={forecast['forecast_count']} saat")
        print(f"  STATUS: OK")
    else:
        print("  STATUS: FAIL - NOAA verisi alinamadi (VPN gerekebilir)")

    # 4. Konfigurasyon
    print("\n[4] KONFIGURASYON")
    print("-" * 40)
    print(f"  Mod: {'SIMULASYON' if CFG.simulation_mode else 'CANLI'}")
    print(f"  Baslangic bakiye: ${CFG.initial_balance:,.0f}")
    print(f"  Min edge: {CFG.min_edge_pct}%")
    print(f"  Max order: ${CFG.max_order_usd}")
    print(f"  Gunluk limit: ${CFG.max_daily_usd}")
    print(f"  Min likidite: ${CFG.min_liquidity:,.0f}")
    print(f"  Poly key: {'VAR' if CFG.poly_private_key else 'YOK'}")

    print("\n" + "=" * 60)
    print("TEST TAMAMLANDI")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_all())
