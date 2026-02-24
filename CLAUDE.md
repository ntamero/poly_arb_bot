# Polymarket Arbitraj Botu — Claude Code Görev Tanımı

## Bu proje nedir?
Hava durumu (NOAA) ve BTC fiyat arbitrajını otomatik olarak Polymarket'te
işleme dönüştüren bir Python botudur.

## Başlatma
```bash
# 1. Kurulum
pip install -r requirements.txt

# 2. .env oluştur
cp .env.example .env
# .env dosyasını düzenle: private key ve adresi gir

# 3. Botu çalıştır
python src/bot.py
```

## Görevler (Claude Code'a)

### Analiz modu (order açmadan):
```
src/bot.py dosyasını çalıştır, fırsatları raporla ama
CFG.max_daily_usd = 0 yaparak order açma.
```

### Canlı mod:
```
src/bot.py dosyasını çalıştır.
Logları izle ve her 5 dakikada döngü yaptığını doğrula.
Hata olursa düzelt.
```

### Backtest:
```
Son 7 günün Polymarket weather market kapanış fiyatlarını çek,
NOAA tahminleri ile karşılaştır, stratejinin win rate'ini hesapla.
```

## Strateji Özeti
- **Hava:** NOAA saatlik tahmin → Polymarket sıcaklık bucket marketleri
  - NOAA güveni yüksek, market yanlış fiyatlamışsa al
  - Edge eşiği: %15
- **BTC:** 4 borsadan ortalama fiyat → Polymarket BTC tahmin marketleri
  - Gerçek fiyat ile implied probability arasındaki farkı yakala
  - Edge eşiği: %15

## Risk Limitleri
- MAX_ORDER_USD: tek işlem maksimum
- MAX_DAILY_USD: günlük limit
- MIN_LIQUIDITY: sadece derin marketler
- MIN_EDGE_PCT: minimum avantaj eşiği

## Dosya Yapısı
```
polymarket-arb-bot/
├── src/bot.py          ← Ana bot
├── .env                ← API keys (git'e commit etme!)
├── .env.example        ← Şablon
├── requirements.txt
├── CLAUDE.md           ← Bu dosya
└── logs/
    ├── bot.log         ← Tüm loglar
    └── trades.json     ← İşlem geçmişi (JSON)
```
