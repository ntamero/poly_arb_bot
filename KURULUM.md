# Polymarket Arbitraj Botu — Kurulum Rehberi

## Gereksinimler
- Python 3.11+
- Polygon ağında USDC bakiyesi olan MetaMask cüzdanı

---

## ADIM 1: Dosyaları İndir

```bash
# Proje klasörüne git
cd polymarket-arb-bot

# Bağımlılıkları yükle
pip install -r requirements.txt
```

---

## ADIM 2: .env Dosyasını Oluştur

```bash
cp .env.example .env
```

`.env` dosyasını aç ve şunları doldur:

| Değişken | Nereden Alınır |
|----------|----------------|
| `POLY_PRIVATE_KEY` | MetaMask → ··· → Hesap Detayları → Private Key'i Dışa Aktar |
| `POLY_ADDRESS` | MetaMask cüzdan adresin (0x...) |
| `MAX_ORDER_USD` | Tek işlemde harcamak istediğin max dolar |
| `MAX_DAILY_USD` | Günlük toplam limit |
| `MIN_EDGE_PCT` | Minimum avantaj % (öneri: 15) |

**Borsa API key'leri isteğe bağlı** — fiyat çekme halka açık API ile de çalışır.

---

## ADIM 3: Cüzdan Hazırlığı

1. MetaMask'ta Polygon ağını ekle:
   - Network: Polygon Mainnet
   - RPC: https://polygon-rpc.com
   - Chain ID: 137

2. USDC transfer et (Polygon USDC)
   - Coinbase, Binance veya OKX'ten Polygon ağında USDC çek
   - Başlangıç için $50-100 yeterli

3. Polymarket'te approval ver:
   - https://polymarket.com adresine git
   - Cüzdanını bağla → Trade et → Approval işlemlerini onayla

---

## ADIM 4: Claude Code ile Çalıştır

### Seçenek A: Doğrudan terminal
```bash
python src/bot.py
```

### Seçenek B: Claude Code ile (önerilen)
```bash
# Claude Code'u başlat
claude

# Claude Code içinde söyle:
> CLAUDE.md dosyasını oku ve botu başlat. Logları izle.
```

---

## Logları İzle

```bash
# Canlı log izle
tail -f logs/bot.log

# İşlem geçmişi
cat logs/trades.json | python -m json.tool
```

---

## Risk Uyarıları

⚠️ **Prediction market'ler spekülatif yatırımlardır.**
⚠️ **Kaybetmeyi göze alamayacağın parayla işlem yapma.**
⚠️ **Polymarket, Türkiye dahil bazı ülkelerde kısıtlı olabilir.**
⚠️ **Private key'ini asla kimseyle paylaşma, .env'i git'e commit etme.**

---

## Sorun Giderme

| Hata | Çözüm |
|------|-------|
| `py-clob-client` import hatası | `pip install py-clob-client` |
| NOAA verisi gelmiyor | VPN dene veya 5dk bekle |
| Order rejected | Cüzdanda USDC var mı? Approval verildi mi? |
| Fırsat bulunamıyor | `MIN_EDGE_PCT`'yi düşür veya `MIN_LIQUIDITY`'yi azalt |
