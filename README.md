# ⚽ FPL Yapay Zeka Kahini

Bu proje, **Fantasy Premier League (FPL)** oyuncuları için **XGBoost** makine öğrenmesi modeli ve **Understat xG** verilerini kullanarak puan tahminleri yapan ve transfer önerileri sunan otonom bir web uygulamasıdır.



## 🚀 Özellikler

* **Dinamik GW Takibi:** FPL API üzerinden sıradaki haftayı (Gameweek) otomatik olarak algılar.
* **Kişisel Transfer Sihirbazı:** Team ID'nizi girerek mevcut kadronuz için yapay zeka destekli transfer önerileri alabilirsiniz.
* **Rüya Takımlar:** Hem önümüzdeki hafta (Short Term) hem de sonraki 5 hafta (Long Term) için optimize edilmiş kadrolar sunar.
* **Model Laboratuvarı:** Modelin başarı oranını (R²), hata payını (RMSE) ve hangi istatistiklerin puanı daha çok etkilediğini analiz eder.
* **Otonom Güncelleme:** Her gün otomatik olarak güncellenen sakatlık, fiyat ve form verileri.

## 📁 Proje Yapısı

* `app.py`: Streamlit tabanlı web arayüzü.
* `updater.py`: Veri çekme, model eğitme ve optimizasyon süreçlerini yöneten ana script.
* `src/data_loader.py`: FPL ve Understat API entegrasyonu.
* `src/model.py`: XGBoost tabanlı puan tahmin modeli.
* `src/optimizer.py`: Kadro optimizasyonu ve transfer algoritması.

## 🛠️ Kurulum ve Çalıştırma

### 1. Depoyu Klonlayın
```bash
git clone [https://github.com/anilaltan/fpl-ai-bot.git](https://github.com/anilaltan/fpl-ai-bot.git)
cd fpl-ai-bot
```
### 2. Sanal Ortamı Kurun ve Kütüphaneleri Yükleyin
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
### 3. Verileri Güncelleyin ve Modeli Eğitin
```bash
python3 updater.py
```
### 4. Uygulamayı Başlatın

#### Manuel Çalıştırma
```bash
streamlit run app.py
```

#### Sistem Servisi Olarak Çalıştırma (Önerilen)
```bash
# Servis yönetimi
./manage_service.sh start    # Başlat
./manage_service.sh stop     # Durdur
./manage_service.sh restart  # Yeniden başlat
./manage_service.sh status   # Durum kontrolü
./manage_service.sh logs     # Log görüntüleme
```

**Servis Özellikleri:**
- ✅ SSH bağlantısı kapansa bile çalışmaya devam eder
- ✅ Sunucu restart olursa otomatik başlar
- ✅ Hata durumunda otomatik yeniden başlatılır
- ✅ Port: 8502
- ✅ URL: `http://sunucu-ip:8502`

### 🔄 Otomatik Güncellemeler

**Her Gece Saat 02:00'da:**
- ✅ FPL verileri otomatik güncellenir
- ✅ Model yeniden eğitilir
- ✅ Streamlit uygulaması yeniden başlatılır
- ✅ Log dosyaları tutulur ve 7 günden eski olanlar temizlenir

**Cron Job:** `0 2 * * * /root/fpl-test/scripts/nightly_update.sh`

**Manuel Güncelleme:**
```bash
./scripts/nightly_update.sh  # Anında güncelleme
```

**Log Kontrolü:**
```bash
ls logs/                    # Güncelleme logları
tail logs/nightly_update_*.log  # Son logu görüntüle
```

## 🤖 Model Performansı
Model, **Ensemble Learning** yaklaşımı kullanır:
- **Technical Score (50%)**: xG, xA, Form - Geleneksel istatistikler
- **Market Score (30%)**: Bahis oranları - Piyasa zekası
- **Tactical Score (20%)**: Eşleşme + Duran top - Kısa vadeli taktik

Güncel başarı metriklerine uygulamanın Model Lab sekmesinden ulaşabilirsiniz.

## 📊 Özellikler
- **Ensemble Model**: 3 uzman modelinin ağırlıklı oylaması
- **Chip Strategy**: Wildcard, Triple Captain, Bench Boost önerileri
- **Walk-Forward Backtesting**: Veri sızıntısı önleme testi
- **Real-Time Data**: FPL API entegrasyonu
- **Auto Team Import**: FPL Team ID ile otomatik takım çekme

Not: Bu proje eğitim amaçlıdır ve yatırım tavsiyesi içermez.
