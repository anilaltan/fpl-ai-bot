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
```bash
streamlit run app.py
```
🤖 Model Performansı
Model, oyuncu dakikalarını, xG (Beklenen Gol), xA (Beklenen Asistan) ve fikstür zorluklarını analiz ederek eğitilmiştir. Güncel başarı metriklerine uygulamanın Model Lab sekmesinden ulaşabilirsiniz.

Not: Bu proje eğitim amaçlıdır ve yatırım tavsiyesi içermez.
