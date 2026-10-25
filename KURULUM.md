# GAZAP N2O Enjektör Uygulaması: Kurulum ve Paylaşım

Klasördeki dosyalar:

| Dosya | Ne işe yarar |
| --- | --- |
| `app.py` | Arayüz (Streamlit) |
| `n2o_enjektor_v7_2.py` | Hesap motoru; Spyder'da kullandığın kodun aynısı, değiştirilmedi |
| `requirements.txt` | Gerekli Python paketleri |

## 1. Kendi bilgisayarında çalıştırma

```
pip install -r requirements.txt
streamlit run app.py
```

Tarayıcıda `http://localhost:8501` açılır. Aynı Wi-Fi'daki biri, terminalde görünen "Network URL" adresinden bağlanabilir.

## 2. İnternete koyup WhatsApp'tan link paylaşma (ücretsiz)

1. github.com'da hesap aç, yeni bir repo oluştur (ör. `gazap-n2o-enjektor`) ve bu üç dosyayı yükle (`Add file > Upload files`).
2. share.streamlit.io'ya GitHub hesabınla gir, **Create app** de.
3. Repo'yu ve dalı (`main`) seç; ana dosya: `app.py`.
4. **Advanced settings** içinden Python sürümünü **3.12** seç (CoolProp için).
5. **Deploy**. İlk kurulum birkaç dakika sürer.
6. Çıkan `https://...streamlit.app` linkini WhatsApp'tan paylaş. Telefondan da açılır, kurulum gerekmez.

Kodu güncellemek için GitHub'daki dosyayı değiştirmen yeterli; uygulama kendini yeniler.

## Bilmen gerekenler

- **Gizlilik:** Uygulama varsayılan olarak linki olan herkese açıktır. Tasarım verisini yalnız ekip görsün istiyorsan Streamlit Cloud'daki uygulama ayarlarından görüntüleyicileri e-posta ile sınırlayabilirsin.
- **Uyku:** Ücretsiz uygulama bir süre kullanılmazsa uyur. İlk açılışta "uyandır" butonuna basılır, ~1 dakika sürer.
- **Hız:** Bir hesap ~30–40 s sürer. Aynı girdiler önbelleğe alındığı için ikinci kez anında gelir; varsayılan girdiler ilk açan kişiden sonra herkese hızlı açılır.
- **Girdi değiştirme:** Sol paneldeki değerleri değiştirip **Hesapla**'ya bas. Geometri seçimi (orifis/annulus) hemen uygulanır.
- **Kalibrasyon:** Sol paneldeki "Kalibrasyon testleri" tablosuna ölçümlerini satır satır gir; boş hücreli satırlar yok sayılır.
