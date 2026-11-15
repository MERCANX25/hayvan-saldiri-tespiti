# Hayvan Saldırısı Tespiti (Ses Tabanlı)

Ders projesi: hayvan seslerinden (hırlama, kükreme, çığlık vb.) saldırı ya da agresyon tespiti.

## İçerik

- `ses-modulu/`: veri indirme, Colab eğitim defteri, canlı demo ve adım adım rehber
  - `veri_seti_olustur.py`: VGGSound ve ESC-50'den ses klipleri indirir
  - `hayvan_ses_egitim.ipynb`: Google Colab'da eğitim (YAMNet + küçük sınıflandırıcı)
  - `canli_demo.py`: eğitilmiş modelle bilgisayarın sesini canlı dinler
  - `yamnet_test.py`: eğitim olmadan YAMNet ile hızlı deneme
  - `ANLATIM_RAPORU.md`: baştan sona anlatım
  - `test-videolari.md`: test için VGGSound YouTube bağlantıları
- `raporlar/`: literatür taraması ve veri stratejisi raporları

## Kurulum

```
pip install -r ses-modulu/requirements.txt
python ses-modulu/veri_seti_olustur.py --esc50
```

Veri ve model dosyaları (`veri/`, `model/`) repoya eklenmez.
