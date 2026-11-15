# Ses ile Hayvan Saldırısı Tespiti: Adım Adım Anlatım

*Bu rapor projeyi hiç bilmeyen birinin de anlayabileceği şekilde yazıldı. Hem kendiniz için rehber, hem de sunumda anlatacaklarınızın taslağı olarak kullanabilirsiniz.*

---

## 1. Bir cümleyle proje

Bir videonun ya da ortamın **sesini dinleyip** içinde hayvan saldırısı ya da agresyonu olup olmadığını söyleyen bir yapay zeka sistemi. Hayvanın hayvana ya da insana saldırısı fark etmez; sistem sese bakar.

---

## 2. Sistemin genel yapısı

```
                         ┌──────────────── EĞİTİM (Colab) ────────────────┐
YouTube (VGGSound)  ──►  │ 10 sn'lik      ──►  YAMNet      ──►  Bizim      │ ──► model klasörü
veri_seti_olustur.py     │ ses klipleri        (hazır,          küçük ağ   │     (Drive'a kaydedilir)
(bilgisayarınızda)       │ + etiketler         özellik          (eğitilir) │
                         │                     çıkarıcı)                   │
                         └─────────────────────────────────────────────────┘

                         ┌──────────────── DEMO (bilgisayarınızda) ───────┐
YouTube videosu     ──►  │ Hoparlörden   ──►  YAMNet  ──►  Bizim ağ  ──►  │ ──►  normal / AGRESYON /
(tarayıcıda çalar)       │ çıkan ses           (aynı)       (eğitilmiş)   │      !!! SALDIRI OLASI !!!
                         │ (canli_demo.py)                  + alarm       │
                         └─────────────────────────────────────────────────┘
```

**Dosyalar ve görevleri:**

| Dosya | Nerede çalışır | Ne yapar |
|---|---|---|
| `veri_seti_olustur.py` | Bilgisayarınız | YouTube'dan ses kliplerini indirip veri setini oluşturur |
| `hayvan_ses_egitim.ipynb` | Google Colab | Modeli eğitir, test eder, grafikleri çizer, modeli kaydeder |
| `canli_demo.py` | Bilgisayarınız | Eğitilmiş modelle YouTube sesini canlı dinler ve alarm verir |
| `yamnet_test.py` | Bilgisayarınız | (İsteğe bağlı) Eğitimsiz YAMNet ile hızlı deneme |
| `requirements.txt` | Bilgisayarınız | Gerekli Python paketlerinin listesi |

---

## 3. Bilmeniz gereken temel kavramlar

### 3.1 Ses bilgisayarda nasıl durur?
Ses, havadaki titreşimdir. Mikrofon bu titreşimi saniyede binlerce kez ölçüp sayıya çevirir. **16 kHz** demek saniyede 16.000 ölçüm demektir. Yani 10 saniyelik bir klip, 160.000 sayıdan oluşan bir listedir. **Mono** demek tek kanal (sağ ve sol ayrı değil) demektir.

### 3.2 Spektrogram: Sesin resmi
Ham sayı listesine bakarak "bu hırlama" demek zordur. Bu yüzden ses bir **spektrograma** çevrilir:
- Yatay eksen **zaman**,
- Dikey eksen **frekans** (sesin ince ya da kalın olması),
- Renk **ses gücü**.

Hırlama, alt frekanslarda yoğun ve titreşimli bir desen; çığlık, üst frekanslarda parlak bir bant; havlama ise kısa ve kesik lekeler bırakır. Böylece ses tanıma problemi bir **görüntü tanıma** problemine dönüşür. Kavga modülünde görüntüyle çalıştığınız için bu mantık size tanıdık gelecek.

### 3.3 YAMNet nedir?
Google'ın **2 milyon YouTube klibiyle** (AudioSet) eğittiği hazır bir ses modeli. 521 farklı sesi tanır: köpek havlaması, hırlama, çığlık, yağmur, müzik vb. İçinde **MobileNet** adında hafif bir evrişimli sinir ağı (CNN) vardır.

### 3.4 Gömü (embedding) nedir?
YAMNet'e bir ses verdiğimizde, her **0,48 saniye** için **1024 sayıdan oluşan bir vektör** de verir. Bu, sesin "parmak izi" ya da özeti gibidir. Birbirine benzeyen seslerin vektörleri de birbirine yakın olur. Biz YAMNet'in kendi 521 sınıfını değil, bu vektörleri kullanıyoruz.

### 3.5 Transfer öğrenme
Sıfırdan bir ses modeli eğitmek milyonlarca klip ve günlerce GPU ister. Bunun yerine YAMNet'in öğrendiği bilgiyi **ödünç alıyoruz**: YAMNet'i değiştirmeden (dondurarak) sadece onun üstüne küçük bir ağ ekleyip **bu küçük ağı** kendi verimizle eğitiyoruz. Buna **transfer öğrenme** denir. Kavga modülünde önceden eğitilmiş bir görüntü modelinin üzerine katman eklemek de aynı fikirdir.

### 3.6 Bizim küçük ağımız
```
1024 sayı  ──►  256 nöron (ReLU)  ──►  Dropout %30  ──►  4 çıktı (Softmax)
(YAMNet'ten)                                              saldırgan / kurban / normal_hayvan / arka_plan
```
- **ReLU:** Negatif değerleri sıfırlar; ağın karmaşık ilişkileri öğrenmesini sağlar.
- **Dropout:** Eğitim sırasında nöronların %30'unu rastgele kapatır, ağın ezber yapmasını zorlaştırır.
- **Softmax:** 4 çıktıyı olasılığa çevirir, toplamları 1 olur. Örneğin: saldırgan 0,82, kurban 0,05, normal 0,10, arka plan 0,03.

---

## 4. Veri seti

### 4.1 Kaynak: VGGSound
Oxford Üniversitesi'nin hazırladığı, YouTube videolarından alınmış **200.000'den fazla** 10 saniyelik etiketli klip. Liste "video ID + başlangıç saniyesi + etiket" şeklindedir; sesler YouTube'dan indirilir. Lisansı CC BY 4.0.

**İkinci kaynak: ESC-50.** 2.000 kliplik (5 sn), 50 sınıflı bir çevresel ses veri seti. GitHub'dan doğrudan indiği için YouTube engeline takılmaz. Normal hayvan (köpek, kedi, domuz, inek, horoz, tavuk, koyun) ve arka plan (yağmur, rüzgar, adım, gülme, alkış, korna, motor, tren, kapı, klavye) seslerini buradan alıyoruz. Lisansı CC BY-NC (ticari olmayan kullanım serbest).

> **Dikkat (kaynak yanlılığı):** Saldırgan sesler hep VGGSound'dan, normal sesler çoğunlukla ESC-50'den gelirse model "saldırıyı" değil "hangi veri setinden geldiğini" öğrenebilir. Bunu azaltmak için normal ve arka plan seslerinden de bir miktar VGGSound klibi ekliyoruz. Raporda bu riski belirtmek iyi olur.

### 4.2 Neden 4 grup?

| Grup | İçindeki sesler | Neden gerekli |
|---|---|---|
| **saldirgan** | köpek hırlaması, kedi hırlaması, kedi tıslaması, aslan kükremesi, aslan hırlaması | Asıl yakalamak istediğimiz sesler |
| **kurban** | insan çığlığı, köpek inlemesi | Saldırıya uğrayanın sesi; "gerçekten saldırı var mı" kanıtı |
| **normal_hayvan** | normal havlama, uluma, miyavlama, mırlama, inek, domuz, at, horoz, koyun | Model "her hayvan sesi saldırıdır" demesin diye |
| **arka_plan** | konuşma, çocuk bağırışı, kalabalık, araba, rüzgar, yağmur, gitar | Model "her gürültü saldırıdır" demesin diye |

Sadece saldırı sesleriyle eğitilen bir model her sese "saldırı" der. **Normal ve arka plan sesleri**, modelin neyin saldırı **olmadığını** öğrenmesi için şarttır. Kavga modülündeki "kavga / kavga değil" ayrımının aynısıdır.

### 4.3 Eğitim / doğrulama / test ayrımı
| Bölüm | Oran | Ne için |
|---|---|---|
| Eğitim | %64 | Model bunlardan öğrenir |
| Doğrulama | %16 | Eğitim sırasında ezber kontrolü |
| Test | %20 | En sonda, modelin hiç görmediği kliplerle ölçüm. **Rapordaki sonuçlar buradan.** |

Aynı YouTube videosundan gelen klipler hep aynı bölüme konur. Yoksa model test sırasında "bu videoyu zaten duymuştum" diyerek hile yapmış olur.

---

## 5. Yapılacaklar: Adım adım

### Adım 1: Veriyi indir (bilgisayarınızda)
1. `hayvan-ses` klasöründe terminali açın.
2. Paketleri kurun: `pip install -r requirements.txt`
3. ESC-50'yi indirin (YouTube gerekmez, birkaç dakika): `python veri_seti_olustur.py --esc50`
4. VGGSound'dan saldırgan ve kurban seslerini yavaşça indirin: `python veri_seti_olustur.py --adet 40 --gruplar saldirgan,kurban --paralel 1`
5. Normal ve arka plan seslerinden de biraz VGGSound ekleyin (kaynak yanlılığına karşı): `python veri_seti_olustur.py --adet 10 --gruplar normal_hayvan,arka_plan --paralel 1`
6. Sonunda klasörde `veri` adında bir klasör oluşur. İçinde `ses` klasörü (WAV dosyaları) ve `metadata.csv` vardır.

**YouTube "Sign in to confirm you're not a bot" derse:** Kısa sürede çok indirme yapıldığı için YouTube geçici olarak engellemiştir. Betik 5 engelden sonra kendini durdurur. 30-60 dakika bekleyip aynı komutu tekrar çalıştırın (kaldığı yerden devam eder). Ya da Firefox'ta YouTube'a giriş yapıp komutun sonuna `--tarayici firefox` ekleyin.

### Adım 2: Veriyi Drive'a yükle
1. drive.google.com'u açın.
2. **Drive'ım** içinde `hayvan-ses` adında bir klasör oluşturun.
3. Bilgisayardaki `veri` klasörünü sürükleyip bu klasörün içine bırakın.
4. Sonuç: `Drive'ım / hayvan-ses / veri / ses / ...` ve `Drive'ım / hayvan-ses / veri / metadata.csv`

### Adım 3: Colab'da eğit
1. colab.research.google.com'u açın.
2. **Dosya > Not defteri yükle** deyip `hayvan_ses_egitim.ipynb`'yi seçin.
3. Üst menüden **Çalışma zamanı > Tümünü çalıştır** deyin (ya da her hücrede ▶'ye basın).
4. İlk hücrede Google hesabınıza izin vermenizi isteyecek; izin verin.
5. Bitince Drive'daki `hayvan-ses/model` klasöründe 4 dosya oluşur.

GPU gerekmez; küçük ağ CPU'da da birkaç dakikada eğitilir. En uzun süren kısım 8. adımdaki özellik çıkarmadır.

### Adım 4: Canlı demo (bilgisayarınızda)
1. Drive'dan `model` klasörünü indirip `hayvan-ses` klasörünün içine koyun.
2. Ses çıkışlarını görün: `python canli_demo.py --cihazlar`
3. YouTube'da bir video açın ve çalıştırın: `python canli_demo.py --canli --cihaz NUMARA`

---

## 6. Colab defterinde ne göreceksiniz?

| Bölüm | Ne yapar | Ekranda ne çıkar |
|---|---|---|
| 1-3 | Drive bağlama, kurulum, ayarlar | İzin penceresi, kurulum mesajları |
| 4. Veri setini yükle | metadata.csv'yi okur | Grup ve sınıf başına klip sayısı tablosu, çubuk grafik |
| 5. Örnekleri dinle | Her gruptan bir ses | Oynatılabilir ses çalarlar |
| 6. Spektrogramlar | Sesleri resme çevirir | 4 renkli spektrogram (sunum için güzel bir görsel) |
| 7. YAMNet'i yükle | Hazır modeli indirir | "YAMNet sınıf sayısı: 521" |
| 8. Özellik çıkar | Her klipten 1024'lük vektörler | İlerleme: "50/2500 klip işlendi" |
| 9. Ayırma | Eğitim/doğrulama/test | Her gruptan her bölümde kaç klip olduğu tablosu |
| 10. Model | Ağı kurar | Katman tablosu (yaklaşık 263 bin parametre) |
| 11. Eğitim | Ağı eğitir | Her tur için kayıp ve doğruluk, 2 grafik |
| 12. Test | Sonuçları ölçer | Doğruluk, F1, sınıf raporu, karışıklık matrisi |
| 13. Karşılaştırma | Eğitimsiz YAMNet ile kıyas | "Eğitim öncesi / sonrası" tablosu |
| 14. Hata analizi | Alt sınıf başarıları | Hangi seste zorlandığı (ör. kedi tıslaması) |
| 15. Kaydet | Modeli Drive'a yazar | Kaydedilen dosyaların listesi |
| 16. Demo | Ses zaman çizelgesi | 4 renkli olasılık eğrisi, alarm anları turuncu/kırmızı gölgeli |
| 17. Kendi dosyanız | Yüklediğiniz sesi analiz eder | Aynı grafik, sizin sesiniz için |

### 6.1 Eğitim grafikleri nasıl okunur?
- **İyi durum:** Eğitim ve doğrulama eğrileri birlikte iner (kayıp) ya da birlikte çıkar (doğruluk).
- **Ezber (overfitting):** Eğitim doğruluğu artmaya devam ederken doğrulama doğruluğu durur ya da düşer. `EarlyStopping` bunu fark edip eğitimi durdurur ve en iyi noktaya döner.

---

## 7. Sonuçlar nasıl yorumlanır?

Örnek üzerinden anlatalım. Testte 100 gerçek "saldırgan" klip olsun:

| Ölçüt | Soru | Örnek |
|---|---|---|
| **Recall (duyarlılık)** | Gerçek saldırgan seslerin kaçını yakaladık? | 100'ün 85'ini yakaladıysak recall = 0,85 |
| **Precision (kesinlik)** | "Saldırgan" dediklerimizin kaçı gerçekten saldırgan? | 95 kez "saldırgan" dedik, 85'i doğruysa precision = 0,89 |
| **F1** | İkisinin dengeli ortalaması | yaklaşık 0,87 |
| **Accuracy (doğruluk)** | Tüm kliplerin yüzde kaçı doğru? | Gruplar dengesizse yanıltıcı olabilir, bu yüzden F1 de raporlanır |

- **Recall düşükse** sistem saldırıları kaçırıyor (tehlikeli).
- **Precision düşükse** sistem çok yanlış alarm veriyor (can sıkıcı).

**Karışıklık matrisi:** Satırlar gerçek grup, sütunlar tahmin. Köşegendeki koyu kutular doğru tahminler. Örneğin "saldırgan" satırında "normal_hayvan" sütunu doluysa model bazı hırlamaları normal havlama sanıyor demektir. Sunumda bu matrisi gösterip "en çok şunlar karışıyor, çünkü..." diye yorum yapmak çok iyi izlenim bırakır.

> Not: Defteri ben sahte (yapay) seslerle baştan sona çalıştırıp hatasız bittiğini doğruladım. **Gerçek sonuçlar sizin verinizle Colab'da çıkacak.** Rapora o sayıları yazacağız.

---

## 8. Alarm mantığı

Model her yarım saniyede 4 olasılık verir. Ani sıçramalar yanlış alarma yol açmasın diye **son 3 pencerenin ortalaması** alınır. Sonra:

| Durum | Koşul | Anlamı |
|---|---|---|
| `normal` | Saldırgan olasılığı < 0,5 | Sorun yok |
| `AGRESYON` | Saldırgan olasılığı ≥ 0,5 | Bir hayvan hırlıyor, tıslıyor ya da kükrüyor |
| `!!! SALDIRI OLASI !!!` | Saldırgan sesten sonraki 4 saniye içinde kurban olasılığı ≥ 0,5 | Agresyonu bir çığlık ya da inleme takip etti: büyük ihtimalle saldırı |

**Neden iki seviye?** Hırlama her zaman saldırı değildir; köpek sadece uyarıyor olabilir. Agresyon sesinin hemen ardından gelen bir çığlık ise saldırının güçlü bir işaretidir. Bu iki aşamalı mantık yanlış alarmları azaltır.

**Neden çığlık tek başına alarm vermiyor?** Çocuklar oyunda, insanlar lunaparkta, filmlerde de çığlık atar. Çığlık, sadece agresyon sesinin ardından gelirse anlamlıdır.

Eşikler (0,5 ve 4 saniye) `model/ayarlar.json` dosyasında. Test sonuçlarına bakıp birlikte ayarlayabiliriz.

---

## 9. Demo senaryoları (sunum için)

### Demo 1: Colab'da zaman çizelgesi (hazırlık gerektirmez)
Defterin 16. bölümü test setinden her gruptan bir sesi alır ve grafiğini çizer:
- **Köpek hırlaması:** Kırmızı (saldırgan) eğri yükselir, arka plan turuncuya boyanır: AGRESYON.
- **Normal havlama:** Yeşil (normal_hayvan) eğri yükselir, alarm yok.
- **Yağmur ya da konuşma:** Gri (arka_plan) eğri yükselir, alarm yok.

Grafikleri ekran görüntüsü alıp sunuma koyabilirsiniz.

### Demo 2: Canlı YouTube demosu
1. `python canli_demo.py --canli --cihaz NUMARA` çalıştırın.
2. Sırayla şu videoları açın ve terminali gösterin:
   - **Köpek kavgası** ("dogs fighting caught on camera"): AGRESYON, ardından insanlar bağırıyorsa SALDIRI OLASI beklenir.
   - **Kedi kavgası** ("cat fight sounds"): Tıslama ve hırlamada AGRESYON.
   - **Aslan avı** ("lion hunt"): Kükremede AGRESYON.
   - **Köpekle oyun** ("dog playing with owner"): normal ya da kısa süreli AGRESYON. Bu, sistemin sınırını dürüstçe göstermek için iyi bir örnek.
   - **Sakin video** (doğa sesi, konuşma): normal.

### Demo 3: Eğitim öncesi ve sonrası
Defterin 13. bölümündeki tabloyu gösterin: "Hazır YAMNet'i elle kurallarla kullanınca doğruluk şuydu, kendi verimizle eğitince şuna çıktı." Bu, eğitimin işe yaradığının en net kanıtıdır.

---

## 10. Kavga modülüyle karşılaştırma

| | Kavga modülü | Hayvan saldırısı ses modülü |
|---|---|---|
| Girdi | Video (görüntü) | Ses |
| Hazır model | Önceden eğitilmiş görüntü modeli | YAMNet (önceden eğitilmiş ses modeli) |
| Yöntem | Transfer öğrenme | Transfer öğrenme (aynı mantık) |
| Ara temsil | Görüntü kareleri | Spektrogram (sesin resmi) |
| Sınıflar | Kavga / kavga değil | Saldırgan / kurban / normal hayvan / arka plan |
| Çıktı | Kavga var/yok | normal / AGRESYON / SALDIRI OLASI |
| Ortak yan | İkisi de Colab'da eğitiliyor, test/doğrulama ayrımı, doğruluk ve F1 ile ölçülüyor | |

İki modül ayrı. Gelecekte istenirse birlikte çalıştırılabilir (görüntü + ses), ama şimdilik bağımsızlar.

---

## 11. Sunumda sorulabilecek sorular ve cevapları

**"Neden hazır veri seti kullanmadınız da kendiniz etiketlemediniz?"**
VGGSound akademide yaygın kullanılan, etiketli ve açık lisanslı bir veri seti. Kendi etiketlememiz hem çok zaman alırdı hem de daha az güvenilir olurdu.

**"Neden sıfırdan model eğitmediniz?"**
Sıfırdan iyi bir ses modeli için milyonlarca örnek gerekir. Transfer öğrenmeyle YAMNet'in 2 milyon klipten öğrendiğini kullanıp az veriyle yüksek başarı elde ettik.

**"Hırlama her zaman saldırı mı?"**
Hayır. Bu yüzden iki seviyeli alarm kullandık: tek başına hırlama "agresyon", hırlamadan sonra çığlık "saldırı olası".

**"Sessiz bir saldırı olursa?"**
Sistem yakalayamaz. Bu bir sınırlılık; görüntüyle birleştirilerek çözülebilir.

**"Gerçek hayatta kullanılabilir mi?"**
Prototip düzeyinde. Gerçek kullanım için daha fazla veri, farklı mikrofonlarla test ve eşik ayarı gerekir.

---

## 12. Sınırlılıklar (raporda dürüstçe yazılmalı)

- **Zayıf etiketler:** VGGSound'da etiket tüm 10 saniyeye ait; hırlama klibin sadece bir bölümünde olabilir.
- **YouTube sesi:** Müzik, anlatım ve kurgu efektleri olan videolarda başarı düşebilir.
- **Oyun hırlaması:** Köpeklerin oyunda hırlaması saldırı sanılabilir.
- **Sessiz saldırılar:** Sesi olmayan saldırılar yakalanamaz.
- **Silinen videolar:** VGGSound'daki bazı YouTube videoları silinmiş, veri miktarı biraz azalabilir.

---

## 13. Sorun giderme

| Sorun | Çözüm |
|---|---|
| `pip` ya da `python` bulunamadı | python.org'dan Python 3.11 kurun, "Add Python to PATH" kutusunu işaretleyin |
| `No module named 'pkg_resources'` | `pip install "setuptools<81"` |
| Canlı demoda her satırda "SESSIZ" | `--cihazlar` ile listeyi açın, YouTube'un çaldığı çıkışın numarasını `--cihaz` ile verin |
| Colab'da `metadata.csv` bulunamadı | Drive'daki klasör yolunun `MyDrive/hayvan-ses/veri` olduğundan emin olun |
| Veri indirme çok yavaş | `--paralel 8` ile aynı anda daha fazla indirme deneyin |
| Bir sınıfta çok az klip indi | Normal; silinmiş videolar atlanıyor. `--adet` değerini artırın |
