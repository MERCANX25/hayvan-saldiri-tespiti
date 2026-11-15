# Ses ile Hayvan Saldırısı Tespiti: Araştırma Raporu

*15 Kasım 2026. Kapsam: hayvanın hayvana ya da insana saldırısının **sesten** tespiti, yalnızca hazır veri setleriyle. Test: YouTube videoları sesli oynatılarak.*

## 1. Kısa özet

- **Hazır "hayvan saldırısı sesi" veri seti yok.** Ama saldırının sesini oluşturan parçalar büyük, açık veri setlerinde zaten etiketli: **hırlama, kükreme, tıslama, agresif havlama, acı/korku sesleri ve insan çığlığı.** Bunları birleştirerek projeyi tamamen hazır verilerle yapabiliriz.
- **En uygun ana kaynak VGGSound** (YouTube kaynaklı, CC BY 4.0): içinde *dog growling* (826 klip), *cat hissing* (320), *cat growling* (387), *lions roaring* (964), *lions growling* (599), *people screaming* (731), *dog whimpering* (461) gibi sınıflar var. **AudioSet** ve **FSD50K** destekleyici kaynak.
- **Önerilen sistem:** Önceden eğitilmiş bir ses modeli (PANNs, YAMNet veya AST) ile 1 saniyelik pencereler halinde dinleyip iki tür sesi aramak: **saldırgan hayvan sesi** (hırlama, kükreme, tıslama) ve **kurban sesi** (çığlık, acı içinde inleme). İkisi kısa süre içinde birlikte duyulursa "saldırı olası" alarmı vermek.
- **Benzeri var mı?** Doğrudan "sesten hayvan saldırısı" tespit eden bir sistem bulamadım. Benzerleri tek türe odaklanıyor: köpek havlamasından bağlam tanıma, domuz çığlıklarından kuyruk ısırma tespiti, sokak köpeğinin hırlamasını sınıflandırma.

## 2. Kullanılabilecek veri setleri

### 2.1 Büyük genel ses veri setleri (ana kaynak)

| Veri seti | Boyut | Saldırıyla ilgili sınıflar (klip/etiket sayısı) | Lisans ve erişim |
|---|---|---|---|
| [**VGGSound**](https://github.com/hche11/VGGSound) | 210.000'den fazla YouTube klibi (10 sn), 310 sınıf | **Saldırgan:** dog growling 826, cat growling 387, cat hissing 320, lions roaring 964, lions growling 599, otter growling 363, alligators/crocodiles hissing 361, snake hissing 285<br>**Kurban/acı:** dog whimpering 461, people screaming 731<br>**Normal hayvan:** dog barking 767, dog howling 929, pig oinking 444, cattle mooing 735, horse neighing 587, elephant trumpeting 310, chimpanzee pant-hooting 473 | CC BY 4.0, videoların telifi sahiplerinde. Liste YouTube ID ve saniye olarak veriliyor, ses indirilmesi gerekiyor |
| [**AudioSet**](https://research.google.com/audioset/) | Yaklaşık 2 milyon klip (10 sn), 527 sınıf | **Köpek:** Growling 787, Bark 2.632, Yip 2.363, Howl 838, Bow-wow 3.884, Whimper 1.826<br>**Kedi:** Hiss 2.583, Caterwaul 500, Meow 1.961, Purr 543<br>**Büyük kediler:** Roar 943<br>**Diğer:** Oink 1.784, Screaming 1.246 | CC BY 4.0. YouTube ID ile; etiketler "zayıf" (klip düzeyinde, sesin hangi saniyede olduğu yok) |
| [**FSD50K**](https://zenodo.org/records/4060432) | 51.197 Freesound klibi, 108 saat, 200 sınıf (AudioSet sınıf adlarıyla) | AudioSet sınıf adlarını kullanıyor. Growling, Bark, Hiss, Roar ve Screaming'in listede olup olmadığını henüz doğrulamadım (vocabulary.csv dosyasından kontrol edilmeli) | Klip başına CC lisansı. **Doğrudan indirilebilir** (Zenodo, 24,7 GB) |
| [**ESC-50**](https://github.com/karolpiczak/ESC-50) | 2.000 klip (5 sn), 50 sınıf | Dog, Cat, Pig, Cow, Rooster, Hen, Sheep, Crow, Crying baby. **Saldırgan ses sınıfı yok** | CC BY-NC. Doğrudan indirilebilir (yaklaşık 600 MB). Normal hayvan sesleri ve test için iyi |
| [**UrbanSound8K**](https://urbansounddataset.weebly.com/urbansound8k.html) | 8.732 klip, 10 sınıf | Sadece dog_bark | Doğrudan indirilebilir. **Arka plan gürültüsü** (trafik, siren, çocuk oyunu) için çok değerli |

### 2.2 Köpek sesine özel veri setleri

| Veri seti | İçerik | Bizim için değeri | Erişim |
|---|---|---|---|
| [**Dog Bark Decoding** (Abzaliev ve ark., 2024)](https://arxiv.org/pdf/2404.18739v1.pdf) | 74 köpek, 16.789 ses parçası. Sınıflar arasında **yabancıya çok agresif havlama** (2.843), **sahibine saldırı anında havlama** (829), yabancıya normal havlama (2.772), oyun havlaması | Literatürdeki tek **agresyon etiketli** köpek ses seti. "Agresif havlama" ile "normal havlama" ayrımı için ideal | **Yazarlardan istek ile** (açık indirme yok) |
| [**Köpek-insan oyun sesleri** (Cuaya ve ark., 2026)](https://www.biorxiv.org/content/10.64898/2026.04.20.719471v1.full.pdf) | 17 köpek, 30 oyun seansı. Etiketler: Growl, Bark/yelp, Whine, Howl, Pant, Grunt, insan sesi | **Oyun sırasındaki hırlama.** Modelin "her hırlama saldırıdır" diye öğrenmesini engelleyen en iyi zor negatif | Açık, [Zenodo](https://doi.org/10.5281/zenodo.18972388) |
| [**EmotionalCanines** (2025)](https://huggingface.co/datasets/ArlingtonCL2/BarkopediaDogEmotionClassification_Data) | Köpek sesleri, uyarılma (düşük/orta/yüksek) ve duygu değeri (negatif/nötr/pozitif) etiketli | "Yüksek uyarılma + negatif" sesler agresyona yakın | CC BY-NC-SA 4.0, Hugging Face |
| [**Barkopedia Activity** (2025)](https://huggingface.co/datasets/ArlingtonCL2/Barkopedia_Dog_Act_Env) | 12.480 klip. Aktiviteler: dinlenme, sese tepki, dikkat çekme, insanla/hayvanla/oyuncakla oyun, yemek isteme | Saldırı olmayan köpek sesleri (negatif) | MIT, Hugging Face |
| [**DogSpeak** (2025)](https://huggingface.co/datasets/mkkimgnu/DogSpeak_Dataset) | 156 köpekten 77.202 havlama dizisi, 5 cins | Etiket yok (sadece cins/cinsiyet). Ön eğitim için | CC BY-NC-SA 4.0 |

### 2.3 Diğer türler

- **Domuz çığlıkları** ([Heseker ve ark., 2024](https://www.nature.com/articles/s41598-024-55336-7.pdf); [otomatik tespit, 2024](https://doaj.org/article/f50ddab3d020434dade317ce86a5fa94)): Kuyruk ısırma ve kavga sırasındaki çığlıklar. Veri açık değil, ama yöntem referansı olarak iyi.

## 3. Benzer çalışmalar ve sonuçları

| Çalışma | Ne yapıyor | Yöntem | Sonuç |
|---|---|---|---|
| [Sokak köpeği + IoT (ISI 2026)](https://iieta.org/journals/isi/paper/10.18280/isi.310630) | Köpek sesini havlama, hırlama ve homurtu olarak sınıflandırıp uyarı veriyor (görüntüyle birlikte) | MFCC (Librosa) + CNN | Ses modeli doğrulamada yaklaşık %88. Veri kaynağı ve boyutu belirsiz |
| [Dog Bark Decoding (2024)](https://arxiv.org/pdf/2404.18739v1.pdf) | Havlamadan bağlamı tanıma (agresif/normal havlama, homurtu, cıyaklama) | İnsan konuşması için eğitilmiş Wav2Vec2 ile ince ayar | 4 sınıflı bağlamda %62,2 (çoğunluk sınıfı tahmini %56,4). Zor bir problem |
| [Domuz çığlığı tespiti (2024)](https://doaj.org/article/f50ddab3d020434dade317ce86a5fa94) | Çığlıkları otomatik yakalayıp kuyruk ısıranı bulma | Ses yüksekliği eşiği, süre ve frekans aralığı | Tespit edilen çığlıkların %89,9'u gerçekten çığlık, hassasiyet 0,81-1,00 |
| [Köpek oyun sesleri (2026)](https://www.biorxiv.org/content/10.64898/2026.04.20.719471v1.full.pdf) | Oyun sırasında hırlama, havlama vb. ayırt etme | pyAudioAnalysis öznitelikleri + SVM | Makro F1 0,29. Klasik yöntemler bu ayrımda zayıf |
| Hazır ses modelleri | AudioSet'in 521-527 sınıfını tanıyor (Growling, Roar, Hiss, Screaming dahil) | [YAMNet](https://www.tensorflow.org/hub/tutorials/yamnet) (MobileNet), [PANNs CNN14](https://github.com/qiuqiangkong/audioset_tagging_cnn), [AST](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) | PANNs CNN14 AudioSet'te mAP 0,431. Hiç eğitim yapmadan bile hırlama ve çığlığı tanıyabiliyorlar |

**Çıkarımlar:**
1. Sesten **duygu veya bağlam** tanımak zor (yaklaşık %60), ama **ses türünü** (hırlama mı, çığlık mı) tanımak çok daha kolay. Bu yüzden sistemi ses türleri üzerine kurmak daha güvenilir.
2. Büyük veriyle önceden eğitilmiş modeller (PANNs, AST) küçük veride sıfırdan eğitilen CNN'lerden belirgin biçimde iyi.
3. Oyun sırasındaki hırlama, gerçek saldırıya çok benziyor. En büyük yanlış alarm kaynağı bu olur.

## 4. Önerilen sistem

### 4.1 Sınıflar

| Grup | İçerdiği sesler | Kaynak |
|---|---|---|
| **A. Saldırgan hayvan sesi** | Köpek/kedi hırlaması, kedi tıslaması, aslan kükremesi ve hırlaması, agresif havlama | VGGSound, AudioSet, FSD50K, (istenirse) Dog Bark Decoding |
| **B. Kurban sesi** | İnsan çığlığı, köpek inlemesi/cıyaklaması (whimper, yelp) | VGGSound, AudioSet |
| **C. Normal hayvan sesi** | Normal havlama, uluma, miyavlama, mırlama, inek, domuz, at, kuş, **oyun hırlaması** | VGGSound, ESC-50, Barkopedia, köpek oyun seti |
| **D. Arka plan** | Konuşma, trafik, müzik, siren, çocuk oyunu | UrbanSound8K, ESC-50 |

### 4.2 Karar mantığı

```
Ses akışı ──► 1 sn'lik pencereler (0,5 sn kaydırmalı, 16 kHz, log-mel)
           ──► Model: her pencere için A, B, C, D olasılığı
           ──► Zaman içinde yumuşatma (son 3-5 pencerenin ortalaması)
           ──► Karar:
               • A yüksek, birkaç pencere boyunca     → "Agresyon sesi" (sarı uyarı)
               • A yüksek ve 3-5 sn içinde B yüksek   → "Saldırı olası" (kırmızı alarm)
               • Diğer durumlar                       → Normal
```

İki aşamalı alarm, "hayvan hırladı ama saldırmadı" ile "hayvan saldırdı ve biri bağırdı" durumlarını ayırır. Böylece hem hayvanın hayvana hem de hayvanın insana saldırısı kapsanır.

### 4.3 Model seçenekleri (karşılaştırmalı deney için)

1. **Taban çizgisi, eğitimsiz:** YAMNet veya PANNs'i olduğu gibi kullanıp Growling, Roar, Hiss, Screaming sınıflarının skorlarını kural ile birleştirmek. Hiç eğitim gerektirmez, ilk gün çalışır.
2. **Ana model:** PANNs CNN14 veya AST'yi A/B/C/D sınıflarıyla ince ayarlamak (ya da bu modellerden çıkan özellik vektörlerinin üzerine küçük bir sınıflandırıcı eğitmek).
3. **Karşılaştırma:** MFCC veya log-mel üzerinde sıfırdan eğitilmiş küçük bir CNN. Literatürdeki sokak köpeği çalışmasıyla aynı yöntem, ders raporu için iyi bir karşılaştırma.

**Veri artırma:** Saldırgan sesleri UrbanSound8K'daki sokak gürültüsüyle karıştırmak, ses yüksekliğini değiştirmek ve SpecAugment uygulamak. Bu, YouTube'daki gürültülü videolarda başarıyı ciddi şekilde artırır.

### 4.4 Farkımız ve artılarımız

- **Sesten hayvan saldırısı tespiti** yapan, birden fazla türü kapsayan bir sistem literatürde yok. Mevcut çalışmalar tek türe (köpek veya domuz) ve genellikle tek ses türüne odaklanıyor.
- **İki aşamalı karar (saldırgan ses + kurban sesi)** ile "hırlama = saldırı" hatasını azaltıyoruz.
- **Oyun hırlamasını** zor negatif olarak açıkça kullanıyoruz.
- **Üç yöntemin karşılaştırması** (eğitimsiz hazır model, ince ayarlı model, sıfırdan CNN) rapor için güçlü bir deney bölümü olur.
- **Gerçek zamanlı demo:** YouTube videosu açılınca sistem canlı dinleyip alarm veriyor.

## 5. YouTube ile test planı

### 5.1 Sesi sisteme verme

İki yol var:

1. **Canlı dinleme (demo için önerilen):** Video tarayıcıda oynatılır, bilgisayarın ses çıkışı programa yönlendirilir.
   - Windows: "Stereo Mix" veya VB-Audio Virtual Cable
   - macOS: BlackHole
   - Linux: PulseAudio "monitor" kaynağı
   - Python'da `sounddevice` ile 16 kHz mono okuma, her 0,5 saniyede bir model çalıştırma, ekranda canlı skor grafiği ve alarm gösterme.
2. **Dosyadan test (ölçüm için):** Videonun sesi önce dosya olarak kaydedilir (ör. canlı dinlemeyle WAV'a kaydetmek), sonra tüm dosya üzerinde aynı pencereleme uygulanır. Sonuçlar tekrarlanabilir olur.

### 5.2 Test seti

- **Saldırı videoları (yaklaşık 20-30):** Köpek saldırısı, köpek kavgası, kedi kavgası, aslan/leopar avı gibi. Her birinde saldırının başladığı ve bittiği saniye elle not edilir.
- **Saldırı olmayan videolar (yaklaşık 20-30):** Köpekle oyun, kapıya havlayan köpek, sakin hayvanat bahçesi, kedi miyavlaması, çocuk bağırışları olan park, korku filmi çığlığı.
- **Önemli:** VGGSound ve AudioSet de YouTube'dan geldiği için **test videoları eğitim setindeki YouTube ID'leriyle çakışmamalı.** Test videolarının ID'leri listelerde aranarak kontrol edilmeli.
- **Arka plan müziği ve anlatım olan derlemeleri** ayrı bir alt grup olarak test etmek gerekir. Bunlarda başarı düşebilir, bunu raporda belirtmek dürüstlük açısından iyi.

### 5.3 Ölçütler

- **Olay düzeyinde:** Kaç saldırı yakalandı (recall), kaç yanlış alarm verildi (precision), F1.
- **Tepki süresi:** Saldırı başladıktan kaç saniye sonra alarm geldi.
- **Saat başına yanlış alarm:** Uzun, saldırısız videolarda (ör. 1 saatlik köpek parkı videosu) kaç alarm verdi.

## 6. Eleştiri ve riskler

| Risk | Açıklama | Çözüm |
|---|---|---|
| **Sessiz saldırılar** | Bazı köpek saldırıları neredeyse sessiz olur, sadece insan bağırır | Raporda sınırlılık olarak belirtmek. B (kurban) sesi tek başına da düşük seviyeli uyarı verebilir |
| **Agresyon her zaman saldırı değil** | Hırlama bir uyarı olabilir, saldırı olmayabilir | Çıktıyı "agresyon sesi" ve "saldırı olası" olarak iki seviyede vermek |
| **Zayıf etiketler** | AudioSet ve VGGSound klip düzeyinde etiketli; 10 saniyelik klibin sadece 1 saniyesinde hırlama olabilir | Eğitimde sesin en yüksek enerjili bölümünü almak, ya da çok örnekli öğrenme (her pencereye klip etiketi vermek yerine en yüksek skorlu pencereyi kullanmak) |
| **Etiket gürültüsü** | Bazı VGGSound kliplerinde arka planda müzik ya da konuşma var | Rastgele örneklerden bir kısmını dinleyip temizlemek |
| **Kırık YouTube linkleri** | VGGSound ve AudioSet videolarının bir kısmı silinmiş olabilir | Önce FSD50K, ESC-50 ve UrbanSound8K gibi doğrudan indirilebilen setlerle başlamak, VGGSound'dan erişilebilen kadarını eklemek |
| **Telif** | Videoların telifi sahiplerinde | Sesleri yalnızca eğitim için kullanmak, yeniden yayımlamamak, ID listesi paylaşmak |
| **Çığlığın başka kaynaktan gelmesi** | Çocuk oyunu, film, sevinç çığlığı | Tek başına çığlığa alarm vermemek; A + B birlikteliğini aramak |

## 7. Yol haritası

1. **İlk deneme:** YAMNet veya PANNs'i olduğu gibi birkaç YouTube videosunda çalıştırıp Growling, Roar, Hiss ve Screaming skorlarına bakmak. Bu, fikrin çalıştığını hemen gösterir.
2. **Veri hazırlığı:** FSD50K, ESC-50 ve UrbanSound8K'yı indirmek; VGGSound'dan ilgili sınıfların seslerini toplamak; A/B/C/D gruplarına ayırmak.
3. **Eğitim:** İnce ayarlı model ve sıfırdan CNN.
4. **Gerçek zamanlı demo:** Canlı dinleme, skor grafiği ve alarm ekranı.
5. **YouTube testi ve rapor.**
