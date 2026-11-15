# Hayvan Saldırısı Tespiti: Araştırma Raporu

*Hazırlanma tarihi: 19 Ekim 2026. Yapay Zeka dersi kavga tespiti projesinin devamı için.*

## 1. Kısa özet

- **Hazır bir veri seti yok.** Kavga tespitindeki RWF-2000 ya da Hockey Fight gibi, "hayvan insana saldırıyor / saldırmıyor" diye etiketlenmiş halka açık bir video veri seti bulamadım. Var olanlar ya hayvan davranışı veri setleri (içinde "attacking" ya da "fight" sınıfı geçiyor), ya çiftlik hayvanları arası saldırganlık üzerine (çoğu kapalı), ya da çok küçük görüntü setleri.
- **Literatürdeki çalışmalar çoğunlukla insanı hedef almıyor.** Yayınlar ya hayvanın hayvana saldırısına (domuzlar arası saldırganlık, yırtıcının sürüye saldırması) ya da vahşi hayvanın yerleşim alanına girmesine (fil, leopar) bakıyor. **Kamera görüntüsünde hayvanın insana saldırması (özellikle sokak köpeği) neredeyse hiç çalışılmamış.** Bu bizim için en büyük fırsat.
- **Önerim:** Türkiye'deki sokak köpeği saldırısı CCTV görüntülerinden küçük ama özgün bir veri seti toplamak ve "hayvan + insan algılama, ardından zamansal sınıflandırma" yapan iki aşamalı, kavga modülünden bağımsız bir modül kurmak. Ayrıntılar 5., 6. ve 9. bölümde.
- **Güncelleme (19 Ekim):** Kavga ve hayvan saldırısı ayrı modüller olarak kalacak. İnsana saldırı verisi ve hazır sistemler için ayrıntılı tarama 9. bölümde.

## 2. Veri setleri

| Veri seti | İçerik | Saldırıyla ilgisi | Erişim | Bizim için değeri |
|---|---|---|---|---|
| [Animal Kingdom](https://sutdcv.github.io/Animal-Kingdom) (CVPR 2022) | 850'den fazla türden 30.100 klip (ortalama 6 sn), 140 eylem sınıfı, poz verisi | "Aggressive" grubunda **Attacking**, "Defensive" grubunda Retreating, "Feeding" grubunda Biting var | Google Form ile istek | Saldırı klipleri ve "normal" negatif örnekler için kaynak |
| [MammalNet](https://mammal-net.github.io) (CVPR 2023) | 18.000'den fazla video (539 saat), 173 memeli kategorisi, 12 davranış | Davranışlar arasında **fight** ve **hunt** var | YouTube linkleri ve CC BY etiketler | Hayvanlar arası kavga/av klipleri, ön eğitim için |
| [PanAf20K](https://arxiv.org/pdf/2401.13554) | Kamera tuzağından yaklaşık 20.000 şempanze/goril videosu, 18 davranış | **Aggression** ve **display** sınıfları var ama nadir | Yazarlardan istek | Yalnızca primatlar, sınırlı |
| [Roboflow "Aggressive dog"](https://universe.roboflow.com/fgdgdfg/aggressive-dog) | Yaklaşık 95 görüntü (person, agressive_dog, notagressive_dog) | Doğrudan köpek saldırganlığı | CC BY 4.0, açık | Çok küçük, ancak YOLO ince ayarı için başlangıç olabilir |
| [Çiftlik tehditleri (MDPI 2024)](https://www.mdpi.com/2076-3417/14/14/6098) | 2.462 görüntü, 7 sınıf (köpek, tilki, çakal, yırtıcı kuş vb.) | Sürüye tehdit oluşturan hayvanların tespiti, saldırı eylemi yok | Kaggle/Roboflow/Ultralytics kaynaklı, CC0 | Hayvan algılama aşaması için |
| [Domuz saldırganlığı (2024)](https://arxiv.org/html/2403.08528v1) | 421 klip (143 saldırgan, 278 değil) | Domuzlar arası saldırganlık | Açık değil | Yöntem karşılaştırması için referans |
| [Koyun aktivite videoları](https://www.sciencedirect.com/science/article/pii/S2352340924000015) | Koyun davranış videoları | Dolaylı | Açık (Data in Brief) | Düşük |

**Kendi veri setimizi toplayabileceğimiz kaynaklar:**
- Türk haber sitelerinde yayınlanan sokak köpeği saldırısı CCTV görüntüleri ([TRT Haber](https://www.trthaber.com/videolar/istanbulda-sokak-kopeklerinin-cocuga-saldirisi-kamerada-51708.html), [CNN Türk](https://www.cnnturk.com/turkiye/galeri/sokak-kopeginin-saldirisi-kamerada-2418838) vb.). Bunlar gerçek güvenlik kamerası açısından çekilmiş, tam da RWF-2000 tarzı görüntüler.
- YouTube'daki "animal attacks caught on camera" derlemeleri (her biri onlarca-yüzlerce olay içeriyor; [örnek](https://www.youtube.com/watch?v=SBpKIcV1tFo)).
- [Pixabay](https://pixabay.com/videos/search/animal%20attack/) gibi telifsiz stok videolar (çoğu hayvanlar arası, bazıları sahnelenmiş).
- Negatif örnekler: insanların köpekle yürüdüğü, oynadığı, köpeklerin sokakta dolaştığı videolar. Kavga projesindeki "non-fight" videolarının mantığıyla aynı.

> Not: YouTube ve haber videolarını yalnızca akademik ders projesi amaçlı kullanmak, veri setini yayımlarken video yerine link ve zaman damgası paylaşmak ve yüzleri bulanıklaştırmak gerekir. İçerik rahatsız edici olabilir, etiketleme yapanlar bunu bilmeli.

## 3. Benzer projeler ve nasıl yaptıkları

| Çalışma | Problem | Yöntem | Sonuç |
|---|---|---|---|
| [Sokak köpeği davranışı + çocuk güvenliği (ISI, 2026)](https://iieta.org/journals/isi/paper/10.18280/isi.310630) | Sokak köpeğinin saldırganlığını tespit edip uyarı göndermek | YOLOv5 ile köpek tespiti ve duygu sınıflandırması (mutlu, üzgün, kızgın, rahat) + MFCC ve CNN ile ses sınıflandırması (havlama, hırlama) + karar düzeyinde füzyon + ESP32 ve Telegram üzerinden uyarı | Yaklaşık %92 genel doğruluk, füzyonla yanlış alarmlarda yaklaşık %30 azalma. Ancak tek kare görüntüyle çalışıyor, saldırı eylemini değil "duyguyu" sınıflandırıyor |
| [Fil davranışı tanıma (ITM Conf., 2026)](https://www.itm-conferences.org/articles/itmconf/pdf/2026/05/itmconf_issf2026_01012.pdf) | İnsan-fil çatışmasını erken fark etmek | YOLOv8 ile tespit ve duruş, DeepSORT ile takip, hız/yön/ivme öznitelikleri, 30-60 karelik dizilerde LSTM | Ortalama F1 %90,7. Transformer %91,3 ile biraz daha iyi ama kenar cihazlar için ağır bulunmuş |
| [Domuz saldırganlığı sınıflandırma (2024)](https://arxiv.org/html/2403.08528v1) | Domuzlar arası saldırganlık | TimeSformer, ViViT, STAM, ResNet3D, ResNet(2+1)D ve CNN-LSTM karşılaştırması, ön eğitimsiz | En iyisi TimeSformer, ortalama doğruluk %76,4. Küçük veri ve ön eğitim olmaması başarıyı sınırlamış |
| [Domuz saldırı davranışı (2023)](https://doi.org/10.1080/21642583.2023.2249934) | Domuz saldırısı | Derin öğrenme tabanlı tespit (makaleye tam erişemedim) | – |
| [Çiftlik tehditleri (MDPI 2024)](https://www.mdpi.com/2076-3417/14/14/6098) | Sürüye yaklaşan yırtıcıyı tespit etmek | YOLOv8, YOLO-NAS ve "Fast-RNN" karşılaştırması | YOLOv8: hassasiyet %93, mAP50 %93,1. Yalnızca nesne tespiti, davranış yok |
| [YOLO-Behaviour (2024)](https://www.biorxiv.org/content/10.1101/2024.08.26.609387v1) | Videodan davranış ölçmek | Davranışın görsel olarak ayırt edici kısmına kutu çizip YOLOv8 eğitmek, SORT ile kareleri olaylara bağlamak | F1 0,62 ile 0,94 arası. Olay başına yaklaşık 1.000-1.500 etiketli kare yetmiş. Harekete dayalı davranışlarda zorlanıyor |
| [MammalNet temel modelleri](https://arxiv.org/pdf/2306.00576) | 12 davranışın tanınması | SlowFast, C3D, I3D, MViT V2 | En iyi MViT V2 (Kinetics ön eğitimli), örnek başına %46,6. Ön eğitim her modelde 5-10 puan kazandırmış |
| [Animal Kingdom temel modelleri](https://arxiv.org/pdf/2204.08129) | 140 sınıflı çok etiketli eylem tanıma | I3D, SlowFast, X3D | X3D ile equalization loss en iyisi, mAP %30,55. Nadir sınıflar çok zor |
| [Anahtar noktalarla hayvan eylemi (Computer Optics, 2023)](https://doaj.org/article/ce2b58dc957c486d85541dfd46a003ab) | Az veriyle hayvan eylemi | Poz (anahtar nokta) tahmini, ardından eylem sınıflandırma, sentetik veri üretimi | Anahtar nokta doğruluğu %92, eylem doğruluğu %72,9 |
| [Köpek eylem tanıma, sentetik veri (2025)](https://scitepress.org/PublishedPapers/2025/131950) | Az veriyle köpek eylemi | Anahtar noktalı ve anahtar noktasız yaklaşım, üretken modellerle sentetik görüntü | Sentetik ve gerçek verinin birlikte kullanılması gerçek görüntülerde işe yaramış (sayı verilmemiş) |

**Karşılaştırma için kavga tespiti literatürü** ([derleme](https://arxiv.org/pdf/2209.11680)): Hockey Fight veri setinde derin modeller yaklaşık %98'e ulaşıyor. Gerçek CCTV görüntülerinden oluşan RWF-2000'de ise I3D yaklaşık %85,8, Cheng ve arkadaşlarının modeli yalnızca 0,27M parametreyle %87,3 başarı veriyor. Yaygın yöntemler 3D CNN (C3D, I3D), CNN + LSTM/ConvLSTM ve optik akışlı iki akışlı ağlar.

## 4. Literatürden çıkan dersler

1. **Tek kare yetmiyor.** Hareketle tanımlanan davranışlarda (saldırı, kovalama) tek kareye bakan YOLO tabanlı yöntemler zorlanıyor. Takip ve zamansal model (LSTM, 3D CNN, Transformer) ekleyen fil çalışması belirgin biçimde daha iyi sonuç almış (yalnızca YOLO ile F1 %80,3, takip + LSTM ile %90,7).
2. **Ön eğitim çok önemli.** Küçük veride sıfırdan eğitilen modeller (domuz çalışması, %76) zayıf kalıyor. Kinetics ile ön eğitilmiş modeller MammalNet'te her durumda daha iyi.
3. **Asıl zor olan "oyun ile saldırıyı" ayırmak.** Köpekle oynayan insan ya da birbiriyle boğuşarak oynayan köpekler, saldırıya çok benziyor. İyi bir veri setinde bu tür "zor negatif" örnekler bilinçli olarak bulunmalı.
4. **Çok kipli (görüntü + ses) yaklaşım yanlış alarmı azaltıyor.** Sokak köpeği çalışmasında füzyonla yanlış pozitiflerde yaklaşık %30 düşüş rapor edilmiş.
5. **Veri az olduğunda poz/anahtar nokta ve sentetik veri işe yarıyor.**

## 5. Biz ne yapabiliriz?

### Önerilen sistem (kavga tespiti projesinin üzerine)

```
Video ──► [1] Algılama: YOLOv8/v11 (insan + köpek/kedi/diğer hayvan)
        ──► [2] Takip: ByteTrack/DeepSORT (her nesneye kimlik)
        ──► [3a] Klip sınıflandırma: Kinetics ön eğitimli model
                  (ör. MobileNet+LSTM, X3D veya VideoMAE ince ayarı)
        ──► [3b] Etkileşim öznitelikleri: hayvan-insan mesafesi,
                  yaklaşma hızı, kutuların örtüşmesi, insanın düşmesi
        ──► [4] Füzyon ──► "Saldırı var / yok" + kim kime + güven skoru
        ──► [5] (Opsiyonel) Ses: hırlama, çığlık (MFCC + CNN)
        ──► Uyarı (Telegram/e-posta) + olay kesiti kaydı
```

- **Aşama 3a:** Kinetics gibi büyük bir video veri setinde önceden eğitilmiş bir modeli hayvan saldırısı verisiyle ince ayarlamak. Bu modül kavga modülünden tamamen ayrı eğitilir.
- **Aşama 3b:** Literatürde en az kullanılan ama bizim problemimiz için en anlamlı kısım. Saldırıda hayvan insana hızla yaklaşır, kutular örtüşür, insan çoğu zaman düşer ya da kaçar. Bu öznitelikler basit bir LSTM veya hatta kural tabanlı bir sistemle güçlü bir sinyal verir.

### Sınıf tasarımı

- Temel (ikili): `saldırı` / `saldırı değil` (RWF-2000 ile aynı yapı, kavga projesiyle doğrudan karşılaştırılabilir)
- Genişletilmiş (zaman kalırsa): `hayvan→insan saldırısı`, `hayvan→hayvan kavgası`, `oyun`, `normal`

### Hedef veri miktarı (ders projesi için gerçekçi)

- 300-500 saldırı klibi (3-5 sn) ve aynı sayıda normal klip. Bunun en az %20'si zor negatif (oyun, koşan köpek, köpek gezdiren insan) olmalı.
- Ek olarak Animal Kingdom'daki "Attacking" ve MammalNet'teki "fight/hunt" klipleriyle ön eğitim.

## 6. Farkımız ve artılarımız

1. **İnsan hedefli saldırı:** Literatür hayvan-hayvan ve çiftlik/vahşi yaşam üzerine yoğunlaşmış. Kamerada hayvanın insana saldırısını tespit eden, video tabanlı ve açık bir çalışma neredeyse yok.
2. **Türkiye bağlamı ve özgün veri seti:** Sokak köpeği saldırıları Türkiye'de güncel ve sık haberleştirilen bir konu. Gerçek CCTV görüntülerinden oluşan küçük bir "Türkiye Sokak Hayvanı Saldırısı" veri seti, tek başına önemli bir katkı.
3. **Kavga modülüyle karşılaştırma (opsiyonel):** Modüller ayrı kalsa da "kavga modülü hayvan saldırısı videolarında ne veriyor?" diye tek bir test yapmak, ayrı bir hayvan modülüne neden ihtiyaç olduğunu sunumda gösterir.
4. **Zamansal ve etkileşim tabanlı yaklaşım:** Mevcut sokak köpeği çalışması tek kareden "duygu" tahmin ediyor. Biz eylemi (yaklaşma, ısırma, düşme) zaman içinde modelleyeceğiz.
5. **Zor negatiflerle değerlendirme:** "Oyun mu saldırı mı" ayrımını açıkça ölçen bir test seti, sistemin gerçek hayatta yanlış alarm üretip üretmeyeceğini gösterir.
6. **Gerçek zamanlı uyarı ve açıklanabilirlik:** Telegram uyarısı ve Grad-CAM ya da dikkat haritalarıyla modelin nereye baktığını göstermek, demoyu etkileyici hale getirir.
7. **Sıfır atış (zero-shot) karşılaştırması (opsiyonel):** CLIP veya X-CLIP gibi görüntü-metin modellerini "a dog attacking a person" sorgusuyla eğitimsiz bir taban çizgisi olarak kullanıp kendi modelimizle karşılaştırmak.

## 7. Önerilen yol haritası

1. **Veri toplama ve etiketleme:** Haber ve YouTube kaynaklarından klipleri kesip etiketlemek (CVAT veya Label Studio). Eğitim, doğrulama ve test bölmesini, aynı olaya ait klipler farklı bölmelere düşmeyecek şekilde yapmak.
2. **Taban çizgileri:** (a) Yalnızca YOLO ile kare bazlı tespit ve basit mesafe kuralı, (b) CLIP/X-CLIP ile sıfır atış.
3. **Ana model:** Kinetics ön eğitimli 3D CNN / VideoMAE ince ayarı.
4. **Etkileşim öznitelikleri ve füzyon.**
5. **Değerlendirme:** Doğruluk, F1, yanlış alarm oranı, zor negatif alt kümesindeki başarı, FPS.
6. **Demo:** Canlı video veya kayıt üzerinde uyarı.

## 8. Riskler

- **Veri azlığı:** Ön eğitim, veri artırma (yansıtma, kırpma, hız değişimi) ve ek kaynak setleriyle hafifletilebilir.
- **Etiket belirsizliği:** Saldırının başlangıç anı belirsiz olabilir. Basit ve yazılı bir etiketleme kılavuzu hazırlamak gerekir.
- **Etik ve hukuki konular:** Rahatsız edici içerik, kişisel veriler ve video telifi. Bölüm 2'deki nota bakın.

## 9. Hayvanın insana saldırısı: veri, hazır sistem ve veri stratejisi

### 9.1 Hayvanın insana saldırdığı bir veri seti var mı?

**Hayır, halka açık ve bu amaçla etiketlenmiş bir veri seti yok.** Kontrol ettiklerim:

- **Şiddet ve anomali veri setleri** ([UCF-Crime](https://www.crcv.ucf.edu/projects/real-world/), RWF-2000, Real Life Violence Situations, [DVD](https://arxiv.org/html/2506.05372v1), [SCVD](https://www.kaggle.com/datasets/toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd)): Yalnızca insan-insan şiddeti içeriyor. UCF-Crime'ın 13 sınıfında (kavga, saldırı, soygun vb.) hayvan saldırısı yok.
- **Hayvan davranışı veri setleri** (Animal Kingdom, MammalNet, PanAf20K): Saldırı ve kavga sınıfları var, ama neredeyse tamamı hayvanın hayvana yaptığı saldırılar.
- **Roboflow ve Kaggle:** Yalnızca küçük görüntü setleri var. "Aggressive dog" yaklaşık 95 görüntü, "injured dog/animal detector" setleri 46-462 görüntü arasında. Saldırı anını gösteren video seti yok.
- **Tek tek vakalar:** Örneğin Rusya'da ölümle sonuçlanan bir sokak köpeği saldırısının CCTV kaydı [adli tıp makalesi](https://link.springer.com/article/10.1007/s12024-025-01015-6) olarak yayımlanmış. Bu bir veri seti değil, tek bir olay.

### 9.2 Bunu tespit eden hazır bir sistem var mı?

**Doğrudan "hayvan insana saldırıyor" diye alarm veren, video tabanlı, yayımlanmış bir sistem bulamadım.** En yakınları:

| Sistem | Ne yapıyor | Eksiği |
|---|---|---|
| [Sokak köpeği + IoT (ISI 2026)](https://iieta.org/journals/isi/paper/10.18280/isi.310630) | YOLOv5 ile köpeğin "kızgın" olup olmadığını tek kareden tahmin ediyor, hırlama sesiyle birleştirip Telegram'dan uyarı gönderiyor | Saldırı eylemini değil duygu durumunu sınıflandırıyor, insan-köpek etkileşimine bakmıyor |
| [Fil davranışı (ITM 2026)](https://www.itm-conferences.org/articles/itmconf/pdf/2026/05/itmconf_issf2026_01012.pdf) | YOLOv8, DeepSORT ve LSTM ile filin "saldırgan" davranışını tanıyıp uyarı gönderiyor | Yalnızca fil, insana temas anını değil önceki riski ölçüyor |
| Vahşi hayvan izinsiz giriş sistemleri ([Bhutan](https://publication.cst.edu.bt/index.php/zm/article/view/127), [YOLOv11](https://dspace.lib.seu.ac.lk/handle/123456789/7892), [Kaziranga](https://rhinoresourcecenter.com/library/references/an-automated-approach-for-human-animal-conflict-minimisation-in-assam-and-protection-of-wildlife-around-the-kaziranga-national-park-using-yolo-and-senet-attention-framework)) | Köye/tarlaya hayvan girdiğinde uyarı veriyor | Yalnızca varlık tespiti, saldırı yok |
| Ticari güvenlik kameraları | "Evcil hayvan algılandı" bildirimi | Yalnızca varlık tespiti |

Yani bu alandaki boşluk gerçek. Projenin ana katkısı da bu olabilir.

### 9.3 Biz nasıl yapabiliriz?

Kavga modülünden bağımsız, şu iki aşamalı yaklaşımı öneriyorum (ayrıntı 5. bölümde):

1. **Algılama ve takip:** YOLO ile her karede `insan` ve `köpek/kedi/diğer hayvan` bulunur, ByteTrack ile her birine kimlik verilir. Bu aşama için hazır COCO ağırlıkları yeterli, çünkü COCO'da person, dog, cat, horse, cow, bear gibi sınıflar zaten var.
2. **Saldırı kararı:** İki seçenek var, ikisi birlikte de kullanılabilir:
   - **Klip sınıflandırıcı:** Bir hayvan ile bir insanın birlikte göründüğü 2-4 saniyelik kırpılmış klipte `saldırı / saldırı değil` kararı veren, Kinetics ön eğitimli bir video modeli (X3D, VideoMAE ya da MobileNet + LSTM).
   - **Etkileşim öznitelikleri:** Hayvanın insana yaklaşma hızı, kutuların örtüşmesi, insanın aniden yere inmesi (düşme) ve kaçış hızı. Bunlardan küçük bir LSTM ya da gradient boosting ile karar verilir. Az veriyle bile çalışır ve açıklaması kolaydır.

### 9.4 Veri yoksa hangi veriyi kullanabiliriz?

Önerdiğim veri karışımı, önem sırasına göre:

| # | Kaynak | Ne sağlar | Not |
|---|---|---|---|
| 1 | **Haber sitelerindeki CCTV kayıtları** (TRT Haber, CNN Türk, Hürriyet vb. "sokak köpeği saldırısı kamerada") | Gerçek, güvenlik kamerası açısından insana saldırı (pozitif) | En değerli kaynak. Yüzlerce haber var, her biri bir olay |
| 2 | **YouTube derlemeleri** ("dog attack caught on camera", "animal attacks caught on CCTV") | Çok sayıda pozitif olay, farklı hayvanlar | Bir derlemeden 50-200 olay kesilebilir. Aynı olayın tekrar edip etmediğine bakılmalı |
| 3 | **Polis/K9 köpek eğitimi videoları** ("bite work", "bite sleeve", "K9 training") | Köpeğin insana atlayıp ısırması, koruyucu kıyafetle, kan ve şiddet yok | Çok bol bulunuyor ve etik olarak daha rahat. Ancak tek başına yetmez: kamera açısı ve sahne gerçek CCTV'den farklı |
| 4 | **Kinetics'teki köpek/hayvan sınıfları** (ör. walking the dog, training dog, grooming dog, petting animal) | İnsanla hayvanın barışçıl etkileşimi (zor negatif) | Modelin "köpek + insan = saldırı" diye ezber yapmasını engeller |
| 5 | **Oyun videoları** ("dog playing with owner", "dogs play fighting") | En zor negatifler | Mutlaka test setinde de olmalı |
| 6 | **Sıradan sokak CCTV'si** (köpek geçiyor, insan yürüyor) | Normal negatifler | Pozitiflerle aynı kamera türünden olmalı |
| 7 | **Animal Kingdom "Attacking" ve MammalNet "fight/hunt"** | Saldırı hareketinin genel görünümü (hayvan-hayvan) | Yalnızca ön eğitim veya ek veri olarak |
| 8 | **Sentetik veri** (opsiyonel): Video üretim modelleri ya da oyun motoru (Unity) ile sahne | Nadir açılar ve gece görüntüleri | Yalnızca eğitimde kullanılmalı, test hep gerçek video olmalı |

**Pratik hedef:** 300-500 pozitif klip (1+2+3), en az aynı sayıda negatif (4+5+6). Test setinde yalnızca 1 ve 2'den gelen gerçek CCTV klipleri ve oyun videoları olmalı.

**Etiketleme önerisi:**
- Klip düzeyinde: `saldırı` / `saldırı değil`, ek olarak `hayvan türü` ve `kaynak` (haber, YouTube, K9, Kinetics) etiketleri.
- Zaman düzeyinde (opsiyonel): Saldırının başladığı ve bittiği saniye. Bu, gerçek zamanlı uyarının ne kadar erken geldiğini ölçmeyi sağlar.
- Araç: CVAT veya Label Studio. Aynı olaydan kesilen klipler hep aynı bölmede (eğitim/doğrulama/test) olmalı.

## 10. Veri kaynaklarının eleştirisi: izin, doğruluk ve çözümler

### 10.1 İzin almak şart mı?

> Bu bölüm hukuki tavsiye değildir. Kesin cevap için dersin hocasına ve gerekiyorsa üniversitenin etik kuruluna danışın.

**Kısa cevap:** Kesin olarak "izin şart değil" diyemem. Ama kapalı bir ders projesinde, aşağıdaki kurallara uyulursa risk düşük ve bu yöntem akademide yaygın. Videoları **yayımlamak ya da paylaşmak** işin içine girdiğinde izin şartı ciddileşir.

| Konu | Durum | Ne yapmalı |
|---|---|---|
| **Telif (FSEK)** | Haber ve YouTube videoları, sahiplerinin eseri. FSEK'te öğretim amaçlı kullanım için sınırlı istisnalar var, ama bunlar veri setini çoğaltıp dağıtmayı kapsamaz | Videoları yalnızca ekip içinde eğitim için kullanın, **yeniden yayımlamayın**. Veri setini paylaşacaksanız video yerine **link + başlangıç/bitiş saniyesi** paylaşın (Kinetics ve MammalNet de böyle yapıyor) |
| **YouTube kullanım şartları** | YouTube, izin verilmeyen durumlarda video indirmeyi şartlarında yasaklıyor | Mümkünse YouTube'da **"Creative Commons" filtresiyle** arayın. Pixabay ve Pexels gibi telifsiz kaynakları tercih edin |
| **Kişisel veriler (KVKK)** | Videolardaki yüzler ve plakalar kişisel veri. KVKK, anonimleştirilmiş veriyle yapılan araştırma ve istatistik çalışmalarını büyük ölçüde kapsam dışında tutuyor | Klipleri kesince **yüzleri bulanıklaştırın** (otomatik yüz algılama ile kolay). Raporda ve sunumda yalnızca bulanık kareler kullanın |
| **Rahatsız edici içerik** | Kanlı ve ağır yaralanmalı görüntüler etiketleyenleri etkileyebilir | Ağır görüntüleri hariç tutun, etiketlemeyi sırayla yapın |
| **İzin istemek** | En temiz yol bu, ama zaman alır ve yanıt garanti değil | Haber ajanslarına ya da kendi CCTV'si olan bir site yönetimine/belediyeye "ders projesi" diye yazmak, ek ve gerçek veri getirebilir |
| **Kendi çekimimiz** | Kendi kameramızla, gönüllülerle (köpekle oyun, köpek gezdirme) çekilen negatif videolar tamamen bizim | Katılımcılardan basit bir yazılı onay alın |

**Önerim:** Önce hocaya bu tabloyu gösterip onay alın, sonra bunu raporda "Etik ve Hukuki Değerlendirme" başlığı altında yazın. Bu, projenin artısı olarak sayılır.

### 10.2 Videoların koşulları eşit değil: doğruluk nasıl etkilenir?

**En büyük risk "kestirme öğrenme" (shortcut learning).** Model saldırıyı değil, videonun **nereden geldiğini** öğrenebilir. Örneğin:

- Saldırı videoları çoğunlukla düşük çözünürlüklü, sıkıştırılmış, üzerinde haber logosu ve "KAMERADA" yazısı olan CCTV kayıtları.
- Saldırı olmayan videolar (Kinetics, oyun videoları) ise net, renkli, elde çekilmiş telefon görüntüleri.

Bu durumda model "bulanık, üstten çekilmiş, logolu video = saldırı" kuralını öğrenir. **Test doğruluğu %95 çıkabilir ama gerçek bir kamerada sistem işe yaramaz.** Literatürde bu, veri seti yanlılığı (dataset bias) olarak bilinen ve sık rastlanan bir hatadır.

Diğer etkiler:

| Sorun | Doğruluğa etkisi |
|---|---|
| Kamera açısı ve mesafe farkı (üstten CCTV ile göz hizasında telefon) | Model belli bir açıya bağımlı hale gelir, yeni kamerada başarısı düşer |
| Gece/gündüz, kızılötesi (siyah-beyaz) görüntü | Saldırılar gece CCTV'sinde daha sık görünüyorsa model "siyah-beyaz = saldırı" öğrenebilir |
| FPS ve çözünürlük farkı | Hareket hızının algılanması bozulur. 10 FPS'lik CCTV'de saldırı "zıplamalı" görünür |
| Kurgu izleri (kesme, yakınlaştırma, ağır çekim tekrarı, kırmızı daire, yazı) | Model kurgu efektini saldırı işareti sanar |
| Derlemelerde aynı olayın tekrar etmesi | Aynı olay hem eğitimde hem testte olursa doğruluk **yapay olarak şişer** (veri sızıntısı) |
| Seçim yanlılığı: haberlere sadece dramatik saldırılar çıkar | Küçük, kısa ısırmaları ya da başlangıç anını kaçırır |
| Tür dengesizliği (çoğu köpek) | Kedi, at, inek saldırılarında başarı düşük olur |
| K9 eğitim videoları (ısırma kolluğu, eğitim sahası) | Model "kolluk/koruyucu kıyafet = saldırı" öğrenebilir |
| Sınıf dengesizliği | Yalnızca "doğruluk" metriği yanıltıcı olur |

### 10.3 Kaynakların artı ve eksileri

| Kaynak | Artıları | Eksileri |
|---|---|---|
| Haber CCTV kayıtları | Gerçek olay, gerçek kamera açısı, Türkiye bağlamı | Logo ve yazılar, kurgu, düşük kalite, telif, sayısı sınırlı |
| YouTube derlemeleri | Çok sayıda olay, farklı hayvanlar | Tekrarlar, kurgu efektleri, sahte veya yapay zekayla üretilmiş videolar olabilir, telif |
| K9 eğitim videoları | Bol, şiddet yok, saldırı hareketi net | Sahnelenmiş, kolluk ve kıyafet ipucu var, gerçek CCTV'ye benzemez |
| Kinetics (köpek gezdirme vb.) | Etiketli, çeşitli, akademik standart | Telefon/elde çekim, CCTV'den farklı; bazı linkler kırık |
| Oyun videoları | En değerli zor negatifler | Çoğu yakından ve telefonla çekilmiş |
| Sıradan sokak CCTV'si | Pozitiflerle aynı koşullar | Bulması ve kesmesi emek ister |
| Animal Kingdom / MammalNet | Akademik, büyük | İnsan yok, hayvan-hayvan; yalnızca ön eğitim için |
| Sentetik / kendi çekimimiz | İzin sorunu yok, kontrollü | Gerçekçilik sınırlı; tek başına test için kullanılamaz |

> Not: Son dönemde yapay zekayla üretilmiş sahte "hayvan saldırısı" videoları yayılıyor ([örnek](https://www.zmescience.com/ecology/animals-ecology/hyper-realistic-animal-videos/)). YouTube derlemelerindeki klipler kontrol edilmeden alınmamalı; mümkünse haber kaynağıyla doğrulanmış olaylar tercih edilmeli.

### 10.4 Nasıl düzeltiriz?

**1. Negatifleri pozitiflerle aynı kaynaktan almak (en etkili çözüm)**
- Her saldırı CCTV videosunda saldırıdan **önceki saniyeler** (köpek dolaşıyor, insan yürüyor) aynı kamera, ışık ve kaliteyle hazır bir negatif örnektir. Böylece "kamera kalitesi" ipucu iki sınıfta da aynı olur.
- Haber sitelerinden saldırı içermeyen sokak/köpek CCTV videoları da toplanmalı.
- Hedef: Her kaynak türünde pozitif ve negatif oranı birbirine yakın olsun.

**2. Görüntüleri tek bir standarda getirmek (ön işleme)**
- Logo, alt yazı, siyah çerçeve ve "kırmızı daire" gibi eklemeleri kırpmak ya da maskelemek.
- Ağır çekim tekrarlarını ve sahne kesmelerini çıkarmak (sahne değişimi otomatik tespit edilebilir, ör. PySceneDetect).
- Bütün videoları aynı FPS'e (ör. 15), aynı çözünürlüğe ve aynı sıkıştırma ayarına dönüştürmek.

**3. Modeli arka plandan uzaklaştırmak**
- Tüm kareyi değil, YOLO'nun bulduğu **insan + hayvan bölgesini** kırpıp modele vermek.
- Etkileşim özniteliklerini (mesafe, yaklaşma hızı, düşme) kullanmak. Bunlar kamera kalitesinden çok daha az etkilenir.

**4. Veri artırma**
- Eğitimde rastgele bulanıklık, JPEG sıkıştırma, parlaklık/kontrast, gürültü, siyah-beyaz dönüşüm ve yatay çevirme uygulamak. Böylece kalite farkı ayırt edici bir ipucu olmaktan çıkar.

**5. Doğru bölme ve tekrar temizliği**
- Algısal hash (pHash) ile aynı olayın tekrarlarını bulup tek kopyaya indirmek.
- Bölmeyi klibe göre değil **olaya ve kaynağa göre** yapmak (bir olaydan kesilen tüm klipler aynı bölmede).

**6. Doğru değerlendirme**
- Yalnızca doğruluk değil, **F1, PR-AUC ve yanlış alarm oranı** raporlamak.
- Sonuçları kaynağa göre ayrı ayrı vermek (haber CCTV, YouTube, K9, oyun).
- **Kaynak dışı test:** Bir kaynağı eğitimden tamamen çıkarıp sadece onunla test etmek (leave-one-source-out).
- Test setinde yalnızca gerçek CCTV ve oyun videoları olmalı. K9 ve sentetik veri testte hiç kullanılmamalı.

**7. Kestirme öğrenmeyi kontrol etmek (raporda güçlü bir bölüm olur)**
- **Kaynak tahmin testi:** Aynı veriyle "bu video hangi kaynaktan?" diye ayrı bir model eğitin. Bu model çok başarılıysa, kaynaklar arasında belirgin görsel farklar var demektir; ana modelin bunları kullanma riski yüksektir.
- **Grad-CAM / dikkat haritaları:** Modelin logoya, köşedeki saate ya da arka plana değil, hayvana ve insana baktığını göstermek.
- **Karıştırma testi:** Kare sırasını karıştırınca başarı düşmüyorsa, model hareketi değil görüntü dokusunu öğreniyor demektir.

**8. Kendi kontrollü test videolarımız (opsiyonel ama değerli)**
- Ekipten biri telefonla ya da ucuz bir IP kamerayla, sahibinin izniyle köpekle oyun ve yürüyüş videoları çekebilir. Bunlar, modelin "hiç görmediği kamera" üzerinde yanlış alarm verip vermediğini gösterir.

---

### Kaynaklar

- Animal Kingdom: https://sutdcv.github.io/Animal-Kingdom · https://arxiv.org/pdf/2204.08129 · [Ek materyal](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Ng_Animal_Kingdom_A_CVPR_2022_supplemental.pdf)
- MammalNet: https://mammal-net.github.io · https://arxiv.org/pdf/2306.00576
- PanAf20K: https://arxiv.org/pdf/2401.13554
- Roboflow Aggressive dog: https://universe.roboflow.com/fgdgdfg/aggressive-dog
- Çiftlik tehditleri: https://www.mdpi.com/2076-3417/14/14/6098
- Domuz saldırganlığı: https://arxiv.org/html/2403.08528v1 · https://doi.org/10.1080/21642583.2023.2249934
- Sokak köpeği + IoT: https://iieta.org/journals/isi/paper/10.18280/isi.310630
- Fil davranışı: https://www.itm-conferences.org/articles/itmconf/pdf/2026/05/itmconf_issf2026_01012.pdf
- YOLO-Behaviour: https://www.biorxiv.org/content/10.1101/2024.08.26.609387v1
- Anahtar noktalı hayvan eylemi: https://doaj.org/article/ce2b58dc957c486d85541dfd46a003ab
- Köpek eylem tanıma (sentetik veri): https://scitepress.org/PublishedPapers/2025/131950
- Şiddet tespiti derlemesi: https://arxiv.org/pdf/2209.11680
- Real Life Violence Situations veri seti: https://ieee-dataport.org/authors/aparna-bhonde
- Görüntü-metin modelleriyle hayvan davranışı: https://arxiv.org/pdf/2510.09203 · https://arxiv.org/pdf/2503.18223
