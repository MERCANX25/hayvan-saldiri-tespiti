"""
Adim 2: Veri seti olusturma (VGGSound + ESC-50).

Iki kaynak kullanir:
  1) VGGSound: YouTube videolarindan 10 sn'lik etiketli klipler. Saldirgan ve kurban
     seslerinin asil kaynagi. YouTube'dan indirildigi icin yavas ve bazen engellenebilir.
  2) ESC-50: GitHub'dan dogrudan inen 5 sn'lik etiketli sesler (YouTube gerekmez).
     Normal hayvan ve arka plan sesleri icin.

Kullanim:
    python veri_seti_olustur.py --esc50                      # once ESC-50 (hizli, YouTube yok)
    python veri_seti_olustur.py --adet 20                    # VGGSound'dan sinif basina 20 klip
    python veri_seti_olustur.py --adet 20 --tarayici firefox # YouTube "bot" derse cerez kullan

Yarida kesilirse ayni komutla tekrar calistirin; inen dosyalari atlar.
"""

import argparse
import csv
import os
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

VGGSOUND_CSV_URL = "https://raw.githubusercontent.com/hche11/VGGSound/master/data/vggsound.csv"
ESC50_CSV_URL = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv"
ESC50_SES_URL = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/"
ORNEK_HIZI = 16000

GRUPLAR = {
    # A: saldirgan hayvan sesleri
    "saldirgan": ["dog growling", "cat growling", "cat hissing", "lions roaring", "lions growling"],
    # B: kurban / aci sesleri
    "kurban": ["people screaming", "dog whimpering"],
    # C: normal hayvan sesleri (alarm vermemeli)
    "normal_hayvan": [
        "dog barking", "dog bow-wow", "dog howling", "cat meowing", "cat purring",
        "cattle mooing", "pig oinking", "horse neighing", "chicken crowing", "sheep bleating",
    ],
    # D: arka plan (alarm vermemeli)
    "arka_plan": [
        "male speech, man speaking", "female speech, woman speaking", "children shouting",
        "people crowd", "car passing by", "wind noise", "raining", "playing acoustic guitar",
    ],
}

# ESC-50 kategorileri -> bizim gruplarimiz
ESC50_GRUPLAR = {
    "normal_hayvan": ["dog", "cat", "pig", "cow", "rooster", "hen", "sheep"],
    "arka_plan": ["rain", "wind", "footsteps", "laughing", "clapping", "car_horn",
                  "engine", "train", "door_wood_knock", "keyboard_typing"],
}

METADATA_BASLIK = ["dosya", "sinif", "grup", "bolum", "youtube_id", "baslangic"]


class BotEngeli(Exception):
    pass


def dosya_adi(sinif):
    return sinif.replace(", ", "_").replace(" ", "_").replace(",", "")


def wav_kaydet(kaynak, hedef):
    import librosa
    import soundfile as sf

    ses, _ = librosa.load(kaynak, sr=ORNEK_HIZI, mono=True)
    sf.write(hedef, ses, ORNEK_HIZI)


# ---------------------------------------------------------------- ESC-50

def esc50_indir(klasor):
    ses_klasoru = os.path.join(klasor, "ses")
    gecici = os.path.join(klasor, "gecici")
    os.makedirs(ses_klasoru, exist_ok=True)
    os.makedirs(gecici, exist_ok=True)

    print("ESC-50 listesi indiriliyor...")
    with urllib.request.urlopen(ESC50_CSV_URL) as r:
        satirlar = list(csv.DictReader(r.read().decode("utf-8").splitlines()))

    kategori_grubu = {k: g for g, ks in ESC50_GRUPLAR.items() for k in ks}
    kayitlar = []
    for kategori, grup in kategori_grubu.items():
        secilen = [s for s in satirlar if s["category"] == kategori]  # her kategoride 40 klip
        for i, s in enumerate(secilen, 1):
            hedef = os.path.join(ses_klasoru, f"esc50_{kategori}__{s['filename']}")
            if not os.path.exists(hedef):
                ham = os.path.join(gecici, s["filename"])
                urllib.request.urlretrieve(ESC50_SES_URL + s["filename"], ham)
                wav_kaydet(ham, hedef)
                os.remove(ham)
            # ayni kaynaktan gelen kayitlar ayni gruba dussun diye src_file'i kimlik olarak kullaniyoruz
            kayitlar.append([hedef, f"esc50 {kategori}", grup, "esc50", f"esc50_{s['src_file']}", 0])
        print(f"  ESC-50 {kategori} ({grup}): {len(secilen)} klip")

    with open(os.path.join(klasor, "esc50_metadata.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(METADATA_BASLIK)
        w.writerows(kayitlar)
    print(f"ESC-50 tamam: {len(kayitlar)} klip.")


# ---------------------------------------------------------------- VGGSound

def vggsound_listesi(klasor):
    yol = os.path.join(klasor, "vggsound.csv")
    if not os.path.exists(yol):
        print("VGGSound listesi indiriliyor...")
        urllib.request.urlretrieve(VGGSOUND_CSV_URL, yol)
    with open(yol, newline="", encoding="utf-8") as f:
        return [satir for satir in csv.reader(f)]  # yt_id, baslangic, sinif, bolum


def klip_indir(yt_id, baslangic, hedef, gecici_klasor, tarayici, cerez):
    import imageio_ffmpeg
    import yt_dlp

    gecici = os.path.join(gecici_klasor, f"{yt_id}_{baslangic}")
    ayarlar = {
        "format": "bestaudio/best",
        "outtmpl": gecici + ".%(ext)s",
        "download_ranges": yt_dlp.utils.download_range_func(None, [(baslangic, baslangic + 10)]),
        "force_keyframes_at_cuts": True,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        # YouTube'u yormamak icin istekler arasinda bekle
        "sleep_interval": 2,
        "max_sleep_interval": 5,
        "sleep_interval_requests": 1,
    }
    if tarayici:
        ayarlar["cookiesfrombrowser"] = (tarayici,)
    if cerez:
        ayarlar["cookiefile"] = cerez
    try:
        with yt_dlp.YoutubeDL(ayarlar) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={yt_id}"])
    except Exception as e:
        mesaj = str(e)
        if "confirm you" in mesaj or "not a bot" in mesaj or "429" in mesaj:
            raise BotEngeli(mesaj)
        raise

    ham = gecici + ".wav"
    wav_kaydet(ham, hedef)
    os.remove(ham)


def vggsound_indir(args):
    ses_klasoru = os.path.join(args.klasor, "ses")
    gecici_klasor = os.path.join(args.klasor, "gecici")
    os.makedirs(ses_klasoru, exist_ok=True)
    os.makedirs(gecici_klasor, exist_ok=True)

    basarisiz_yol = os.path.join(args.klasor, "silinmis_videolar.txt")
    basarisizlar = set()
    if os.path.exists(basarisiz_yol):
        basarisizlar = set(open(basarisiz_yol, encoding="utf-8").read().split())

    satirlar = vggsound_listesi(args.klasor)
    secili_gruplar = args.gruplar.split(",") if args.gruplar else list(GRUPLAR)
    sinif_grubu = {s: g for g, siniflar in GRUPLAR.items() if g in secili_gruplar for s in siniflar}
    random.seed(42)

    def hedef_yolu(yt_id, bas, sinif):
        return os.path.join(ses_klasoru, f"{dosya_adi(sinif)}__{yt_id}_{bas}.wav")

    # Her sinif icin aday listesi (silinmis videolar icin yedekli)
    adaylar = {}
    for sinif in sinif_grubu:
        liste = [r for r in satirlar if r[2] == sinif]
        if not liste:
            print(f"UYARI: '{sinif}' VGGSound'da bulunamadi")
        random.shuffle(liste)
        adaylar[sinif] = [(r[0], int(r[1]), sinif, sinif_grubu[sinif], r[3]) for r in liste[: int(args.adet * 1.5) + 2]]

    # Siniflari sirayla karistir (round-robin): yarida kesilse bile her siniftan biraz inmis olur
    isler = []
    for i in range(max((len(v) for v in adaylar.values()), default=0)):
        for sinif in sinif_grubu:
            if i < len(adaylar[sinif]):
                isler.append(adaylar[sinif][i])

    # Daha once inmis dosyalari say (onceki calistirmalardan kalanlar dahil)
    bolumler = {(r[0], r[1]): r[3] for r in satirlar}
    tum_siniflar = {s: g for g, siniflar in GRUPLAR.items() for s in siniflar}
    sayac = {s: 0 for s in tum_siniflar}
    kayitlar = []
    mevcut = os.listdir(ses_klasoru)
    for sinif, grup in tum_siniflar.items():  # secilmeyen gruplarin onceki dosyalari da korunur
        onek = f"{dosya_adi(sinif)}__"
        for ad in sorted(a for a in mevcut if a.startswith(onek) and a.endswith(".wav")):
            if sayac[sinif] >= args.adet:
                break
            yt_id, bas = ad[len(onek):-4].rsplit("_", 1)
            sayac[sinif] += 1
            kayitlar.append([os.path.join(ses_klasoru, ad), sinif, grup,
                             bolumler.get((yt_id, bas), "train"), yt_id, int(bas)])

    bekleyen = [
        i for i in isler
        if f"{i[0]}_{i[1]}" not in basarisizlar and not os.path.exists(hedef_yolu(i[0], i[1], i[2]))
    ]
    print(f"VGGSound: {len(sinif_grubu)} sinif, sinif basina en fazla {args.adet} klip. "
          f"Zaten inen: {len(kayitlar)}.")

    bot_sayisi = 0
    durduruldu = False
    with ThreadPoolExecutor(max_workers=args.paralel) as havuz, \
            open(basarisiz_yol, "a", encoding="utf-8") as bf:
        for parca_bas in range(0, len(bekleyen), args.paralel * 4):
            parca = [i for i in bekleyen[parca_bas: parca_bas + args.paralel * 4] if sayac[i[2]] < args.adet]
            gelecekler = {
                havuz.submit(klip_indir, i[0], i[1], hedef_yolu(i[0], i[1], i[2]),
                             gecici_klasor, args.tarayici, args.cerez): i
                for i in parca
            }
            for g in as_completed(gelecekler):
                yt_id, bas, sinif, grup, bolum = gelecekler[g]
                hedef = hedef_yolu(yt_id, bas, sinif)
                try:
                    g.result()
                except BotEngeli:
                    bot_sayisi += 1
                    continue
                except Exception:
                    bf.write(f"{yt_id}_{bas}\n")  # silinmis/ozel video: bir daha deneme
                    continue
                bot_sayisi = 0
                if sayac[sinif] >= args.adet:
                    os.remove(hedef)
                    continue
                sayac[sinif] += 1
                kayitlar.append([hedef, sinif, grup, bolum, yt_id, bas])
                print(f"[{sum(sayac.values())}] {sinif}: {sayac[sinif]}/{args.adet}", flush=True)
            if bot_sayisi >= 5:
                durduruldu = True
                break

    with open(os.path.join(args.klasor, "vggsound_metadata.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(METADATA_BASLIK)
        w.writerows(kayitlar)

    print("\nVGGSound ozeti (sinif: inen klip sayisi):")
    for g in secili_gruplar:
        print(f"  [{g}]")
        for s in GRUPLAR.get(g, []):
            print(f"    {s}: {sayac.get(s, 0)}")

    if durduruldu:
        print("\n!!! YouTube 'bot musunuz' diye indirmeyi engelledi, indirme durduruldu. !!!")
        print("Cozum: 1) 30-60 dakika bekleyip ayni komutu tekrar calistirin (kaldigi yerden devam eder).")
        print("       2) ya da tarayici cerezi ile deneyin: --tarayici firefox  (Firefox'ta YouTube'a giris yapmis olun)")


# ---------------------------------------------------------------- birlestirme

def metadata_birlestir(klasor):
    tum = []
    for ad in ["vggsound_metadata.csv", "esc50_metadata.csv"]:
        yol = os.path.join(klasor, ad)
        if os.path.exists(yol):
            with open(yol, newline="", encoding="utf-8") as f:
                tum += [r for r in csv.reader(f)][1:]
    tum = [r for r in tum if os.path.exists(r[0])]
    with open(os.path.join(klasor, "metadata.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(METADATA_BASLIK)
        w.writerows(tum)

    print("\n=== Toplam veri seti (grup: klip sayisi) ===")
    for g in GRUPLAR:
        print(f"  {g}: {sum(1 for r in tum if r[2] == g)}")
    print(f"Toplam {len(tum)} klip. Liste: {os.path.join(klasor, 'metadata.csv')}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adet", type=int, default=20, help="sinif basina en fazla kac klip")
    p.add_argument("--klasor", default="veri", help="verinin kaydedilecegi klasor")
    p.add_argument("--paralel", type=int, default=2, help="ayni anda kac indirme (YouTube icin 1-2 onerilir)")
    p.add_argument("--esc50", action="store_true", help="sadece ESC-50'yi indir (YouTube gerekmez)")
    p.add_argument("--gruplar", help="sadece bu gruplari YouTube'dan indir, ör: saldirgan,kurban")
    p.add_argument("--tarayici", help="YouTube cerezlerinin alinacagi tarayici: firefox, edge, chrome")
    p.add_argument("--cerez", help="cookies.txt dosyasinin yolu")
    args = p.parse_args()

    os.makedirs(args.klasor, exist_ok=True)
    if args.esc50:
        esc50_indir(args.klasor)
    else:
        vggsound_indir(args)
    metadata_birlestir(args.klasor)


if __name__ == "__main__":
    main()
