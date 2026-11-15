"""
Adim 1: YAMNet ile hizli test (egitim yok).

YAMNet, AudioSet'in 521 ses sinifini taniyan hazir bir modeldir.
Bu betik sesi 1 saniyelik pencerelere bolup saldiriyla ilgili
siniflarin skorlarini ekrana yazar ve basit bir alarm verir.

Kullanim:
    # Ses cihazlarini listele
    python yamnet_test.py --cihazlar

    # Bilgisayarin hoparloründen cikan sesi canli dinle (YouTube'u tarayicida ac)
    python yamnet_test.py --canli
    python yamnet_test.py --canli --cihaz 2      # listedeki 2 numarali cikisi dinle

    # Bir ses/video dosyasini analiz et (wav, mp3, mp4 ...)
    python yamnet_test.py --dosya ornek.wav
"""

import argparse
import csv
import queue
import sys
import threading
import time
import warnings

import numpy as np

ORNEK_HIZI = 16000  # YAMNet 16 kHz mono ister
PENCERE_SN = 1.0
ADIM_SN = 0.5

# Saldirgan hayvan sesleri (A grubu)
SALDIRGAN = ["Growling", "Roar", "Hiss", "Caterwaul"]
# Kurban / aci sesleri (B grubu)
KURBAN = ["Screaming", "Whimper (dog)", "Yell", "Crying, sobbing"]

ESIK_SALDIRGAN = 0.20
ESIK_KURBAN = 0.20
KURBAN_PENCERESI_SN = 4.0  # saldirgan sesten sonra kac saniye icinde kurban sesi aranir
SESSIZLIK_ESIGI = 0.001  # bunun altindaki ses seviyesi "sessiz" sayilir


def model_yukle():
    import tensorflow_hub as hub
    import tensorflow as tf

    print("YAMNet yukleniyor (ilk seferde internetten iner)...")
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    sinif_dosyasi = model.class_map_path().numpy().decode("utf-8")
    with tf.io.gfile.GFile(sinif_dosyasi) as f:
        siniflar = [satir["display_name"] for satir in csv.DictReader(f)]
    return model, siniflar


def indeksler(siniflar, isimler):
    return {isim: siniflar.index(isim) for isim in isimler if isim in siniflar}


class Alarm:
    def __init__(self):
        self.son_saldirgan_zamani = None

    def guncelle(self, zaman, saldirgan_skor, kurban_skor):
        durum = "normal"
        if saldirgan_skor >= ESIK_SALDIRGAN:
            self.son_saldirgan_zamani = zaman
            durum = "AGRESYON"
        if (
            kurban_skor >= ESIK_KURBAN
            and self.son_saldirgan_zamani is not None
            and zaman - self.son_saldirgan_zamani <= KURBAN_PENCERESI_SN
        ):
            durum = "!!! SALDIRI OLASI !!!"
        return durum


def pencereyi_isle(model, siniflar, ses, idx_a, idx_b):
    skorlar, _, _ = model(ses.astype(np.float32))
    ort = skorlar.numpy().mean(axis=0)  # pencere icindeki karelerin ortalamasi
    a = {k: float(ort[i]) for k, i in idx_a.items()}
    b = {k: float(ort[i]) for k, i in idx_b.items()}
    en_iyi = int(np.argmax(ort))
    return a, b, (siniflar[en_iyi], float(ort[en_iyi]))


def satir_yaz(zaman, seviye, a, b, en_iyi, durum):
    en_a = max(a, key=a.get)
    en_b = max(b, key=b.get)
    not_ = "  (SESSIZ: ses yakalanmiyor)" if seviye < SESSIZLIK_ESIGI else ""
    print(
        f"{zaman:7.1f}s | seviye {seviye:.3f} | en olasi: {en_iyi[0][:18]:<18} {en_iyi[1]:.2f} "
        f"| saldirgan: {en_a:<9} {a[en_a]:.2f} | kurban: {en_b:<14} {b[en_b]:.2f} | {durum}{not_}",
        flush=True,
    )


def isle_ve_yaz(model, siniflar, ses, zaman, idx_a, idx_b, alarm):
    seviye = float(np.sqrt(np.mean(ses**2)))
    a, b, en_iyi = pencereyi_isle(model, siniflar, ses, idx_a, idx_b)
    durum = alarm.guncelle(zaman, max(a.values()), max(b.values()))
    satir_yaz(zaman, seviye, a, b, en_iyi, durum)


def dosya_modu(yol, model, siniflar, idx_a, idx_b):
    import librosa

    ses, _ = librosa.load(yol, sr=ORNEK_HIZI, mono=True)
    pencere = int(PENCERE_SN * ORNEK_HIZI)
    adim = int(ADIM_SN * ORNEK_HIZI)
    alarm = Alarm()
    for bas in range(0, max(1, len(ses) - pencere + 1), adim):
        isle_ve_yaz(model, siniflar, ses[bas : bas + pencere], bas / ORNEK_HIZI, idx_a, idx_b, alarm)


def cihazlari_listele():
    import soundcard as sc

    print("Hoparlor / ses cikislari (--cihaz ile numarasini verin):")
    for i, h in enumerate(sc.all_speakers()):
        isaret = "  <- varsayilan" if h.name == sc.default_speaker().name else ""
        print(f"  {i}: {h.name}{isaret}")


def canli_mod(model, siniflar, idx_a, idx_b, cihaz_no):
    import soundcard as sc

    warnings.filterwarnings("ignore", message="data discontinuity")
    hoparlor = sc.all_speakers()[cihaz_no] if cihaz_no is not None else sc.default_speaker()
    print(f"Dinlenen ses cikisi: {hoparlor.name}")
    print("YouTube videosunu BU cikistan oynatin. Durdurmak icin Ctrl+C.\n")
    mikrofon = sc.get_microphone(id=str(hoparlor.name), include_loopback=True)

    pencere = int(PENCERE_SN * ORNEK_HIZI)
    adim = int(ADIM_SN * ORNEK_HIZI)
    kuyruk = queue.Queue()

    # Kayit ayri bir is parcaciginda surekli calisir; model yavas kalsa bile ses kaybolmaz.
    def kaydet():
        with mikrofon.recorder(samplerate=ORNEK_HIZI) as kayit:
            while True:
                veri = kayit.record(numframes=adim)
                kuyruk.put(veri.mean(axis=1).astype(np.float32))  # tum kanallarin ortalamasi = mono

    threading.Thread(target=kaydet, daemon=True).start()

    tampon = np.zeros(pencere, dtype=np.float32)
    alarm = Alarm()
    basla = time.time()
    while True:
        yeni = kuyruk.get()
        while not kuyruk.empty():  # geride kaldiysak en guncel sese atla
            yeni = kuyruk.get()
        tampon = np.concatenate([tampon, yeni])[-pencere:]
        isle_ve_yaz(model, siniflar, tampon, time.time() - basla, idx_a, idx_b, alarm)


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--canli", action="store_true", help="hoparlor cikisini canli dinle")
    g.add_argument("--dosya", help="analiz edilecek ses/video dosyasi")
    g.add_argument("--cihazlar", action="store_true", help="ses cikislarini listele")
    p.add_argument("--cihaz", type=int, help="--cihazlar listesindeki cikis numarasi")
    args = p.parse_args()

    if args.cihazlar:
        cihazlari_listele()
        return

    model, siniflar = model_yukle()
    idx_a = indeksler(siniflar, SALDIRGAN)
    idx_b = indeksler(siniflar, KURBAN)
    print("Takip edilen saldirgan siniflar:", list(idx_a))
    print("Takip edilen kurban siniflari:  ", list(idx_b))

    try:
        if args.canli:
            canli_mod(model, siniflar, idx_a, idx_b, args.cihaz)
        else:
            dosya_modu(args.dosya, model, siniflar, idx_a, idx_b)
    except KeyboardInterrupt:
        print("\nDurduruldu.")
        sys.exit(0)


if __name__ == "__main__":
    main()
