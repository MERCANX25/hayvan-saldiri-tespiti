"""
Adim 4: Egitilmis modelle canli demo.

Colab'da egitilen modeli kullanarak bilgisayarin hoparloründen cikan sesi
(ör. YouTube videosu) canli dinler ve 4 grubun olasiligini ekrana yazar:
saldirgan, kurban, normal_hayvan, arka_plan.

Gerekenler: Colab'in Drive'a kaydettigi "model" klasoru bu betikle ayni klasorde olmali:
    model/hayvan_ses_agirliklari.npz
    model/ayarlar.json

Kullanim:
    python canli_demo.py --cihazlar               # ses cikislarini listele
    python canli_demo.py --canli                  # varsayilan cikisi dinle
    python canli_demo.py --canli --cihaz 1        # 1 numarali cikisi dinle
    python canli_demo.py --dosya ornek.wav        # bir dosyayi analiz et
"""

import argparse
import json
import os
import queue
import sys
import threading
import time
import warnings
from collections import deque

import numpy as np

ORNEK_HIZI = 16000
PENCERE_SN = 1.0
ADIM_SN = 0.5
YUMUSATMA = 3  # son kac pencerenin ortalamasi alinsin
SESSIZ_ESIK = 0.001  # bu seviyenin altindaki pencerelerde karar verilmez


class Siniflandirici:
    """Colab'da egitilen kucuk agin numpy ile calistirilmasi (Keras surumunden bagimsiz)."""

    def __init__(self, klasor):
        w = np.load(os.path.join(klasor, "hayvan_ses_agirliklari.npz"))
        self.W1, self.b1, self.W2, self.b2 = w["W1"], w["b1"], w["W2"], w["b2"]
        with open(os.path.join(klasor, "ayarlar.json"), encoding="utf-8") as f:
            self.ayarlar = json.load(f)
        self.gruplar = self.ayarlar["gruplar"]

    def olasilik(self, gomuler):
        gizli = np.maximum(0, gomuler @ self.W1 + self.b1)  # Dense + ReLU (Dropout tahminde kapali)
        z = gizli @ self.W2 + self.b2
        z = np.exp(z - z.max(axis=1, keepdims=True))  # Softmax
        return (z / z.sum(axis=1, keepdims=True)).mean(axis=0)


class Alarm:
    def __init__(self, ayarlar):
        self.a = ayarlar
        self.son_saldirgan = None

    def guncelle(self, t, p_saldirgan, p_kurban):
        durum = "normal"
        if p_saldirgan >= self.a["esik_saldirgan"]:
            self.son_saldirgan, durum = t, "AGRESYON"
        if (
            p_kurban >= self.a["esik_kurban"]
            and self.son_saldirgan is not None
            and t - self.son_saldirgan <= self.a["kurban_penceresi_sn"]
        ):
            durum = "!!! SALDIRI OLASI !!!"
        return durum


def yamnet_yukle():
    import tensorflow_hub as hub

    print("YAMNet yukleniyor...")
    return hub.load("https://tfhub.dev/google/yamnet/1")


class Analizci:
    def __init__(self, yamnet, sinif):
        self.yamnet, self.sinif = yamnet, sinif
        self.gecmis = deque(maxlen=YUMUSATMA)
        self.alarm = Alarm(sinif.ayarlar)

    def isle(self, ses, t):
        seviye = float(np.sqrt(np.mean(ses**2)))
        if seviye < SESSIZ_ESIK:
            # Ses yoksa karar verme: sessizlik "normal" demek degildir
            print(f"{t:7.1f}s | seviye {seviye:.3f} | SES YOK - karar verilmedi", flush=True)
            return
        _, gomu, _ = self.yamnet(ses.astype(np.float32))
        self.gecmis.append(self.sinif.olasilik(gomu.numpy()))
        p = np.mean(self.gecmis, axis=0)
        durum = self.alarm.guncelle(t, p[0], p[1])
        parcalar = " | ".join(f"{g} {v:.2f}" for g, v in zip(self.sinif.gruplar, p))
        print(f"{t:7.1f}s | seviye {seviye:.3f} | {parcalar} | {durum}", flush=True)


def dosya_modu(yol, analizci):
    import librosa

    ses, _ = librosa.load(yol, sr=ORNEK_HIZI, mono=True)
    pencere, adim = int(PENCERE_SN * ORNEK_HIZI), int(ADIM_SN * ORNEK_HIZI)
    for bas in range(0, max(1, len(ses) - pencere + 1), adim):
        analizci.isle(ses[bas : bas + pencere], bas / ORNEK_HIZI)


def cihazlari_listele():
    import soundcard as sc

    print("Hoparlor / ses cikislari (--cihaz ile numarasini verin):")
    for i, h in enumerate(sc.all_speakers()):
        isaret = "  <- varsayilan" if h.name == sc.default_speaker().name else ""
        print(f"  {i}: {h.name}{isaret}")


def canli_mod(analizci, cihaz_no):
    import soundcard as sc

    warnings.filterwarnings("ignore", message="data discontinuity")
    hoparlor = sc.all_speakers()[cihaz_no] if cihaz_no is not None else sc.default_speaker()
    print(f"Dinlenen ses cikisi: {hoparlor.name}")
    print("YouTube videosunu BU cikistan oynatin. Durdurmak icin Ctrl+C.\n")
    mikrofon = sc.get_microphone(id=str(hoparlor.name), include_loopback=True)

    pencere, adim = int(PENCERE_SN * ORNEK_HIZI), int(ADIM_SN * ORNEK_HIZI)
    kuyruk = queue.Queue()

    def kaydet():
        with mikrofon.recorder(samplerate=ORNEK_HIZI) as kayit:
            while True:
                kuyruk.put(kayit.record(numframes=adim).mean(axis=1).astype(np.float32))

    threading.Thread(target=kaydet, daemon=True).start()
    tampon = np.zeros(pencere, dtype=np.float32)
    basla = time.time()
    while True:
        yeni = kuyruk.get()
        while not kuyruk.empty():
            yeni = kuyruk.get()
        tampon = np.concatenate([tampon, yeni])[-pencere:]
        analizci.isle(tampon, time.time() - basla)


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--canli", action="store_true")
    g.add_argument("--dosya")
    g.add_argument("--cihazlar", action="store_true")
    p.add_argument("--cihaz", type=int)
    p.add_argument("--model", default="model", help="Colab'dan indirilen model klasoru")
    args = p.parse_args()

    if args.cihazlar:
        cihazlari_listele()
        return

    sinif = Siniflandirici(args.model)
    print(f"Model yuklendi. Test dogrulugu (Colab): {sinif.ayarlar.get('test_dogrulugu')}")
    analizci = Analizci(yamnet_yukle(), sinif)
    try:
        if args.canli:
            canli_mod(analizci, args.cihaz)
        else:
            dosya_modu(args.dosya, analizci)
    except KeyboardInterrupt:
        print("\nDurduruldu.")
        sys.exit(0)


if __name__ == "__main__":
    main()
