"""Kerangka jalur TL (transfer learning) -- Tim 3, kata bangun 'Halo Meja'.

Salin file ini ke models/<nim>/run.py. NIM diambil otomatis dari nama folder.

Jalankan dari akar repo:
    python models/<nim>/run.py --seed 1

Status: KERANGKA. Alur dari data sampai berkas hasil sudah jalan; backbone
masih ResNet18 ImageNet sebagai placeholder dan harus diganti backbone
keyword spotting pralatih pada sprint berikutnya.
"""
import argparse
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.protokol.data import muat  # noqa: E402  (pemuat data beku milik tim)

NIM = Path(__file__).resolve().parent.name
JALUR = "tl"
HASIL = ROOT / "results" / NIM


# Argumen dan seed
def baca_argumen():
    p = argparse.ArgumentParser(description="Jalur TL - kata bangun 'Halo Meja'")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epoch", type=int, default=2, help="cukup 1-2 untuk kerangka")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--perangkat", default=None,
                   help="cpu / cuda; default: otomatis")
    p.add_argument("--tanpa-pralatih", action="store_true",
                   help="jangan unduh bobot pralatih (untuk tes offline)")
    p.add_argument("--fine-tune", action="store_true",
                   help="buka seluruh backbone (default: hanya kepala yang dilatih)")
    return p.parse_args()


def atur_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Data
def siapkan_x(x):
    """Pastikan bentuk (N, 1, pita_mel, waktu) dan float32.

    ASUMSI: muat() mengembalikan log-mel berbentuk (N, 64, T) atau
    (N, 1, 64, T). Kalau bentuk dari data.py berbeda, ubah di sini.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3:
        x = x[:, None, :, :]
    if x.ndim != 4:
        raise ValueError(f"Bentuk X tidak dikenali: {x.shape}. "
                         "Harapannya (N, 64, T) atau (N, 1, 64, T).")
    return x


def buat_loader(x, y, batch, acak):
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(np.asarray(y, dtype=np.float32)))
    return DataLoader(ds, batch_size=batch, shuffle=acak)


def bangun_model(pralatih, fine_tune):
    """PLACEHOLDER: ResNet18 ImageNet dengan masukan 1 kanal (log-mel).

    TODO sprint berikutnya: ganti dengan backbone keyword spotting pralatih
    (PROTOKOL/bagian 7 PDF: dataset KWS publik boleh untuk pralatih).
    """
    from torchvision import models

    bobot = None
    if pralatih:
        try:
            bobot = models.ResNet18_Weights.IMAGENET1K_V1
        except AttributeError:
            bobot = None
    try:
        net = models.resnet18(weights=bobot)
    except Exception as e:  # mis. tidak ada internet untuk unduh bobot
        print(f"[peringatan] gagal memuat bobot pralatih ({e}); "
              "memakai bobot acak. Ini BUKAN transfer learning.")
        net = models.resnet18(weights=None)

    # conv1 dari 3 kanal -> 1 kanal: jumlahkan bobot per kanal agar tetap berguna
    lama = net.conv1
    baru = nn.Conv2d(1, lama.out_channels, lama.kernel_size, lama.stride,
                     lama.padding, bias=False)
    with torch.no_grad():
        baru.weight.copy_(lama.weight.sum(dim=1, keepdim=True))
    net.conv1 = baru

    # kepala biner (dropout = regularisasi pertama; catat di catatan.md)
    net.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(net.fc.in_features, 1))

    if not fine_tune:  # tahap 1: bekukan backbone, latih kepala saja
        for nama, param in net.named_parameters():
            param.requires_grad = nama.startswith("fc")
    return net


def latih(net, loader, epoch, lr, pos_weight, perangkat):
    net.to(perangkat).train()
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr)
    # pos_weight: kelas positif ('Halo Meja') jauh lebih jarang daripada negatif
    kriteria = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=perangkat))
    for e in range(1, epoch + 1):
        total, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(perangkat), yb.to(perangkat)
            opt.zero_grad()
            loss = kriteria(net(xb).squeeze(1), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
            n += len(xb)
        print(f"epoch {e}/{epoch}  loss={total / max(n, 1):.4f}")


@torch.no_grad()
def prediksi(net, x, batch, perangkat):
    """Mengembalikan probabilitas kelas positif, bentuk (N,)."""
    net.to(perangkat).eval()
    keluar = []
    for i in range(0, len(x), batch):
        xb = torch.from_numpy(x[i:i + batch]).to(perangkat)
        keluar.append(torch.sigmoid(net(xb).squeeze(1)).cpu().numpy())
    return np.concatenate(keluar)


def tulis_hasil(seed, durasi, perangkat):
    """Panggil penilai starter sebagai subprocess.

    evaluate mengambil label dari split beku sendiri dan menghitung data_hash
    sendiri, jadi kita hanya mengirim berkas prediksi. Tidak ada try/except:
    kalau evaluate gagal, skrip ikut gagal dengan pesan error aslinya.

    ASUMSI: nama argumen CLI di bawah. Cocokkan dengan
    `python -m src.common.evaluate --help` bila ada yang beda.
    """
    perintah = [
        sys.executable, "-m", "src.common.evaluate",
        "--nim", NIM,
        "--model", JALUR,
        "--seed", str(seed),
        "--pred-val", str(HASIL / "pred_val.npy"),
        "--durasi", f"{durasi:.1f}",
        "--perangkat", perangkat,
    ]
    subprocess.run(perintah, check=True, cwd=ROOT)


def main():
    args = baca_argumen()
    atur_seed(args.seed)
    perangkat = args.perangkat or ("cuda" if torch.cuda.is_available() else "cpu")
    mulai = time.time()

    # 1. Data (split beku seed 2026 sudah di dalam muat(); jangan split ulang)
    X_tr, y_tr, X_va, y_va, X_te, y_te = muat()
    X_tr, X_va, X_te = (siapkan_x(a) for a in (X_tr, X_va, X_te))

    # normalisasi memakai statistik DATA LATIH saja
    mu, sd = X_tr.mean(), X_tr.std() + 1e-6
    X_tr, X_va, X_te = ((a - mu) / sd for a in (X_tr, X_va, X_te))

    # 2. Model
    net = bangun_model(pralatih=not args.tanpa_pralatih, fine_tune=args.fine_tune)

    # 3. Latih singkat
    y_tr_arr = np.asarray(y_tr)
    n_pos = max(int((y_tr_arr == 1).sum()), 1)
    n_neg = max(int((y_tr_arr == 0).sum()), 1)
    latih(net, buat_loader(X_tr, y_tr, args.batch, acak=True),
          args.epoch, args.lr, pos_weight=n_neg / n_pos, perangkat=perangkat)

    # 4. Prediksi validasi dan uji
    p_val = prediksi(net, X_va, args.batch, perangkat)
    p_test = prediksi(net, X_te, args.batch, perangkat)

    # 5. Simpan prediksi (data uji TIDAK dipakai untuk metrik/ambang di sini;
    #    evaluasi uji dilakukan sekali oleh Penjaga Data pada sprint terakhir)
    HASIL.mkdir(parents=True, exist_ok=True)
    np.save(HASIL / "pred_val.npy", p_val)
    np.save(HASIL / "pred_test.npy", p_test)

    # 6. Metrik via evaluate.py + catat durasi dan perangkat
    durasi = time.time() - mulai
    tulis_hasil(args.seed, durasi, perangkat)
    print(f"selesai: seed={args.seed} durasi={durasi:.1f}s perangkat={perangkat} -> {HASIL}")


if __name__ == "__main__":
    main()
