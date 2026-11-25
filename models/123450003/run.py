"""Kerangka jalur TL (transfer learning) -- Tim 3, kata bangun 'Halo Meja'."""
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


def baca_argumen():
    p = argparse.ArgumentParser(description="Jalur TL - kata bangun 'Halo Meja'")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epoch", type=int, default=5, help="jumlah epoch pelatihan")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--perangkat", default=None, help="cpu / cuda; default: otomatis")
    p.add_argument("--tanpa-pralatih", action="store_true", help="jangan unduh bobot pralatih")
    p.add_argument("--fine-tune", action="store_true", help="buka seluruh backbone")
    return p.parse_args()


def atur_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def siapkan_x(x):
    """Pastikan bentuk (N, 1, pita_mel, waktu) dan float32."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3:
        x = x[:, None, :, :]
    if x.ndim != 4:
        raise ValueError(f"Bentuk X tidak dikenali: {x.shape}. Harapannya (N, 64, T) atau (N, 1, 64, T).")
    return x


def buat_loader(x, y, batch, acak):
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(np.asarray(y, dtype=np.float32)))
    return DataLoader(ds, batch_size=batch, shuffle=acak)


def bangun_model(pralatih, fine_tune):
    """ResNet18 ImageNet dengan masukan 1 kanal (log-mel) dan regularisasi Dropout(0.3)."""
    from torchvision import models

    bobot = None
    if pralatih:
        try:
            bobot = models.ResNet18_Weights.IMAGENET1K_V1
        except AttributeError:
            bobot = None
    try:
        net = models.resnet18(weights=bobot)
    except Exception as e:
        print(f"[peringatan] gagal memuat bobot pralatih ({e}); memakai bobot acak.")
        net = models.resnet18(weights=None)

    lama = net.conv1
    baru = nn.Conv2d(1, lama.out_channels, lama.kernel_size, lama.stride, lama.padding, bias=False)
    with torch.no_grad():
        baru.weight.copy_(lama.weight.sum(dim=1, keepdim=True))
    net.conv1 = baru

    net.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(net.fc.in_features, 1))

    if not fine_tune:
        for nama, param in net.named_parameters():
            param.requires_grad = nama.startswith("fc")
    return net


def latih(net, loader, epoch, lr, pos_weight, perangkat):
    net.to(perangkat).train()
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr)
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
    """Panggil penilai starter sebagai subprocess."""
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

    # 1. Data
    X_tr, y_tr, X_va, y_va, X_te, y_te = muat()
    X_tr, X_va, X_te = (siapkan_x(a) for a in (X_tr, X_va, X_te))

    mu, sd = X_tr.mean(), X_tr.std() + 1e-6
    X_tr, X_va, X_te = ((a - mu) / sd for a in (X_tr, X_va, X_te))

    # 2. Model
    net = bangun_model(pralatih=not args.tanpa_pralatih, fine_tune=args.fine_tune)

    # 3. Latih
    y_tr_arr = np.asarray(y_tr)
    n_pos = max(int((y_tr_arr == 1).sum()), 1)
    n_neg = max(int((y_tr_arr == 0).sum()), 1)
    latih(net, buat_loader(X_tr, y_tr, args.batch, acak=True),
          args.epoch, args.lr, pos_weight=n_neg / n_pos, perangkat=perangkat)

    # 4. Prediksi validasi dan uji (konversi ke kelas biner 0/1)
    prob_val = prediksi(net, X_va, args.batch, perangkat)
    prob_test = prediksi(net, X_te, args.batch, perangkat)
    p_val = (prob_val >= 0.5).astype(np.int64)
    p_test = (prob_test >= 0.5).astype(np.int64)

    # Hitung metrik khusus Tim 3 (Recall & Alarm palsu per jam)
    y_va_arr = np.asarray(y_va).ravel()
    tp = int(((p_val == 1) & (y_va_arr == 1)).sum())
    fn = int(((p_val == 0) & (y_va_arr == 1)).sum())
    fp = int(((p_val == 1) & (y_va_arr == 0)).sum())
    n_neg_val = int((y_va_arr == 0).sum())
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fah = fp / (n_neg_val / 3600.0) if n_neg_val > 0 else 0.0
    print(f"[Metrik Tim 3] val_recall={recall:.4f} | val_alarm_palsu_per_jam={fah:.2f}")

    # 5. Simpan prediksi
    HASIL.mkdir(parents=True, exist_ok=True)
    np.save(HASIL / "pred_val.npy", p_val)
    np.save(HASIL / "pred_test.npy", p_test)

    # 6. Metrik via evaluate.py
    durasi = time.time() - mulai
    tulis_hasil(args.seed, durasi, perangkat)
    print(f"selesai: seed={args.seed} durasi={durasi:.1f}s perangkat={perangkat} -> {HASIL}")


if __name__ == "__main__":
    main()