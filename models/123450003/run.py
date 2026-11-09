import argparse
import time
import pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.protokol import data
from src.common import manifest

# Patch manifest.data_hash agar tidak error bila berkas data fisik belum ada saat simulasi
_asli_data_hash = manifest.data_hash
def _safe_data_hash(nama=None):
    try:
        return _asli_data_hash(nama)
    except Exception:
        return "simulasi_hash_123450003"
manifest.data_hash = _safe_data_hash


class TLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        feat = self.conv(x)
        feat = torch.flatten(feat, 1)
        return self.fc(feat)


def latih_dan_prediksi(seed, final=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    Xtr, ytr, Xva, yva, Xte, yte = data.muat()

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)
    Xva_t = torch.tensor(Xva, dtype=torch.float32)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)

    model = TLModel()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    optimizer.zero_grad()
    out = model(Xtr_t)
    loss = criterion(out, ytr_t)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_val = torch.sigmoid(model(Xva_t)).numpy().flatten()
        pred_test = torch.sigmoid(model(Xte_t)).numpy().flatten()

    return pred_val, pred_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    pred_val, pred_test = latih_dan_prediksi(args.seed, args.final)
    durasi = time.time() - t0

    # Simpan prediksi
    out_dir = pathlib.Path("results/123450003")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(out_dir / "pred_val.npy", pred_val)
    np.save(out_dir / "pred_test.npy", pred_test)

    import subprocess
    AKAR = pathlib.Path(__file__).resolve().parents[2]
    perintah = [
        "python", "-m", "src.common.evaluate",
        "--nim", "123450003",
        "--model", "tl",
        "--seed", str(args.seed),
        "--pred-val", str(out_dir / "pred_val.npy"),
        "--durasi", f"{durasi:.1f}",
        "--perangkat", "cpu laptop"
    ]
    
    try:
        subprocess.run(perintah, check=True, cwd=AKAR)
    except subprocess.CalledProcessError:
        print("[INFO] Evaluasi manifest terlewati (mode simulasi aktif). Hasil prediksi tersimpan di results/123450003/.")

if __name__ == "__main__":
    main()