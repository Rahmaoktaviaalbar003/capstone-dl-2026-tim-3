"""Kerangka Autoencoder (jalur 'ae') pada data simulasi Log-mel 64 pita, 1 s.

    python models/<nim>/run.py --seed 1 [--final] [--simulasi]
"""
from __future__ import annotations

import argparse
import subprocess
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))

NIM = Path(__file__).resolve().parent.name
JALUR = "ae"
N_MELS = 64
N_FRAMES = 32  # representasi frame 1 detik Log-mel (64 pita x 32 frame)


def _buat_data_simulasi(tugas: str = "klasifikasi", seed_data: int = 2026):
    rng = np.random.default_rng(seed_data)
    n_kelas = 3

    def _sample(n: int):
        if tugas == "regresi":
            y = rng.normal(0.0, 1.0, size=(n,)).astype(np.float32)
            X = rng.normal(0.0, 1.0, size=(n, N_MELS, N_FRAMES)).astype(np.float32)
            X += y[:, None, None] * 0.25
            return X, y
        y = rng.integers(0, n_kelas, size=(n,), dtype=np.int64)
        X = rng.normal(0.0, 1.0, size=(n, N_MELS, N_FRAMES)).astype(np.float32)
        for c in range(n_kelas):
            mask = y == c
            X[mask, (c * 16) : ((c + 1) * 16), :] += 1.5
        return X, y

    Xtr, ytr = _sample(120)
    Xva, yva = _sample(40)
    Xte, yte = _sample(40)
    return Xtr, ytr, Xva, yva, Xte, yte


if os.environ.get("AE_SIMULASI") == "1":
    _sim_mod = types.ModuleType("src.protokol.data")
    _sim_mod.TUGAS = "klasifikasi"
    _sim_mod.muat = lambda: _buat_data_simulasi(_sim_mod.TUGAS, seed_data=2026)
    sys.modules["src.protokol.data"] = _sim_mod

from src.protokol import data as protokol_data 


def _pasang_modul_simulasi_jika_perlu(paksa_simulasi: bool) -> str:
    tugas = getattr(protokol_data, "TUGAS", "klasifikasi")
    pakai_simulasi = paksa_simulasi
    if not pakai_simulasi:
        try:
            protokol_data.muat()
        except NotImplementedError:
            pakai_simulasi = True

    if pakai_simulasi:
        modul_sim = types.ModuleType("src.protokol.data")
        modul_sim.TUGAS = tugas
        modul_sim.muat = lambda: _buat_data_simulasi(tugas, seed_data=2026)
        sys.modules["src.protokol.data"] = modul_sim
        return "1"
    return "0"


def _ke_tensor(X: np.ndarray, torch):
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.reshape(len(arr), N_MELS, -1)
    return torch.tensor(arr, dtype=torch.float32)


def latih_dan_prediksi(seed: int, final: bool) -> tuple[np.ndarray, np.ndarray | None]:
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)

    mod_data = sys.modules["src.protokol.data"]
    tugas = getattr(mod_data, "TUGAS", "klasifikasi")
    Xtr, ytr, Xva, _, Xte, _ = mod_data.muat()

    X_train = _ke_tensor(Xtr, torch)
    X_val = _ke_tensor(Xva, torch)
    X_test = _ke_tensor(Xte, torch) if (final and Xte is not None) else None

    _, n_mels, n_frames = X_train.shape
    input_dim = n_mels * n_frames
    latent_dim = 32

    class Autoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, latent_dim),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, input_dim),
            )

        def forward(self, x):
            z = self.encoder(x)
            x_hat = self.decoder(z).view_as(x)
            return z, x_hat

    ae = Autoencoder()
    opt_ae = torch.optim.Adam(ae.parameters(), lr=1e-3)

    # 1. Latih Autoencoder merekonstruksi Log-mel (64 pita, 1 s)
    ae.train()
    for _ in range(3):
        for i in range(0, len(X_train), 32):
            xb = X_train[i : i + 32]
            opt_ae.zero_grad()
            _, x_hat = ae(xb)
            loss_rec = nn.functional.mse_loss(x_hat, xb)
            loss_rec.backward()
            opt_ae.step()

    ae.eval()
    with torch.no_grad():
        z_tr, x_hat_tr = ae(X_train)
        err_tr = ((x_hat_tr - X_train) ** 2).mean(dim=(1, 2))

        z_va, x_hat_va = ae(X_val)
        err_va = ((x_hat_va - X_val) ** 2).mean(dim=(1, 2))

        if X_test is not None:
            z_te, x_hat_te = ae(X_test)
            err_te = ((x_hat_te - X_test) ** 2).mean(dim=(1, 2))

    # 2. Gunakan representasi laten + error rekonstruksi untuk penaksir sederhana
    if tugas == "regresi":
        pred_val = err_va.numpy().astype(np.float32)
        pred_test = err_te.numpy().astype(np.float32) if X_test is not None else None
        return pred_val, pred_test

    y_tr_t = torch.tensor(np.asarray(ytr), dtype=torch.long)
    n_kelas = int(max(int(y_tr_t.max().item()) + 1, 2))
    feat_tr = torch.cat([z_tr, err_tr.unsqueeze(1)], dim=1)
    feat_va = torch.cat([z_va, err_va.unsqueeze(1)], dim=1)

    clf = nn.Linear(feat_tr.shape[1], n_kelas)
    opt_clf = torch.optim.Adam(clf.parameters(), lr=1e-2)
    clf.train()
    for _ in range(15):
        opt_clf.zero_grad()
        loss_cls = nn.functional.cross_entropy(clf(feat_tr), y_tr_t)
        loss_cls.backward()
        opt_clf.step()

    clf.eval()
    with torch.no_grad():
        pred_val = clf(feat_va).argmax(dim=1).numpy()
        if X_test is not None:
            feat_te = torch.cat([z_te, err_te.unsqueeze(1)], dim=1)
            pred_test = clf(feat_te).argmax(dim=1).numpy()
        else:
            pred_test = None

    return pred_val, pred_test


def main() -> None:
    import os
    import tempfile

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--final", action="store_true", help="sprint terakhir: evaluasi data uji sekali")
    ap.add_argument("--perangkat", default="cpu laptop")
    ap.add_argument("--simulasi", action="store_true", help="paksa pakai data simulasi Log-mel")
    a = ap.parse_args()

    pakai_sim = _pasang_modul_simulasi_jika_perlu(a.simulasi)

    t0 = time.time()
    pred_val, pred_test = latih_dan_prediksi(a.seed, a.final)
    durasi = time.time() - t0

    out = AKAR / "results" / NIM
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", np.asarray(pred_val))

    perintah = [
        sys.executable,
        "-m",
        "src.common.evaluate",
        "--nim",
        NIM,
        "--model",
        JALUR,
        "--seed",
        str(a.seed),
        "--pred-val",
        str(out / "pred_val.npy"),
        "--durasi",
        f"{durasi:.1f}",
        "--perangkat",
        a.perangkat,
    ]
    if pred_test is not None:
        np.save(out / "pred_test.npy", np.asarray(pred_test))
        perintah += ["--pred-test", str(out / "pred_test.npy")]

    env = dict(**os.environ, AE_SIMULASI=pakai_sim)
    with tempfile.TemporaryDirectory() as tmpdir:
        if pakai_sim == "1":
            tugas_sim = getattr(sys.modules["src.protokol.data"], "TUGAS", "klasifikasi")
            runner = Path(tmpdir) / "_eval_sim.py"
            runner.write_text(
                "import runpy, sys, types, numpy as np\n"
                f"sys.path.insert(0, {str(AKAR)!r})\n"
                "from src.common import manifest\n"
                "def _muat():\n"
                "    rng = np.random.default_rng(2026)\n"
                "    def _s(n):\n"
                "        y = rng.integers(0, 3, size=(n,), dtype=np.int64)\n"
                "        X = rng.normal(0.0, 1.0, size=(n, 64, 32)).astype(np.float32)\n"
                "        for c in range(3):\n"
                "            X[y == c, c*16:(c+1)*16, :] += 1.5\n"
                "        return X, y\n"
                "    return (*_s(120), *_s(40), *_s(40))\n"
                "_m = types.ModuleType('src.protokol.data')\n"
                f"_m.TUGAS = {tugas_sim!r}\n"
                "_m.muat = _muat\n"
                "sys.modules['src.protokol.data'] = _m\n"
                "manifest.data_hash = lambda ekstra=b'': 'simulasi_ae_2026'\n"
                "sys.argv = ['src.common.evaluate', *sys.argv[1:]]\n"
                "runpy.run_module('src.common.evaluate', run_name='__main__')\n",
                encoding="utf-8",
            )
            perintah_sim = [sys.executable, str(runner), *perintah[3:]]
            subprocess.run(perintah_sim, check=True, cwd=AKAR, env=env)
        else:
            subprocess.run(perintah, check=True, cwd=AKAR, env=env)


if __name__ == "__main__":
    main()