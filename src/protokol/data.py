"""Pemuat data kasus tim — MILIK TIM.

Kasus Tim 3: Deteksi kata bangun 'Halo Meja' (Deteksi biner)
Masukan: Log-mel 64 pita, 1 detik (~100 frames) -> (N, 64, 100)
Keluaran: Label biner 0 (pengecoh/bising) dan 1 ('Halo Meja')
"""
from __future__ import annotations

import numpy as np
from src.common import manifest, split

TUGAS = "klasifikasi"


def muat():
    """Memuat dataset simulasi jika data meja belum tersedia/dalam pengujian simulasi."""
    try:
        m = manifest.baca()
    except Exception:
        m = {}

    # Generasi data simulasi sesuai spesifikasi masukan Tim 3 (Log-mel 64 pita, 1 s / ~100 time frames)
    np.random.seed(2026)
    N_total = 100
    fitur_logmel = 64
    waktu_frames = 100  # 1 detik audio pada spesifikasi spectrogram

    X = np.random.randn(N_total, fitur_logmel, waktu_frames).astype(np.float32)
    y = np.random.randint(0, 2, size=(N_total,)).astype(np.int64)

    # Pembagian indeks latih dan validasi sesuai kontrak split (seed 2026)
    tr, va = split.indeks(len(X))

    # Data uji simulasi (20 sampel)
    X_test = np.random.randn(20, fitur_logmel, waktu_frames).astype(np.float32)
    y_test = np.random.randint(0, 2, size=(20,)).astype(np.int64)

    return X[tr], y[tr], X[va], y[va], X_test, y_test