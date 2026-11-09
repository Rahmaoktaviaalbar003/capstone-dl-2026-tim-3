"""Pemuat data kasus tim 3 (Kata bangun 'Halo Meja')."""
from __future__ import annotations

import numpy as np
from src.common import split

TUGAS = "klasifikasi"


def muat():
    """Generasi data simulasi Log-mel 64 pita, 1 detik (100 frame)."""
    np.random.seed(2026)
    N_total = 100
    fitur_logmel = 64
    waktu_frames = 100

    # (N, 64, 100)
    X = np.random.randn(N_total, fitur_logmel, waktu_frames).astype(np.float32)
    y = np.random.randint(0, 2, size=(N_total,)).astype(np.int64)

    tr, va = split.indeks(len(X))

    X_test = np.random.randn(20, fitur_logmel, waktu_frames).astype(np.float32)
    y_test = np.random.randint(0, 2, size=(20,)).astype(np.int64)

    return X[tr], y[tr], X[va], y[va], X_test, y_test