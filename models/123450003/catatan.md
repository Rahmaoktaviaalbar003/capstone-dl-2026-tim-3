# Catatan Model 123450003

- **TIM:** TIM 3 PRISMA
- **Jalur / Tugas:** Transfer Learning (TL) / Deteksi Biner Kata Bangun 'Halo Meja'

## Alasan Optimizer
Menggunakan **Adam (LR 1e-3)** karena memiliki *adaptive learning rate* yang stabil untuk *fine-tuning* spektrogram Log-mel 64 pita serta menangani gradien *sparse* pada fitur audio.

## Regularisasi
1. **BatchNorm2d:** Menstabilkan aktivasi fitur konvolusi.
2. **Weight Decay (1e-4):** Mencegah pembobotan eksplosif.
3. **SpecAugment:** Frequency & Time Masking untuk mengatasi bising kipas latar.
4. **Early Stopping:** Menghentikan latihan jika loss validasi plateau 5 epoch.

## Pencarian Hyperparameter
| Eksperimen | LR | Batch | Optimizer | SpecAugment | Val Acc | Val F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| HP-01 | 1e-2 | 16 | SGD | Tidak | 0.650 | 0.582 |
| HP-02 | 1e-3 | 16 | Adam | Tidak | 0.825 | 0.810 |
| **HP-03 (Terbaik)** | **1e-3** | **32** | **Adam** | **Ya** | **0.915** | **0.902** |
| HP-04 | 1e-4 | 32 | Adam | Ya | 0.870 | 0.855 |

## Ablation
1. **Tanpa SpecAugment:** Val Macro F1 turun ~9.2% karena model overfitting pada bising kipas.
2. **Freezing Backbone:** Recall kata bangun 'Halo Meja' turun dari 90% ke 72%. Fine-tuning penuh terbukti lebih baik.

## Hasil 3 Seed Konfigurasi Akhir
| Seed | Val Acc | Val Macro F1 | Durasi | Status Berkas |
| :---: | :---: | :---: | :---: | :--- |
| `--seed 1` | 0.915 | 0.902 | 5.3s | Written to `results/123450003/metrics.json` |
| `--seed 2` | 0.900 | 0.891 | 5.1s | Written to `results/123450003/metrics.json` |
| `--seed 3` | 0.925 | 0.918 | 5.2s | Written to `results/123450003/metrics.json` |
| **Rata-rata** | **0.913** | **0.903** | **5.2s** | Evaluasi Berhasil |