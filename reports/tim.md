# Laporan tim — Tim 3

Satu dokumen bersama, enam bagian. Angka di bagian Hasil ditarik dari `results/*/metrics.json`
(tabel otomatis: `reports/hasil_komparasi.md` dari `python run_all.py`).

## Ringkasan
Laporan ini membandingkan kinerja jalur pembanding (*baseline*) non-*deep learning* dengan empat jalur arsitektur *deep learning* (*Transfer Learning*, CNN dari nol, *Autoencoder*, dan RNN/LSTM/GRU) pada dataset utama proyek menggunakan protokol pembagian data yang seragam. Temuan awal menetapkan batas bawah (*lower bound*) kinerja dari jalur *baseline* sebesar `val_acc` **0.3** dan `val_macro_f1` **0.230769** sebagai acuan perbaikan bagi seluruh jalur model tim.

## Protokol
* **Pembagian Data (*Split*):** Menggunakan *split* beku (`seed 2026`) melalui modul protokol tim (`src/common/split.py`) agar seluruh jalur model dievaluasi pada partisi data validasi dan uji yang identik.
* **Metrik Evaluasi:** Menggunakan `val_acc` dan `val_macro_f1` selama pengembangan serta `test_acc` untuk evaluasi akhir.
* **Anggaran & Reproduksibilitas:** Setiap jalur model dijalankan menggunakan konfigurasi seed yang ditetapkan dengan pencatatan durasi eksekusi dan perangkat keras pada `metrics.json`.
* **Evaluasi Data Uji:** Data uji (`pred_test.npy`) tidak pernah digunakan untuk penyetelan hiperparameter maupun penentuan ambang batas (*threshold*), dan hanya dievaluasi satu kali pada tahap akhir sprint oleh Penjaga Data & Evaluasi (Feby Wulandari).

## Hasil
Tabel komparasi sementara (batas bawah *baseline* telah tercatat dari `results/baseline/metrics.json`):

| Jalur / Model | Deskripsi | `val_acc` | `val_macro_f1` | Jumlah Parameter | Durasi (detik) | `test_acc` (sekali) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | Pembanding non-*deep learning* (`src/common/baseline.py`) | `0.3` | `0.230769` | - (Non-DL) | - | *Menunggu evaluasi akhir* |
| **Transfer Learning** (`123450003`) | *Transfer learning / fine-tuning* (`models/123450003/run.py`) | *Dalam proses* | *Dalam proses* | *Dalam proses* | *Dalam proses* | *Menunggu evaluasi akhir* |
| **CNN dari nol** | Arsitektur *Convolutional Neural Network* | *Dalam proses* | *Dalam proses* | *Dalam proses* | *Dalam proses* | *Menunggu evaluasi akhir* |
| **Autoencoder** | Representasi laten / *Autoencoder* | *Dalam proses* | *Dalam proses* | *Dalam proses* | *Dalam proses* | *Menunggu evaluasi akhir* |
| **RNN / LSTM / GRU** | Model sekuensial rekuren | *Dalam proses* | *Dalam proses* | *Dalam proses* | *Dalam proses* | *Menunggu evaluasi akhir* |

* **Analisis Kesalahan (Baseline):** Tahap pembanding awal (`val_acc` 0.3, `val_macro_f1` 0.230769), model klasik cenderung memprediksi kelas mayoritas sehingga kelas minoritas sering tertukar dan menurunkan skor *macro F1*.

## Analisis
* **Fakta Eksperimen:** Model *baseline* non-*deep learning* menghasilkan `val_acc` sebesar 0.3 dan `val_macro_f1` sebesar 0.230769 pada partisi validasi standar.
* **Interpretasi:** Selisih antara `val_acc` (0.3) dan `val_macro_f1` (0.230769) menunjukkan performa klasifikasi yang belum merata di seluruh kelas.
* **Dugaan:** Model linier/klasik sederhana belum mampu menangkap representasi fitur non-linier yang kompleks pada dataset, sehingga keempat jalur arsitektur *deep learning* tim diharapkan dapat meningkatkan pemisahan antar-kelas secara signifikan.

## Rekomendasi
Jalur *baseline* hanya direkomendasikan sebagai batas bawah acuan (*lower bound*) dan bukan model produksi karena skor `val_macro_f1` masih rendah (0.230769); pemilihan model akhir akan ditentukan setelah seluruh jalur *deep learning* menyelesaikan evaluasi multi-seed di dalam batas anggaran komputasi.

## Pembagian kerja
| Anggota | NIM | Jalur | Tugas sampingan | Bagian laporan yang ditulis |
| :--- | :--- | :--- | :--- | :--- |
| Rahma Oktavia Albar | 123450003 | Transfer learning / fine-tuning (+ Baseline) | Analis Utama | Ringkasan, Hasil (Baseline), Analisis |
| Feby Wulandari | *(isi NIM)* | CNN dari nol | Penjaga Data & Evaluasi | Protokol, Evaluasi Data Uji |
| Khansa Anyaveda Liyantri | *(isi NIM)* | Autoencoder (representasi) | Integrator | Hasil (Komparasi `run_all.py`), Analisis |
| Muhammad Dzikra | *(isi NIM)* | RNN -> LSTM/GRU | Reproduksi & Demo | Rekomendasi, Pembagian kerja |

Komponen yang tidak tersedia (bila ada): Hasil multi-seed lengkap dari keempat jalur model *deep learning* dan skor `test_acc` akhir belum diisi karena evaluasi data uji hanya dilakukan satu kali pada sprint terakhir.