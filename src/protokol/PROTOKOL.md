# Protokol evaluasi tim — <Nama tim>

Diisi Penjaga Data & Evaluasi pada Sprint 0, digabung ke `main` lewat *pull request* yang
disetujui pembimbing, lalu **dibekukan**. Perubahan setelah itu hanya lewat *pull request*
baru yang disetujui pembimbing.

| Butir | Isi |
|---|---|
| Kasus (dari kartu Kasus tim) | Deteksi biner kata bangun (*wake-word detection*) "Halo Meja" menggunakan Log-Mel Spectrogram 64 pita berdurasi 1 detik untuk pengembangan pemicu suara pada perangkat meja. |
| Sumber dataset | Dataset primer direkam sendiri oleh tim pada meja uji menggunakan mikrofon 16 kHz. Referensi starter repository: https://github.com/sains-data/capstone-dl-2026-starter. Data bersama berupa rekaman program harian. Data dari tim lain hanya boleh digunakan sebagai data uji tambahan. |
| Jenis tugas (`TUGAS`) | Klasifikasi biner. |
| Target / label | Positif: ucapan target "HALO MEJA". Negatif: ucapan non-target, meliputi "halo teman", "meja makan", kondisi diam, dan rekaman program harian. |
| Ukuran data latih / validasi / uji | Belum ditetapkan; diisi berdasarkan jumlah data aktual setelah pengumpulan dan pembagian data. |
| Cara split | Pemisahan berdasarkan peserta, sesi, atau blok rekaman untuk mencegah kebocoran data. Seed 2026 dan porsi validasi 10% digunakan sesuai mekanisme `src.common.split`, dengan pengelompokan data yang harus dipastikan kompatibel dengan implementasi split. |
| Metrik utama untuk komparasi | False Alarms per Hour (FA/hour) pada rekaman program harian sebagai metrik utama dan Recall sebagai metrik pendamping. Hasil dilaporkan sebagai rata-rata ± simpangan baku dari 3 seed. Ukuran model ONNX (KB) dan latensi (ms) pada laptop meja juga dicatat. |
| Praproses yang sama untuk semua model | Audio direkam dengan sampling rate 16 kHz. Satu detik pertama setiap kondisi dibuang. Audio diproses menjadi Log-Mel Spectrogram 64 pita berdurasi 1 detik. Normalisasi dihitung dari data latih saja dan diterapkan secara konsisten pada data validasi dan data uji. |
| Anggaran komputasi per run | Maksimal 20 menit menggunakan CPU laptop atau sesuai persetujuan pembimbing. |
| Evaluasi data uji | Evaluasi dilakukan satu kali oleh Penjaga Data & Evaluasi pada sprint terakhir menggunakan opsi `--final`. Data dari tim lain hanya digunakan sebagai data uji tambahan. |
| `data_hash` saat dibekukan | Diisi berdasarkan nilai `data_hash` dari `results/baseline/metrics.json` setelah baseline berhasil dijalankan. |


Disetujui pembimbing pada: … (tautan *pull request*)
