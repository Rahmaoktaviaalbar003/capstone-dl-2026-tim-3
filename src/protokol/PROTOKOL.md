# Protokol evaluasi tim — Tim 3

Diisi Penjaga Data & Evaluasi pada Sprint 0, digabung ke `main` lewat *pull request* yang
disetujui pembimbing, lalu **dibekukan**. Perubahan setelah itu hanya lewat *pull request*
baru yang disetujui pembimbing.

| Butir | Isi |
|---|---|
| Kasus (dari kartu Kasus tim) | Deteksi kata bangun “Halo Meja” pada audio 1 detik |
| Sumber dataset | Data primer hasil rekaman eksperimen Tim 3 pada lingkungan meja/table |
| Jenis tugas (`TUGAS`) | klasifikasi |
| Target / label | `1` = HALO MEJA, `0` = bukan HALO MEJA |
| Ukuran data latih / validasi / uji | Ditentukan berdasarkan hasil rekaman dan pembagian grup peserta |
| Cara split | Split berdasarkan peserta; seluruh data dari satu peserta tetap berada pada satu fold |
| Metrik utama untuk komparasi | false alarms per hour dan recall |
| Praproses yang sama untuk semua model | Audio 16 kHz; buang 1 detik pertama setiap kondisi; ambil window 1 detik di sekitar puncak energi suara; kondisi diam diambil pada bagian tengah |
| Anggaran komputasi per run | ≤ 20 menit CPU laptop (atau yang disetujui pembimbing) |
| Evaluasi data uji | Sekali, oleh Penjaga Data, pada sprint terakhir (`--final`) |
| `data_hash` saat dibekukan | … (diisi dari `results/baseline/metrics.json`) |

Disetujui pembimbing pada: … (tautan *pull request*)