# Praktikum MLOps — Model Experimentation & Tracking

Implementasi praktikum MLOps Minggu 6 yang mencakup eksperimen dan pelacakan model klasifikasi gambar Fashion-MNIST menggunakan PyTorch, MLflow, Optuna, dan DVC.

Proyek ini mencakup pelatihan CNN baseline, *hyperparameter tuning*, evaluasi menggunakan beberapa *random seed*, *ablation study*, registrasi model, serta evaluasi model melalui alias `champion` pada MLflow Model Registry.

## 1. Teknologi yang Digunakan

- **Python dan PyTorch** — implementasi serta pelatihan model CNN.
- **MLflow** — pencatatan parameter, metrik, tag, dan artefak eksperimen.
- **Optuna** — pencarian konfigurasi *hyperparameter* terbaik.
- **DVC (Data Version Control)** — pencatatan versi dataset.
- **Git dan GitHub** — pengelolaan versi kode dan dokumentasi proyek.

## 2. Struktur Proyek

```text
mlops_fashionmnist_starter/
├── .dvc/
├── data/
│   ├── FashionMNIST/
│   ├── dataset_metadata.json
│   └── FashionMNIST.dvc
├── report/
│   └── 123450009_May Talitha Dahlia_RA_Tugas Pertemuan 6 MLOPS.pdf
├── src/
│   ├── experiment.py
│   └── prepare_data.py
├── best_config.json
├── requirements.txt
└── README.md
```

Folder `.venv`, `mlartifacts`, dan database MLflow lokal digunakan untuk kebutuhan lingkungan dan eksperimen lokal sehingga tidak perlu dimasukkan ke GitHub.

## 3. Dataset

Dataset yang digunakan adalah Fashion-MNIST, yang terdiri atas citra grayscale berukuran 28 × 28 piksel dengan 10 kelas kategori pakaian.

| Informasi | Nilai |
|---|---:|
| Data latih | 60.000 |
| Data uji | 10.000 |
| Jumlah kelas | 10 |
| Ukuran citra | 28 × 28 piksel |
| Hash SHA-256 metadata dataset | `5494a24b2ad208b03ce26083281279486b6c2ab468ee1ec5be9e50c4cdd8ea61` |
| Hash DVC direktori dataset | `b99a460e16e5dea108b244b4242e339db.dir` |

Pembagian data latih dan validasi dilakukan oleh program eksperimen. Data uji digunakan untuk evaluasi akhir model.

## 4. Persiapan Lingkungan

Jalankan perintah berikut melalui PowerShell dari direktori utama proyek.

Buat dan aktifkan *virtual environment*:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Jika Python 3.11 tidak tersedia, gunakan versi Python yang kompatibel dengan dependensi proyek.

## 5. Persiapan Dataset dan DVC

Jalankan program persiapan dataset:

```powershell
python src/prepare_data.py
```

Dataset dicatat menggunakan DVC melalui perintah:

```powershell
dvc add data/FashionMNIST
dvc status
```

File `data/FashionMNIST.dvc` menyimpan informasi identitas dan versi direktori dataset. Metadata SHA-256 dicatat dalam `data/dataset_metadata.json`.

Untuk menyimpan perubahan metadata dataset ke Git:

```powershell
git add data/FashionMNIST.dvc data/dataset_metadata.json .dvc .dvcignore
git status
git commit -m "Track Fashion-MNIST dataset with DVC"
git push origin master
```

Perintah commit hanya diperlukan jika terdapat perubahan yang belum di-commit.

**Catatan:** file DVC yang tersimpan di Git belum otomatis menyediakan dataset bagi pengguna lain. Agar dataset dapat diambil menggunakan `dvc pull`, proyek memerlukan DVC remote yang dapat diakses dan cache dataset yang telah diunggah ke remote tersebut.

## 6. Menjalankan MLflow Tracking Server

Buka terminal pertama pada direktori utama proyek dan jalankan:

```powershell
mlflow server --backend-store-uri "sqlite:///mlflow.db" --default-artifact-root "./mlartifacts" --host 127.0.0.1 --port 5000
```

Biarkan terminal tersebut tetap berjalan selama eksperimen.

Buka antarmuka MLflow melalui:

[http://127.0.0.1:5000](http://127.0.0.1:5000)

Pada terminal kedua, aktifkan *virtual environment* dan atur tracking URI:

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
```

## 7. Menjalankan Eksperimen

Seluruh mode eksperimen dijalankan melalui `src/experiment.py`.

### 7.1 Baseline CNN

```powershell
python src/experiment.py --mode baseline
```

Mode ini digunakan untuk memperoleh performa awal model CNN.

### 7.2 Hyperparameter Tuning dengan Optuna

```powershell
python src/experiment.py --mode tune --trials 20 --epochs 2
```

Tuning dijalankan dengan minimal 20 *trial*. Setiap *trial* menggunakan dua *epoch* pada konfigurasi eksperimen yang tercatat.

### 7.3 Evaluasi Tiga Random Seed

```powershell
python src/experiment.py --mode seeds --epochs 5
```

Mode ini mengevaluasi konfigurasi pada *seed* 42, 123, dan 2026, kemudian merangkum rata-rata dan standar deviasi sampel akurasi validasi.

### 7.4 Ablation Study

```powershell
python src/experiment.py --mode ablation --epochs 5
```

*Ablation study* membandingkan model lengkap, model tanpa augmentasi, dan model tanpa *dropout*.

### 7.5 Registrasi Model

```powershell
python src/experiment.py --mode register
```

Mode ini mendaftarkan model yang dipilih ke MLflow Model Registry dengan nama `fashion-mnist-cnn` dan menggunakan alias `champion`.

### 7.6 Evaluasi Model Champion

```powershell
python src/experiment.py --mode evaluate
```

Mode ini memuat model melalui alias `champion`, mengevaluasinya pada data uji, dan mencatat metrik serta artefak `evaluation_results.json` di MLflow.

Jalankan mode secara berurutan sesuai kebutuhan. Mode tuning perlu diselesaikan sebelum evaluasi seed, *ablation study*, dan registrasi model karena tahapan tersebut menggunakan konfigurasi terbaik.

## 8. Ringkasan Hasil Eksperimen

Berikut ringkasan hasil eksperimen yang telah diperoleh selama praktikum.

| Tahapan | Hasil |
|---|---|
| Baseline CNN | Akurasi validasi 87,14% pada eksperimen baseline dua *epoch* |
| Optuna tuning | 20 *trial*; akurasi validasi terbaik 90,20% pada *trial* ke-12 |
| Evaluasi tiga seed | Rata-rata akurasi validasi 91,67%; standar deviasi sampel 0,0053 |
| Registrasi model | `fashion-mnist-cnn`, versi 1, alias `champion` |
| Evaluasi data uji | Akurasi 90,03%; *test loss* sekitar 0,2781 |

Konfigurasi terbaik dari tuning menggunakan *learning rate* sekitar 0,002676, *batch size* 64, *dropout* sekitar 0,1111, dan 32 *channels*.

Hasil *ablation study* yang tercatat pada eksperimen menunjukkan akurasi validasi sebesar 90,90% untuk model lengkap, 90,90% untuk model tanpa augmentasi, dan 91,20% untuk model tanpa *dropout*.

Hasil antarpercobaan perlu dibandingkan dengan memperhatikan konfigurasi, jumlah *epoch*, *random seed*, dan prosedur evaluasi yang digunakan.

## 9. Pelacakan Eksperimen

MLflow digunakan untuk mencatat hasil eksperimen, termasuk parameter, metrik, dan tag dataset. Informasi yang dicatat meliputi:

- Identitas dataset Fashion-MNIST.
- Hash SHA-256 dari metadata dataset.
- Hash MD5 direktori dataset yang tercatat pada file DVC.
- Parameter dan metrik eksperimen.
- Informasi versi model yang terdaftar.
- Artefak evaluasi model *champion*.

MLflow Model Registry digunakan untuk menyimpan model terpilih dan mengelolanya melalui alias `champion`.

## 10. Dokumentasi dan Laporan

Laporan praktikum tersedia pada folder `report/` dalam repositori ini.

Dokumentasi eksperimen dapat dilengkapi dengan *screenshot* MLflow yang memperlihatkan hasil tuning, evaluasi tiga seed, *ablation study*, Model Registry, serta pencatatan identitas dataset.

## 11. Catatan Metodologis

- Konfigurasi model dipilih berdasarkan data validasi, bukan data uji.
- Data uji digunakan untuk evaluasi akhir model.
- Hasil tiga seed dirangkum menggunakan rata-rata dan standar deviasi sampel.
- Jumlah *epoch* yang relatif singkat digunakan untuk efisiensi eksperimen.
- Hash dataset membantu identifikasi data, sedangkan DVC dan penyimpanan data yang sesuai diperlukan untuk pengelolaan versi serta reproduksi data.
- Hasil eksperimen yang dicatat di MLflow menjadi dasar dokumentasi dan analisis performa model.

## 12. Repositori

Kode program, konfigurasi eksperimen, metadata dataset, dan laporan praktikum dikelola dalam repositori GitHub:

[https://github.com/MayTalithaDahlia/mlops-fashionmnist](https://github.com/MayTalithaDahlia/mlops-fashionmnist)