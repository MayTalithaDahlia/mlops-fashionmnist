# Praktikum MLOps — Model Experimentation & Tracking

Implementasi tugas Minggu 6 menggunakan Fashion-MNIST, PyTorch, MLflow, Optuna, dan DVC.

## Isi proyek
- `src/prepare_data.py`: mengunduh Fashion-MNIST dan membuat hash dataset.
- `src/experiment.py`: baseline CNN, logging per epoch, tuning minimal 20 trial, evaluasi 3 seed, ablation, registrasi model, dan prediksi lewat alias.
- `requirements.txt`: dependensi.
- `data/`: lokasi dataset yang dikelola DVC (dataset diunduh saat menjalankan persiapan).

## 1. Buat environment
Windows PowerShell:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Jika aktivasi PowerShell diblokir, jalankan sekali pada terminal pengguna:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 2. Siapkan dataset dan DVC
```powershell
python src/prepare_data.py
dvc init
dvc add data/FashionMNIST
git add data/FashionMNIST.dvc .gitignore .dvc
git commit -m "data: version Fashion-MNIST"
```
Folder data besar sebaiknya tidak dimasukkan langsung ke Git. File `.dvc` melacak versi data; untuk kolaborasi, konfigurasi DVC remote sesuai penyimpanan yang tersedia.

## 3. Jalankan MLflow Tracking Server
Buka **Terminal 1** di folder proyek:
```powershell
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000
```
Biarkan terminal ini tetap berjalan. Buka `http://127.0.0.1:5000` di browser.

Buka **Terminal 2**, aktifkan environment, lalu:
```powershell
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
```

## 4. Jalankan eksperimen
Baseline CNN:
```powershell
python src/experiment.py --mode baseline
```
Tuning 20 trial Optuna (default 2 epoch per trial agar praktis; bisa dinaikkan):
```powershell
python src/experiment.py --mode tune --trials 20 --epochs 2
```
Evaluasi konfigurasi terbaik dengan tiga seed:
```powershell
python src/experiment.py --mode seeds --epochs 5
```
Ablation dua komponen:
```powershell
python src/experiment.py --mode ablation --epochs 5
```
Registrasi model terbaik dan prediksi lewat alias `champion`:
```powershell
python src/experiment.py --mode register
```

> Jalankan mode secara berurutan. Mode `seeds`, `ablation`, dan `register` membaca hasil tuning dari file `best_config.json`. Jangan mengklaim hasil sebelum skrip benar-benar selesai dijalankan.

## 5. Bukti dan laporan
- Screenshot MLflow UI: daftar run, parameter, metrik, dan compare view.
- Simpan output terminal.
- Catat mean dan standar deviasi akurasi dari tiga seed.
- Laporan singkat: hipotesis, konfigurasi, hasil aktual, analisis ablation, keterbatasan, dan kesimpulan.
- Commit kode dan file DVC ke Git.

## Catatan metodologis
- Data validasi digunakan untuk memilih konfigurasi; test set digunakan hanya pada evaluasi akhir.
- Dataset hash dicatat sebagai tag setiap run.
- Tuning diatur singkat untuk demonstrasi. Untuk eksperimen final, naikkan epoch dan laporkan biaya komputasi.
- Alias `champion` hanya dipromosikan setelah model tersedia dan evaluasi dijalankan.
