## 📁 Struktur Direktori

```
project/
│
├── 📁 Dataset                          # Dataset mentah / original
│   └── dataset_1
│       ├── 0                           # Kelas 0 (normal)
│       └── 1                           # Kelas 1 (diabetes / abnormal)
│
├── 📁 Dataset_Preprocessing             # Dataset setelah preprocessing
│   └── dataset_1
│       ├── 0
│       └── 1
│
├── 📁 Dataset_Extract_A                 # Dataset hasil ekstraksi fitur untuk skenario A
│   └── dataset_1
│       ├── 0
│       └── 1
│
├── 📁 Dataset_Extract_B                 # Dataset hasil ekstraksi fitur untuk skenario B
│   └── dataset_1
│       ├── 0
│       └── 1
│
├── 📁 Dataset_Extract_D                 # Dataset hasil ekstraksi fitur untuk skenario D
│   └── dataset_1
│       ├── 0
│       └── 1
│
├── 📁 Excel                             # File Excel
│
├── 📁 Model_A                           # Model hasil training skenario A
│   └── dataset_1
│
├── 📁 Model_B                           # Model hasil training skenario B
│   └── dataset_1
│
├── 📁 Model_C                           # Model hasil training skenario C
│   └── dataset_1
│
├── 📁 Model_D                           # Model hasil training skenario D
│   └── dataset_1
│
├── 📁 Plot                              # Folder umum untuk grafik / visualisasi
│
├── 📁 Plot_Dataset                      # Visualisasi dataset awal
│   └── dataset_1
│       ├── 0
│       └── 1
│
├── 📁 Plot_Dataset_Extract_A            # Visualisasi dataset hasil ekstraksi fitur A
│   └── dataset_1
│       ├── 0
│       └── 1
│
├── 📁 Plot_Dataset_Extract_B            # Visualisasi dataset hasil ekstraksi fitur B
│   └── dataset_1
│       ├── 0
│       └── 1
│
├── 📁 Plot_Dataset_Extract_D            # Visualisasi dataset hasil ekstraksi fitur D
│   └── dataset_1
│       ├── 0
│       └── 1
│
├── 📁 Plot_Evaluasi                     # Plot evaluasi model
│   └── dataset_1
│
├── 📁 Tools                             # Script tambahan / helper tools / utilitas
│
├── 📄 dataset_preparation.ipynb        # Notebook untuk menyiapkan dataset
├── 📄 gui.py                           # Program GUI untuk menjalankan pipeline secara interaktif
├── 📄 requirements.txt                 # Daftar library Python yang dibutuhkan
├── 📄 settings.json                    # File konfigurasi
│
├── 📄 skenario A.ipynb                 # Notebook eksperimen skenario A
├── 📄 skenario B.ipynb                 # Notebook eksperimen skenario B
├── 📄 skenario C.ipynb                 # Notebook eksperimen skenario C
├── 📄 skenario D.ipynb                 # Notebook eksperimen skenario D
│
├── 📄 skenario_A_final.json            # Hasil akhir skenario A
├── 📄 skenario_B_final.json            # Hasil akhir skenario B
├── 📄 skenario_C_final.json            # Hasil akhir skenario C
└── 📄 skenario_D_final.json            # Hasil akhir skenario D
```

> ⚙️ Catatan:
> Sebagian besar folder akan **dibuat otomatis** saat menjalankan `dataset_preparation.ipynb`.
