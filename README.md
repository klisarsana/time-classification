# Klasifikasi Waktu pada Gambar Jam Analog (CNN)

Model CNN (Keras `Sequential`) untuk mengklasifikasikan waktu yang ditunjukkan gambar jam analog ke dalam **144 kelas** (jam 1-12 × kelipatan 5 menit, contoh label `1-20`, `2-55`).

## Dataset

- Sumber: [gpiosenka/time-image-datasetclassification](https://www.kaggle.com/datasets/gpiosenka/time-image-datasetclassification) (Kaggle), diunduh lewat `kagglehub`.
- Total data: 14.400 gambar, 144 kelas.
- **Pembagian data dilakukan mandiri**, tidak memakai split bawaan dataset. Seluruh gambar dari folder `train`, `valid`, dan `test` bawaan digabung, lalu dibagi ulang dengan `train_test_split` (`stratify` berdasarkan label, `random_state=42`) agar proporsi 144 kelas tetap seimbang di setiap split.

| Split | Proporsi | Jumlah gambar |
|-------|----------|---------------|
| Train | 80% | 11.520 (69.120 setelah augmentasi) |
| Validation | 10% | 1.440 |
| Test | 10% | 1.440 |

Augmentasi hanya diterapkan pada data train, dan dilakukan setelah split, sehingga validation dan test tidak mengandung gambar hasil augmentasi.

## Alur Pengerjaan

1. **Load & gabung data**: path dan label dibaca dari struktur folder dataset bawaan.
2. **Split mandiri**: 80/10/10 dengan stratifikasi.
3. **Visualisasi**: contoh 6 gambar acak beserta labelnya.
4. **Label encoding**: `LabelEncoder` (scikit-learn) pada 144 kelas.
5. **Augmentasi**: setiap gambar train dirotasi 30°, 60°, 90°, 120°, 180° dan disimpan sebagai grayscale (11.520 → 69.120).
6. **Pipeline `tf.data`**: baca JPEG 1 channel, resize `128×128`, normalisasi, `cache` → `shuffle` → `batch(32)` → `prefetch`.
7. **Training**: maksimal 30 epoch dengan callback.
8. **Evaluasi**: akurasi dan loss pada data train dan test, plot kurva training.
9. **Ekspor model**: SavedModel, TF-Lite, TFJS.
10. **Inference**: prediksi memakai SavedModel pada test set + visualisasi hasil.

## Arsitektur Model

Input `128×128×1` (grayscale)

| Blok | Layer |
|------|-------|
| 1-4 | `Conv2D` (32 → 64 → 128 → 256 filter, 3×3, ReLU, `same`) → `BatchNormalization` → `MaxPooling2D` |
| Head | `GlobalAveragePooling2D` → `Dense(512, ReLU)` → `Dropout(0.4)` → `Dense(144, softmax)` |

- Optimizer: Adam
- Loss: `sparse_categorical_crossentropy`
- Callback:
  - `EarlyStopping` (monitor `val_loss`, patience 5, `restore_best_weights`)
  - `ModelCheckpoint` (`best_model.keras`, monitor `val_accuracy`)
  - `ReduceLROnPlateau` (factor 0.5, patience 3, `min_lr` 1e-6)

## Hasil

Training berjalan penuh selama 30 epoch.

| Metrik | Nilai |
|--------|-------|
| Train accuracy (evaluasi akhir) | 99,89% (loss 0,0035) |
| Validation accuracy (epoch 30) | 99,03% |
| **Test accuracy** | **98,75%** (loss 0,0736) |
| Akurasi SavedModel (inference) | 98,75% |

Selisih train dan test sekitar 1,1 poin persentase, sehingga model tidak mengalami overfitting yang berarti. Akurasi inference dari SavedModel sama dengan evaluasi model Keras, jadi proses ekspor tidak mengubah performa.

## Output Model

| Format | File / Folder |
|--------|---------------|
| SavedModel | `saved_model/` |
| TF-Lite | `tflite/model.tflite` |
| TensorFlow.js | `tfjs_model/` |
| Daftar label | `tflite/label.txt` (urutan sesuai indeks output model) |
| Checkpoint terbaik | `best_model.keras` |

## Cara Menjalankan

Notebook disusun untuk Google Colab / Kaggle.

```bash
pip install -r requirements.txt
```

1. Siapkan akses Kaggle (untuk `kagglehub`) bila diminta.
2. Jalankan seluruh sel `notebook.ipynb` secara berurutan.
3. Sel yang memakai `google.colab.files` hanya berfungsi di Colab; lewati bila dijalankan di tempat lain.

## Inference Singkat

```python
import tensorflow as tf

loaded = tf.saved_model.load("saved_model")
infer = loaded.signatures["serving_default"]
input_name = list(infer.structured_input_signature[1].keys())[0]

# images: tensor float32 shape (N, 128, 128, 1), dinormalisasi seperti saat training
probs = list(infer(**{input_name: tf.constant(images)}).values())[0].numpy()
labels = open("tflite/label.txt").read().splitlines()
pred = labels[probs.argmax(axis=1)[0]]
```

## Teknologi

Python, TensorFlow/Keras, TensorFlow Lite, TensorFlow.js, scikit-learn, OpenCV, Pillow, NumPy, Matplotlib.
