## 📊 Perbandingan Kinerja Model

Tabel berikut merangkum hasil evaluasi dan analisis dari tiga model yang digunakan pada proyek ini.

| Nama Model | Akurasi | Hasil Analisis |
|----------|--------:|---------------|
| LSTM Base (Non-Pretrained) | 0.78 | Model baseline dengan embedding yang dilatih dari nol. Performa cukup baik pada kelas mayoritas (Neutral), namun masih kurang stabil pada kelas Positive karena keterbatasan representasi kata. |
| LSTM + GloVe (Transfer Learning) | 0.81 | Performa meningkat karena embedding pretrained membantu memetakan konteks kata lebih baik. Model lebih sensitif terhadap variasi kalimat, tetapi masih terdampak oleh ketidakseimbangan kelas. |
| LSTM + FastText (Transfer Learning) | 0.83 | Memberikan hasil terbaik. FastText secara efektif menangani kata turunan / variasi morfologi sehingga prediksi lebih konsisten pada teks pendek. Namun bias ke kelas Neutral tetap muncul pada beberapa sampel karena distribusi data yang tidak seimbang. |
