# BAB I 
## PENDAHULUAN

### 1.1 Latar Belakang
Tumor adalah suatu kondisi yang ditandai dengan pertumbuhan sel abnormal yang membentuk massa atau neoplasma, yang sering kali menyerupai pembengkakan (Resnet & Saputra, 1907). Tumor dapat berkembang di berbagai organ tubuh manusia, termasuk otak (Candra et al., 2024). Berdasarkan data epidemiologi dari tinjauan sistematis, insidensi tumor otak di seluruh dunia tercatat sebesar 10,82 per 100.000 penduduk per tahun, dengan rentang antara 0,01 hingga 25,95 per 100.000 penduduk per tahun (Pratama et al., 2024). 

Tumor otak dapat dibedakan menjadi dua jenis, yaitu tumor primer yang berkembang langsung di otak dan tumor sekunder yang merupakan hasil metastasis dari organ lain (Otak et al., n.d.). Glioma merupakan jenis tumor otak primer yang paling sering ditemukan, di mana sekitar 78% dari total kasus tumor otak ganas termasuk dalam kategori ini (Septipalan et al., 2024). Selain itu, data dari Central Brain Tumor Registry of the United States (CBTRUS) menunjukkan bahwa meningioma adalah tumor otak yang paling sering terdiagnosis secara histologis dengan angka 36,8%, diikuti oleh tumor pituitari sebesar 16,2% (Candra et al., 2024).  

Untuk mendeteksi keberadaan tumor otak secara akurat, pasien umumnya disarankan menjalani pemeriksaan pencitraan medis seperti **CT Scan** atau **MRI** (Pratama et al., 2024). Dari hasil pencitraan medis tersebut, tumor dapat diklasifikasikan berdasarkan lokasi dan jenisnya. Namun, klasifikasi secara manual oleh tenaga medis sering kali membutuhkan waktu yang lama dan memiliki potensi kesalahan. Oleh karena itu, diperlukan suatu metode berbasis kecerdasan buatan yang dapat membantu mengklasifikasikan tumor otak dengan lebih efisien dan akurat. Salah satu metode yang saat ini banyak digunakan dalam analisis pencitraan medis adalah **Convolutional Neural Network (CNN)** (Otak et al., n.d.).  

CNN adalah teknik dalam *deep learning* yang sangat efektif dalam mengenali pola pada citra, termasuk pencitraan medis. Dengan menggunakan CNN, proses klasifikasi tumor otak dapat dilakukan secara otomatis berdasarkan karakteristik visual dari citra MRI. Salah satu model arsitektur CNN yang telah terbukti efektif dalam tugas klasifikasi citra adalah **VGG-16**. Model ini dikembangkan oleh **K. Simonyan dan A. Zisserman** dari Universitas Oxford dan berhasil mencapai kinerja yang sangat baik dalam pengenalan gambar pada dataset skala besar (Resnet & Saputra, 1907).  

Dalam penelitian ini, metode CNN dengan model **VGG-16** diterapkan untuk mengklasifikasikan jenis tumor otak berdasarkan citra MRI. Tumor otak akan dikategorikan ke dalam empat kelas, yaitu **Glioma Tumor, Meningioma Tumor, No Tumor, dan Pituitary Tumor**. Tujuan dari penelitian ini adalah untuk mengembangkan sistem klasifikasi berbasis *deep learning* yang dapat membantu tenaga medis dalam mendiagnosis tumor otak dengan lebih cepat dan akurat. Dengan adanya sistem ini, diharapkan dapat meningkatkan efisiensi diagnosis serta membantu dalam upaya deteksi dini tumor otak, sehingga penanganan dapat dilakukan lebih tepat dan efektif (Septipalan et al., 2024).

### 1.2 Teori Terkait

#### a. Tumor Otak
Tumor adalah hasil pertumbuhan tidak normal dari sel-sel yang merupakan komponen dasar dalam pembentukan jaringan dan organ dalam tubuh. Dalam kasus tumor otak, sel-sel yang tidak biasa berkembang dan membentuk benjolan di sekitar otak, yang bisa mengganggu fungsi normal dari otak itu sendiri.

Tumor otak dibagi menjadi dua yaitu, **tumor otak primer** dan **sekunder**. Tumor otak primer merupakan perubahan sel yang tidak normal dan tidak terkontrol yang berasal dari sel otak itu sendiri. Sedangkan, tumor otak sekunder merupakan tumor yang menyebar ke otak dari kanker tubuh bagian lain (Septipalan et al., 2024).

#### b. Convolutional Neural Network (CNN)
**Convolutional Neural Network (CNN)** merupakan salah satu jenis jaringan saraf tiruan yang banyak digunakan dalam pemrosesan gambar. CNN berfungsi untuk mengenali serta mengidentifikasi objek dalam suatu gambar. Meskipun memiliki prinsip kerja yang serupa dengan jaringan saraf tiruan pada umumnya, CNN terdiri dari neuron yang memiliki bobot, bias, dan fungsi aktivasi. Secara struktural, CNN tersusun atas beberapa lapisan utama, yaitu **lapisan konvolusi, pooling, dan fully connected layer**. Arsitektur umum dari CNN mencakup ketiga lapisan tersebut yang bekerja secara berurutan untuk mengekstrak fitur dan melakukan klasifikasi pada citra (Septipalan et al., 2024).

#### c. Model VGG-16
**VGG-16** adalah model CNN yang dibuat oleh **Visual Geometry Group (VGG)**. Model ini dikemukakan oleh **K. Simonyan dan A. Zisserman** dari Universitas Oxford. Tujuan utama VGG adalah merancang model dengan mempertimbangkan pengaturan kedalaman lapisan yang sesuai tanpa meningkatkan kompleksitas jaringan.

VGG-16 merupakan salah satu model *pre-trained* yang dapat digunakan untuk implementasi *deep learning* dalam bidang citra. Arsitektur VGG-16 telah terbukti memiliki kemampuan ekstraksi fitur yang kuat, terutama setelah pelatihan pada dataset yang besar (Candra et al., 2024).

### 1.3 Tujuan Tugas
Penelitian ini bertujuan untuk mengembangkan model berbasis **Convolutional Neural Network (CNN)** dengan arsitektur **VGG-16** yang mampu mengklasifikasikan citra MRI menjadi dua kategori utama, yaitu **tumor** dan **bukan tumor**. Dengan model ini, diharapkan sistem dapat membantu mendeteksi keberadaan tumor otak secara lebih akurat dan efisien.

Selain itu, penelitian ini juga berfokus pada peningkatan kinerja model dengan menerapkan teknik **augmentasi data**, sehingga model dapat lebih baik dalam mengenali pola pada citra MRI. Untuk memastikan hasil klasifikasi yang optimal, evaluasi akan dilakukan menggunakan metrik.

Melalui penelitian ini, diharapkan dapat tercipta sistem pendukung diagnosis berbasis *deep learning* yang dapat digunakan oleh tenaga medis untuk membantu mendeteksi tumor otak lebih cepat dan mengurangi risiko kesalahan diagnosis.

---

# BAB II 
## METODE

### 2.1 Langkah-langkah

#### a. Persiapan Data
Langkah pertama dalam pembuatan model ini adalah menyiapkan data yang akan digunakan untuk melatih dan menguji model. Dataset yang digunakan terdiri dari gambar **MRI otak** yang dibagi menjadi dua kategori, yaitu **Tumor** dan **Normal**. Data gambar ini kemudian diubah menjadi format yang dapat diproses oleh model menggunakan **ImageDataGenerator**.

#### b. Preprocessing dan Augmentasi Data
Sebelum digunakan untuk pelatihan, gambar-gambar diubah ukurannya menjadi **224x224 piksel** karena ukuran tersebut cocok dengan arsitektur model **VGG16** yang akan digunakan. Selain itu, gambar juga diproses agar nilai pikselnya berada dalam rentang **[0, 1]** dengan melakukan normalisasi. **Augmentasi gambar** juga dilakukan untuk menambah variasi data dan mencegah *overfitting*, misalnya dengan rotasi gambar, pergeseran, dan pembalikan horizontal.

#### c. Arsitektur Model
Dalam model ini, digunakan arsitektur **VGG16** yang sudah dilatih sebelumnya pada dataset **ImageNet**. Kami memanfaatkan **transfer learning** dengan menggunakan model VGG16 tanpa lapisan klasifikasinya, karena lapisan tersebut tidak dibutuhkan untuk dataset kita. Kemudian, lapisan klasifikasi baru ditambahkan di atasnya, terdiri dari lapisan **dense** dengan fungsi aktivasi **ReLU** dan **sigmoid** untuk output biner (Tumor atau Normal).

#### d. Pelatihan Model
Setelah arsitektur model siap, langkah berikutnya adalah pelatihan. Proses pelatihan dilakukan selama beberapa epoch, di mana model belajar untuk memprediksi kategori gambar berdasarkan data pelatihan. Kami menggunakan optimizer **Adam** dengan **learning rate** yang sangat kecil agar proses pelatihan berjalan lebih stabil. Setiap epoch, hasil model dievaluasi menggunakan data validasi untuk memeriksa seberapa baik model bekerja pada data yang tidak terlihat sebelumnya.

#### e. Evaluasi Model
Setelah pelatihan selesai, evaluasi dilakukan pada model untuk mengukur seberapa akurat model dalam mengklasifikasikan gambar. Pengukuran dilakukan menggunakan **akurasi**, yaitu persentase gambar yang diklasifikasikan dengan benar. Selain itu, **confusion matrix** juga digunakan untuk menunjukkan bagaimana model mengklasifikasikan gambar pada setiap kategori.

### 2.2 Visualisasi Model

#### a. Plot Akurasi dan Loss
Selama pelatihan, dua grafik utama yang digunakan untuk memantau kinerja model adalah grafik **akurasi** dan grafik **loss**. Grafik akurasi menunjukkan seberapa baik model dalam memprediksi dengan benar pada data pelatihan dan validasi, sedangkan grafik loss menggambarkan seberapa besar kesalahan model dalam memprediksi. Visualisasi ini membantu kita melihat apakah model sudah belajar dengan baik atau perlu penyesuaian lebih lanjut.

#### b. Confusion Matrix
Untuk memeriksa performa model lebih detail, digunakan **confusion matrix**, yang menunjukkan distribusi prediksi benar dan salah oleh model. Matriks ini membantu mengidentifikasi jenis kesalahan yang dilakukan oleh model, apakah lebih sering mengklasifikasikan gambar tumor sebagai normal, atau sebaliknya. Visualisasi **confusion matrix** menggunakan **heatmap** sangat berguna untuk mempermudah analisis.

#### c. Visualisasi Citra
Selain itu, beberapa gambar dari **dataset validasi** ditampilkan untuk menunjukkan bagaimana model mengenali gambar **tumor** dan **normal**. Visualisasi gambar ini berguna untuk memverifikasi apakah model benar-benar memahami perbedaan antara gambar tumor dan normal, serta memberi gambaran lebih jelas tentang data yang digunakan.

---
# BAB III
## HASIL

### 3.1 Distribusi Data MRI dalam Dataset
Sebelum melatih model, penting untuk memahami distribusi jumlah gambar dalam dataset. Grafik berikut menunjukkan jumlah citra pada masing-masing kelas (**Tumor** dan **Normal**).

### 3.2 Visualisasi Contoh Citra MRI
Setelah memuat dataset menggunakan **ImageDataGenerator**, dilakukan visualisasi beberapa contoh citra **MRI otak** dari masing-masing kelas.

### 3.3 Proses Training Model CNN (VGG16)
Model **VGG16** digunakan sebagai **feature extractor**, dengan menambahkan beberapa lapisan **fully connected** di bagian akhir untuk klasifikasi biner (**Tumor vs. Normal**). Pelatihan dilakukan selama **10 epoch**, dengan dataset yang telah melalui augmentasi.

### 3.4 Evaluasi Model CNN (VGG16)

#### a. Confusion Matrix: Evaluasi Kinerja Model
**Confusion matrix** digunakan untuk melihat bagaimana model melakukan prediksi terhadap data validasi.

#### b. Evaluasi dengan Classification Report

---

# BAB IV
## KESIMPULAN

### 4.1 Ringkasan Temuan
Dalam eksperimen ini, model **CNN** dengan arsitektur **VGG16** diterapkan untuk mengklasifikasikan gambar **MRI tumor otak** menjadi dua kategori: **Tumor** dan **Normal**. Berdasarkan hasil pelatihan dan evaluasi yang dilakukan, model VGG16 menunjukkan kinerja yang cukup baik dalam mendeteksi tumor otak pada dataset yang digunakan. **Akurasi validasi** mencapai nilai yang memadai, namun perlu diperhatikan bahwa hasil prediksi dan akurasi dapat dipengaruhi oleh berbagai faktor, seperti kualitas data dan teknik augmentasi gambar yang diterapkan.

Sebagai tambahan, penggunaan teknik augmentasi seperti **rotasi**, **pergeseran lebar dan tinggi**, serta **pembalikan horizontal** terbukti efektif dalam memperkaya data pelatihan, yang pada gilirannya membantu meningkatkan kemampuan model untuk mengenali variasi gambar. **Confusion Matrix** yang dihasilkan dari evaluasi model menunjukkan bahwa model dapat membedakan dengan baik antara gambar tumor dan gambar normal, meskipun masih ada beberapa kesalahan klasifikasi yang perlu diperbaiki di masa mendatang.

### 4.2 Batasan Pekerjaan

#### a. Keterbatasan Data
Dataset yang digunakan terbatas pada jumlah dan jenis gambar, yang mungkin memengaruhi akurasi model. Model CNN umumnya membutuhkan data yang cukup banyak untuk mencapai performa yang optimal.

#### b. Augmentasi Gambar
Meskipun augmentasi gambar membantu memperbaiki akurasi model, masih ada potensi teknik augmentasi lainnya yang dapat diuji untuk lebih meningkatkan kinerja model.

#### c. Durasi Pelatihan
Model VGG16 membutuhkan waktu pelatihan yang relatif panjang, sehingga akurasi yang lebih tinggi dapat dicapai dengan jumlah epoch yang lebih banyak atau penggunaan teknik lain untuk mempercepat proses pelatihan.

### 4.3 Rekomendasi untuk Pekerjaan di Masa Depan
Untuk meningkatkan kualitas dan akurasi model klasifikasi tumor otak di masa depan, beberapa langkah berikut dapat dipertimbangkan:

#### a. Perluasan Dataset
Mengumpulkan lebih banyak data gambar **MRI** dari berbagai sumber dapat membantu meningkatkan kemampuan model untuk melakukan generalisasi yang lebih baik.

#### b. Penerapan Teknik Augmentasi Lainnya
Eksplorasi teknik augmentasi gambar lainnya, seperti **perputaran sudut yang lebih ekstrim**, **perubahan kontras**, dan **teknik pencahayaan**, dapat membantu memperkaya dataset dan meningkatkan kinerja model.

#### c. Penggunaan Transfer Learning dengan Arsitektur yang Lebih Canggih
Model dengan arsitektur yang lebih kompleks, seperti **ResNet** atau **DenseNet**, serta eksplorasi **transfer learning** dengan pretrained models yang lebih kuat, dapat meningkatkan performa model.

#### d. Penggunaan Teknologi Cloud dan GPU
Untuk mempercepat pelatihan model, penggunaan **GPU** atau layanan **cloud computing** dapat dioptimalkan, terutama untuk model dengan arsitektur besar seperti **VGG16** yang memerlukan banyak daya komputasi.
