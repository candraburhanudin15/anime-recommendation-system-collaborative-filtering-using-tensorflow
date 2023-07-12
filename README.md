
  

# Laporan Proyek _Machine Learning_ Oleh Candra Burhanudin

  

>  ## Sistem rekomendasi anime menggunakan metode model development melalui pendeketan deep learning dengan teknik collaboratory filtering

  

# 🌐*Project Overview*


  

## **Latar Belakang**

  

Sistem Rekomendasi memiliki peran penting dalam meningkatkan pengalaman pengguna, meningkatkan retensi pengguna serta memperluas pengetahuan dan eksplorasi. Dalam era informasi yang kaya dengan pilihan yang tak terbatas, sistem rekomendasi menjadi inovasi dalam membantu pengguna menavigasi dan memanfaatkan konten yang tersedia dengan lebih baik. Salah satu penerapannya pada bidang hiburan, Sistem rekomendasi Tontonan Anime menjadi semakin penting dalam industri hiburan, terutama dengan pertumbuhan popularitas anime di seluruh dunia. Penggemar anime sering kali dihadapkan pada pilihan yang sangat luas ketika mencari anime baru untuk ditonton. Dalam konteks ini, pengembangan sebuah sistem rekomendasi Tontonan Anime menjadi krusial untuk membantu pengguna menemukan anime yang sesuai dengan preferensi mereka. Dengan adanya sistem rekomendasi, pengguna lebih mudah menemukan anime yang mungkin belum diketahui serta sesuai dengan preferensi mereka. selain itu, pengembangan sistem rekomendasi anime juga dapat memberikan manfaat bagi penyedia konten anime layaknya platform streaming konten anime untuk lebih dapat meningkatkan efisiensi dan produktivitas bisnis dengan menyajikan konten sesuai preferensi pengguna masing-masing. dengan dikembangkannya model sistem rekomendasi ini, diharapkan dapat memberikan kontribusi positif dalam industri anime

  

- **Penyelesaian**

berdasarkan latar belakang yang telah dipaparkan, sistem rekomendasi dalam halnya untuk meningkatkan pengalaman pengguna dalam menemukan anime yang sesuai dengan preferensi pengguna platform peneliti menggunakan dataset yang sudah ada untuk memulai pembuatan model sistem rekomendasi dengan klasifikasi *collaborative filtering* dengan deep learning menggunakan TensorFlow. dataset yang digunakan berjumlah 7813737. namun pada penelitian ini dataset dipotong menjadi jumlah 10000 record untuk proses training yang lebih cepat. Penelitian ini menggunakan teknik collaboration filtering dengan bergantung pada pendapat komunitas pengguna dan termasuk pada personalized recommender system. goal proyek ini adalah menghasilkan rekomendasi sejumlah judul anime yang sesuai dengan preferensi pengguna berdasarkan *rating* yang telah diberikan sebelumnya. record nilai rating memiliki rentang angka dari 0 hingga 10 .

  

# 💼 _Business Understanding_

  

untuk menggambarkan penerapan sistem rekomendasi khususnya pada platform streaming online yang memiliki fokus untuk dapat memberikan rekomendasi anime yang relevan kepada user, tentunya memiliki potensi meningkatnya keuntungan bisnis terlebih banyaknya orang yang mendapat konten yang lebih relevan dapat meningkatkan daya tarik pelanggan dan meningkatkan penjualan serta keuntungan sistem rekomendasi dapat mempengaruhi keputusan pembelian dan mendorong pelanggan untuk melakukan transaksi pembelian tambahan.

  

## _Problem Statement_

  

- **Masalah 1 : Bagaimana mengembangkan sistem rekomendasi anime dengan teknik *collaborative filtering* untuk membantu pengguna menemukan anime yang sesuai dengan preferensi pengguna ?**

- **Masalah 2 : Bagaimana mengimplementasikan sistem rekomendasi  yang efektif untuk dapat memberikan hasil rekomendasi anime ?**

- **Masalah 3 : Bagaimana meningkatkan akurasi sitem rekomendasi anime untuk memberikan rekomendasi yang lebih relevan? ?**

  

## _Goals_

  

- **Jawaban Pernyataan Masalah 1**

mengembangkan sistem rekomendasi anime dengan teknik *collaborative filtering* dapat melalui analisis pola interaksi pengguna dengan anime yang berasal dari dataset yang diteliti, sistem mengidentifikasi hubungan yang relevan antara pengguna dan data anime sebelumnya yang telah tercatat, sehingga dapat memberikan rekomendasi yang lebih personal pada pengguna tertentu


- **Jawaban Pernyataan Masalah 2**


untuk dapat mengimplementasikan sistem rekomendasi yang efektif dapat menggunakan beberapa cara baik dari kategori non-personalized atau personalized. pada penelitian ini menggunakan personalized recommender system dengan kategori *collaborative filtering*. pada kategori ini terdapat teknik memory based dan model based. penelitian ini menggunakan model based dengan pendekatan deep learning.

  

- **Jawaban Pernyataan Masalah 3**

Untuk meningkatkan akurasi sistem rekomendasi, selain faktor banyaknya jumlah data pada dataset, konfigurasi model, pemilihan *collaborative filtering* memiliki skalabilitas yang baik dalam menghadapi jumlah pengguna dan item yang besar. Dalam metode *Collaborative Filtering*, perhitungan dapat dilakukan secara paralel dan dapat dijalankan pada sistem yang mendukung pemrosesan paralel atau distribusi. Ini memungkinkan sistem rekomendasi *Collaborative Filtering* untuk digunakan pada skala besar dengan jumlah pengguna dan item yang sangat besar.

  

## _Solution Statements_

  
dari permasalahan yang dialami, solusi yang dapat diterapkan dalam proses pengembangan sistem rekomendasi anime dapat menggunakan beberapa teknik dengan klasifikasikan kategori personalized diantara-Nya :

  

1. _Content based Filtering_: 
pada *content based filtering* sistem akan memilih dan melakukan peringkat item berdasarkan kesamaan profil pengguna dan profil item. Algoritma ini populer karena mudah digunakan dan kerjanya juga cepat. Algoritma penyaringan berbasis konten merekomendasikan sebuah item kepada pengguna berdasarkan kemiripan antara deskripsi item dan profil dari minat pengguna [2]. Keuntungan dari pendekatan ini adalah pengguna mendapatkan wawasan tentang mengapa suatu item dianggap relevan untuk mereka, karena konten di setiap item diketahui dari representasinya.
  

2. _*Collaborative Filtering*_: 
Metode *collaborative filtering* adalah salah satu metode pada sistem rekomendasi dimana sistem akan melakukan proses dengan dilakukan penjumlahan terhadap *rating* atau pilihan suatu produk, selanjutnya akan ditemukan pola atau profile dari pengguna dengan cara melihat history *rating* dari pengguna ke sistem yang akan memberikan rekomendasi baru berdasarkan pada perbandingan antar pola atau profile dari pengguna yang sudah ada [3]. pada *collaborative filtering* terdapat 2 jenis yaitu memory based dan model based. metode berbasis memori dibagi lagi menjadi 2 kategori, yaitu user based dan item based sedangkan jenis model based terbagi menjadi 3 pendekatan, yaitu cluster based algorithm, matrix factorization, dan deep learning.
  

3. _Hybrid Method_: 
Metode Hybrid adalah metode yang  dimana kedua metode ini digabungkan dengan harapan dapat mengurangi kelemahan dari satu metode melalui kelebihan dari satu metode lainnya dan menghasilkan sistem rekomendasi yang lebih baik [1]. metode gabungan yang dimaksud adalah gabungan antara *content based filtering* dan *collaborative filtering*
  

> **Pemilihan metode untuk penelitian ini menggunakan  *collaborative filtering* dengan teknik deep learning. pemilihan *collaborative filtering* memiliki skalabilitas yang baik dalam menghadapi jumlah pengguna dan item yang besar. Dalam metode *Collaborative Filtering*, perhitungan dapat dilakukan secara paralel dan dapat dijalankan pada sistem yang mendukung pemrosesan paralel atau distribusi. Ini memungkinkan sistem rekomendasi *Collaborative Filtering* untuk digunakan pada skala besar dengan jumlah pengguna dan item yang sangat besar.**

  

# 🧷 _Data Understanding_

  

## _Overview Dataset_

  

Dataset yang digunakan pada penelitian ini yaitu **_Anime Recommendations Database_**. Anime Recommendations Database adalah kumpulan data yang berisi informasi mengenai data preferensi pengguna dari 73516 pengguna terhadap 12294 anime. setiap pengguna dapat menambahkan anime ke daftar completed list dan memberinya *rating*. dataset ini terbagi menjadi 2 file, yaitu anime.csv dan *rating*.csv. Untuk detail variabel anime.csv dapat dilihat pada **Tabel 1**, dan untuk variabel *rating*.csv dapat dilihat pada **Tabel 2**.

  

**Link _Dataset_ : [Anime Recommendations Database](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database?select=rating.csv)**

  

**Jumlah _Record_ Data Anime.csv  : 12.294**

**Jumlah _Record_ Data *Rating*.csv  : 7.813.737**

  

**_Data Source_ : myanimelist.net**

  

**_Usage Ideas_ : _Recommendation System_**

  
  

**Tabel 1 Variabel Pada Anime.csv**

| **anime_id** |                      **name**                     |                     **genre**                     | **type** | **episodes** | **rating** | **members** |
|:------------:|:-------------------------------------------------:|:-------------------------------------------------:|:--------:|:------------:|:----------:|:-----------:|
|         3848 | One Piece Movie 9: Episode of Chopper Plus - F... | Action, Adventure, Comedy, Fantasy, Shounen, S... |    Movie |            1 |       7.65 |       29771 |


  

Keterangan :

> - ***anime_id* :** id unik dari myanimelist.net yang mengidentifikasi sebuah anime.

> - ***name* :** nama lengkap anime.

> - ***genre* :** daftar genre yang dipisahkan dengan koma untuk anime ini.  

> - ***type* :** movie, TV, OVA, etc.

> - ***episodes* :** berapa banyak episode dalam acara ini. (bernilai 1 jika film).

> - ***rating* :** nilai rata-rata dari 10 untuk anime ini.

> - ***members* :** jumlah anggota komunitas yang ada di anime ini  "grup".

 **Tabel 2 Variabel Pada *Rating*.csv**
 
 | **user_id** | **anime_id** | **rating** |
|:-----------:|:------------:|:----------:|
|          54 |           53 |         -1 |

Keterangan :

> - ***user_id* :** id pengguna dibuat secara acak..

> - ***anime_id* :** id unik dari myanimelist.net yang mengidentifikasi sebuah anime..

> - ***rating* :** peringkat dari 10 yang telah diberikan pengguna ini (-1 jika pengguna menontonnya tetapi tidak memberikan peringkat).

## _exploratory data analysis_

  
### _Preview Dataset_


**Tabel 3 Preview Data frame Anime**

| **anime_id** |                      **name**                     |                     **genre**                     | **type** | **episodes** | **rating** | **members** |
|:------------:|:-------------------------------------------------:|:-------------------------------------------------:|:--------:|:------------:|:----------:|:-----------:|
|         3848 | One Piece Movie 9: Episode of Chopper Plus - F... | Action, Adventure, Comedy, Fantasy, Shounen, S... |    Movie |            1 |       7.65 |       29771 |
|        29745 |                                   Korokoro Animal |                                              Kids |       TV |           60 |       6.00 |          43 |

 **Tabel 4 Preview Data frame *Rating*.csv**
 
 | **user_id** | **anime_id** | **rating** |
|:-----------:|:------------:|:----------:|
|          54 |           53 |         -1 |
|          44 |        14189 |          6 |

### Penggunaan Data
>  Pada proses penelitian ini data frame Anime & *Rating* yang digunakan hanya berjumlah **10000 Record Data**. Proses Pemotongan data dilakukan dengan library **pandas**. hal ini bertujuan untuk pemrosesan training pada Google Colab tidak memakan waktu lama.

### Filter Data Anime Dengan Menghapus Genre Negatif
> Anime memiliki banyak genre hingga memiliki segmentasi tontonan anak hingga dewasa. pada penelitian ini peneliti menghapus kategori negatif yang termasuk genre negatif salah satunya genre **'Hentai'**.

### Perubahan User_id di Dataframe *Rating*
> Data frame *Rating* memiliki variabel user_id, hanya saja penyajian user_id hanya berupa angka urut saja, untuk itu peneliti menambahkan abjad U di depan pada setiap user_id output data dapat dilihat pada **Tabel 5**.

**Tabel 5 output Perubahan User_id**

| **user_id** | **anime_id** | **rating** |
|:-----------:|:------------:|:----------:|
|          U5 |        25157 |          1 |
|         U93 |        15117 |          7 |
|         U22 |          853 |         10 |

### Perubahan Nilai *Rating*
> Pada Data frame *rating* ada yang memiliki nilai -1. Nilai umum *rating* pada Film , Anime biasanya dengan rentang 0 - 10. 0 disini dengan maksud bahwa user tidak memberikan *rating* pada anime yang di tonton. untuk itu untuk mengubah data agar tidak terjadi ambigu peneliti mengubah *rating* -1 menjadi *rating* 0

### Jumlah Pembagian Data Training & Data Testing

 
> Visualisasi pembagian data dan perbandingan jumlah data dapat dilihat pada `Tabel 6`. Data yang paling banyak digunakan adalah data *training* sebanyak 80% sedangkan untuk data *testing* digunakan sebanyak 20%.


**Tabel 6 Diagram Pembagian Dataset Dan Perbandingan Jumlah data**

  |                                                                                                          Jenis Diagram                                                                                                         |                                                                           Keterangan                                                                           |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| ![perbandingan jumlah data](https://github.com/candraburhanudin15/anime-recommendation-system-collaborative-filtering-using-tensorflow/assets/62823773/baa1d165-2258-465c-8936-6b7cdec46180) Gambar 1 Perbandingan Jumlah Data | Pada Gambar 1 Jumlah Data training yang digunakan sebanyak 7948 dan untuk data testing sebanyak 1978 dari jumlah total dataset yang digunakan berjumlah 10000.                                                                |
| ![dataset split](https://github.com/candraburhanudin15/anime-recommendation-system-collaborative-filtering-using-tensorflow/assets/62823773/4bd81c82-89fb-4539-82ef-028cac30aaf9)                                              Gambar 2 Pembagian Dataset | Pada Gambar 2 Dataset dibagi menjadi Data Training Dan Data Testing dengan perbandingan 80 : 20. |

  

### Sebaran Data *Rating*

![sebaran data rating](https://github.com/candraburhanudin15/anime-recommendation-system-collaborative-filtering-using-tensorflow/assets/62823773/06cac850-e589-4510-9701-fbe455e76788)

**Gambar 3 Sebaran Data *Rating***

> Dari total 10000 data yang digunakan terlihat masih banyak orang yang tidak memberikan penilaian pada anime ditandai dengan *rating* 0 pada diagram dengan jumlah 2204 user. disusul oleh penilaian tertinggi kedua jatuh pada *rating* yang diberikan user   banyak yang memberikan nilai anime dengan *rating* 8 dengan sebanyak 1899 orang yang memberikan nilai.

  

# 🧮 Data Preparation

  

**Proses yang dilakukan pada data preparation yaitu :**

  

1. _Encoding_ FItur _user_id_ dan _anime_id_
2. memetakan user_id dan anime_id pada data frame
3. Pengecekan Missing value
4. Standarisasi Data
5. _Train Val Split_

  

**Alasan Dilakukan Data Preparation :**


1. *Encoding* pada fitur *user_id dan anime_id* proses ini dilakukan untuk menyandingkan fitur user_id dan anime_id ke dalam indeks integer.
2. memetakan fitur user_id dan anime_id dilakukan untuk menambahkan fitur ke data frame untuk proses training data nanti.
3.  dengan melakukan pengecekan missing value peneliti dapat mengidentifikasi data yang hilang, menghindari bias atau distorsi untuk meningkatkan kualitas analisis serta menjadi suatu pengambilan keputusan untuk menangani data.
4. Dengan menggunakan standarisasi data, variabel dalam dataset dapat diperlakukan secara seragam dan dibandingkan dengan lebih baik. Ini membantu meningkatkan interpretasi hasil analisis, mengurangi bias, dan meningkatkan konsistensi dalam analisis dan pemodelan.
5. *Train Val Split* dilakukan untuk proses evaluasi kinerja, serta mempermudah dalam tuning parameter dengan membagi data training dan validation dengan perbandingan 80 : 20


  

## _Encoding_ FItur _user_id_ dan _anime_id_

  

> proses encoding user_id pertama dilakukan perubahan user_id menjadi list tanpa nilai yang sama lalu user_id dilakukan encode dengan fungsi `enumerate()`. Fungsi `enumerate()` mengembalikan objek enumerasi yang berupa iterasi. Setiap elemen dalam objek enumerasi adalah pasangan tuple yang terdiri dari indeks dan elemen. Indeks dimulai dari 0 dan bertambah satu setiap iterasi.  Proses encoding ini dilakukan serupa juga terhadap fitur anime_id.
Output:
> encode user_id : {20: 0, 24: 1, 79: 2, 226: 3, 241: 4, 355: 5, 356: 6,
> 442: 7, . . . } encode anime_id :{20: 0, 24: 1, 79: 2, 226: 3, 241: 4,355: 5, 356: 6, 442: 7, . . . }


## memetakan user_id dan anime_id pada dataframe

> setelah proses encoding pada user_id dan anime_id . fitur encoding user_id disimpan pada fitur baru dengan nama user dan pada fitur encoding anime_id disimpan pada fitur baru dengan nama anime. berikut output data frame *rating* setelah melalui filtrasi dan encoding pada user dan anime dapat dilihat pada **Tabel 7** 

**Tabel 7**  Dataframe *rating* dengan filtrasi dan proses encoding
  | user_id | anime_id | rating | user | anime |
|--------:|---------:|-------:|-----:|------:|
|     U54 |      105 |    0.0 |   53 |  1732 |
|     U60 |    22319 |    9.0 |   59 |   125 |
|     U53 |     6547 |   10.0 |   52 |    36 |


## Pengecekkan Missing value

untuk menghindari bias atau distorsi dilakukan proses pengecekan missing value pada data *rating* untuk meningkatkan kualitas analisis serta menjadi suatu pengambilan keputusan untuk menangani data. hasil pengecekan didapatkan tidak terdapat missing value pada data frame *rating*.
 
 **tabel 8 Hasil Pengecekan Missing Value Dataframe *Rating***
|  user_id | 0 |
|---------:|--:|
| anime_id | 0 |
| rating   | 0 |
| user     | 0 |
| anime    | 0 |
  

## Standarisasi Data

standarisasi data dilakukan pada nilai *rating* dimana nilai *rating* dengan rentang 0 - 10 diubah menjadi float
**Tabel 9 Hasil Standarisasi Fitur *Rating***

| **user_id** | **anime_id** | **rating** | **user** | **anime** |
|:-----------:|:------------:|:----------:|:--------:|:---------:|
|     U111    |     8795     |     8.0    |    110   |    927    |
|     U106    |     15051    |     7.0    |    105   |    1113   |
|     U14     |     21431    |     6.0    |    13    |    558    |
  

## *Train-Val-Split*

  

> Proses Train Val Split dilakukan dengan cara membuat variabel x dahulu untuk mencocokkan data user dan anime menjadi satu value menggunakan nilai fitur user dan anime pada data frame *rating*, lalu membuat variabel y untuk membuat *rating* dari hail akses kolom *rating* pada data frame *rating* dengan menerapkan sebuah fungsi lambda ke setiap elemen dalam series *rating*. Fungsi lambda ini melakukan normalisasi terhadap nilai *rating* dengan mengurangi nilai minimum (`min_rating`) dari setiap nilai *rating*, kemudian membaginya dengan selisih antara nilai maksimum (`max_rating`) dan nilai minimum. Dengan demikian, nilai *rating* akan dinormalisasi dalam rentang 0 hingga 1.

> Proses selanjutnya yaitu membagi data training sebanyak 80% dan data validation sebanyak 20% dengan total data yang digunakan 10000

  

# 🛠️ Modeling

  

Dari permasalahan tersebut metode yang digunakan pada penelitian ini adalah *Collaborative filtering*. penelitian ini menggunakan *Collaborative filtering* dengan metode berbasis model melalui pendekatan deep learning 

  

> Kelebihan Dan Kekurangan Collaboratory Filtering
> Kelebihan

1.  Tidak memerlukan informasi konten: *Collaborative Filtering* tidak bergantung pada informasi konten seperti genre, tag, atau deskripsi produk. Ini berarti sistem rekomendasi *Collaborative Filtering* dapat memberikan rekomendasi yang lebih obyektif dan tidak terbatas pada karakteristik konten tertentu. 
    
2.  Menangani "Cold Start" problem: *Collaborative Filtering* cenderung lebih efektif dalam menangani "Cold Start" problem, yaitu saat sistem rekomendasi dihadapkan pada item baru atau pengguna baru yang belum memiliki riwayat interaksi yang cukup. Dalam *Collaborative Filtering*, rekomendasi dapat didasarkan pada perilaku pengguna lain yang memiliki preferensi serupa, sehingga bisa memberikan rekomendasi yang relevan bahkan tanpa informasi konten yang cukup.
    
3.  Identifikasi pola dan preferensi yang kompleks: *Collaborative Filtering* dapat mengidentifikasi pola dan preferensi yang kompleks di antara pengguna. Ini memungkinkan sistem rekomendasi untuk menemukan hubungan yang lebih halus antara pengguna dan item, bahkan jika hubungan tersebut tidak jelas berdasarkan informasi konten. 
    
4.  Skalabilitas yang baik: *Collaborative Filtering* memiliki skalabilitas yang baik dalam menghadapi jumlah pengguna dan item yang besar. Dalam metode *Collaborative Filtering*, perhitungan dapat dilakukan secara paralel dan dapat dijalankan pada sistem yang mendukung pemrosesan paralel atau distribusi. Ini memungkinkan sistem rekomendasi *Collaborative Filtering* untuk digunakan pada skala besar dengan jumlah pengguna dan item yang sangat besar.
  
  >Kekurangan
1. Skalabilitas: *Collaborative Filtering* dapat menghadapi tantangan skala ketika jumlah pengguna atau item dalam sistem sangat besar. Semakin banyak pengguna dan item, semakin kompleks pula komputasi yang diperlukan untuk menghasilkan rekomendasi yang relevan. Hal ini dapat menyebabkan kinerja yang lambat dan membutuhkan sumber daya yang lebih besar.

2. *Collaborative Filtering* cenderung memberikan rekomendasi yang cenderung mengikuti tren popularitas. Ini terjadi karena item yang lebih populer akan memiliki lebih banyak interaksi dan lebih sering muncul dalam dataset. Akibatnya, item yang kurang populer atau mungkin lebih relevan bagi sebagian pengguna dapat terabaikan dalam rekomendasi.
  

## Sistem Rekomendasi dengan *Collaboratory Filtering*

  

### Sistem Rekomendasi Collaboratory Filtering menggunakan tensorflow keras
> Tahap yang dilakukan yaitu, pertama melakukan proses embedding terhadap data user dan data anime. selanjutnya lakukan operasi perkalian dot produk antara embedding user dan data anime, peneliti disini menambahkan bias untuk setiap fitur user dan fitur anime dan terakhir skor kecocokan akan ditetapkan dalam skala rentang 0, 1 menggunakan fungsi aktivasi sigmoid. 

> Proses Selanjutnya model dilakukan Compile menggunakan parameter `loss` dengan BinaryCrossentropy , `optimizer` yang digunakan adalah Adam, serta menggunakan parameter `metrics` RootMeanSquaredError. Proses Training menggunakan parameter `epochs` sebanyak 100.

  

>Penyelesaian Penelitian ini dapat menggunakan metode *Content Based Filtering* namun Pemilihan metode    pemilihan *collaborative filtering* memiliki skalabilitas yang baik dalam menghadapi jumlah pengguna dan item yang besar. Dalam metode *Collaborative Filtering*, perhitungan dapat dilakukan secara paralel dan dapat dijalankan pada sistem yang mendukung pemrosesan paralel atau distribusi. Ini memungkinkan sistem rekomendasi *Collaborative Filtering* untuk digunakan pada skala besar dengan jumlah pengguna dan item yang sangat besar dibanding dengan Content-based Filtering memiliki keterbatasan dalam Diversifikasi Rekomendasi: Content-based Filtering cenderung menghasilkan rekomendasi yang mirip atau serupa dengan item yang sudah disukai oleh pengguna. Hal ini dapat mengakibatkan kurangnya variasi dalam rekomendasi


  

# 📋*Evaluation*

  

> hasil proses training menghasilkan metrik Loss dan Accuracy dengan menggunakan method evaluate(). Method `evaluate()` akan mengembalikan hasil evaluasi dalam bentuk array, di mana setiap elemen array menunjukkan nilai metrik evaluasi yang dihitung. evaluasi ini ke dalam variabel atau langsung mencetaknya untuk ditampilkan dapat dilihat pada **Tabel 10**.

|     **Root Mean Squared Error (RMSE)**     | **val_Loss** |
|:--------------------:|:---------------------------:|
|  0.31835827231407166 | 0.6933207511901855          |

![hasil Training](https://github.com/candraburhanudin15/anime-recommendation-system-collaborative-filtering-using-tensorflow/assets/62823773/e1756f04-92ad-45b7-8694-c55f05bfd6ae)
**Gambar 4 Hasil Training**
>Dari hasil Training di atas nilai Val_loss masih tergolong cukup tinggi dibandingkan RMSE yang sudah memiliki tingkat error yang rendah
### Keterangan Metrik

1. Loss: Loss adalah metrik yang menggambarkan seberapa baik model dapat mempelajari pola-pola dalam data pelatihan. Loss mengukur kesalahan atau perbedaan antara prediksi model dan nilai yang sebenarnya dalam dataset. Tujuan dari model machine learning adalah untuk meminimalkan nilai loss, yang berarti model lebih baik dalam mempelajari pola-pola dan memberikan prediksi yang lebih akurat.
  
### *Loss*
Dalam sistem rekomendasi menggunakan *Collaborative Filtering*, umumnya digunakan fungsi loss berbasis perbedaan antara nilai prediksi dan nilai aktual dari interaksi pengguna dengan item. Salah satu contoh fungsi loss yang sering digunakan adalah Mean Squared Error (MSE). Berikut adalah persamaan fungsi loss MSE :

  $$MSE = (1 / N) * Σ[(r_{ui} - \hat{r}_{ui})^2]$$
  
Di mana:

-   $N$ adalah jumlah total interaksi pengguna dengan item dalam dataset.
-   $u$ adalah pengguna.
-   $i$ adalah item.
-   $r_{ui}$​ adalah nilai aktual dari interaksi pengguna u dengan item i.
-   $\hat{r}_{ui}$ adalah nilai prediksi dari interaksi pengguna u dengan item i yang dihasilkan oleh model *Collaborative Filtering*.

# Hasil Uji Coba

> Hasil uji coba menggunakan sample user_id acak, peneliti melakukan percobaan model prediksi dengan menampilkan anime yang memiliki *rating* tinggi  dari user dan menampilkan 10 rekomendasi anime untuk pengguna tersebut. Detail hasil uji coba dapat dilihat pada **Tabel 10**

**Tabel 10 Hasil Coba Model**

| No | Data Pengujian                        | Hasil Pengujian                                                                                       |
|----|----------------------------------|-------------------------------------------------------------------------------------------------------|
| 1  | **User_ID**                      | U52                                                                                                   |
| 2  | **Anime With ratings from user** | Tengen Toppa Gurren Lagann : Action, Adventure, Comedy, Mecha, Sci-Fi                                 |
|    |                                  | Darker than Black: Kuro no Keiyakusha : Action, Mystery, Sci-Fi, Super Power                          |
|    |                                  | Steins;Gate : Sci-Fi, Thriller                                                                        |
|    |                                  | Cowboy Bebop : Action, Adventure, Comedy, Drama, Sci-Fi, Space                                        |
|    |                                  | Durarara!!x2 Ketsu : Action, Mystery, Supernatural                                                    |
| 3  | **Top 10 Anime Recomendation**   | Neon Genesis Evangelion: The End of Evangelion : Dementia, Drama, Mecha, Psychological, Sci-Fi        |
|    |                                  | Rose of Versailles : Adventure, Drama, Historical, Romance, Shoujo                                    |
|    |                                  | Interstella5555: The 5tory of The 5ecret 5tar 5ystem : Adventure, Drama, Music, Sci-Fi                |
|    |                                  | Dragon Ball Z Special 1: Tatta Hitori no Saishuu Kessen : Adventure, Comedy, Fantasy, Sci-Fi, Shounen |
|    |                                  | Kimi ga Nozomu Eien : Drama, Romance, Slice of Life                                                   |
|    |                                  | Tokyo Majin Gakuen Kenpucho: Tou : Action, Drama, Fantasy, Horror, Martial Arts, School, Supernatural |
|    |                                  | K-On!: Live House! : Comedy, Music, School, Slice of Life                                             |
|    |                                  | Mahou Shoujo Lyrical Nanoha : Action, Comedy, Drama, Magic, Super Power                               |
|    |                                  | Mobile Suit Gundam : Action, Mecha, Military, Sci-Fi, Space                                           |
|    |                                  | Ninja Senshi Tobikage : Adventure, Mecha, Sci-Fi, Shounen, Space                                      |

Dari Hasil Uji Coba  pada **Tabel 10** Model memberikan hasil 10 rekomendasi anime teratas dengan berbagai macam genre

  

# Kesimpulan

  

Pembuatan Sistem Rekomendasi menggunakan *Collaborative Filtering* Telah Berhasil dilakukan,  model mampu memberikan rekomendasi anime berdasarkan preferensi pengguna dengan memanfaatkan data historis interaksi pengguna. Nilai Root Mean Squared Error (RMSE) telah mencapai angka error yang rendah namun untuk val_loss masih memiliki tingkat error yang cukup tinggi untuk itu perlu peningkatan dalam mengeksplorasi model yang lebih kompleks dengan berbagai macam teknik.
  

# Daftar Pustaka


[1] H. H. Arfisko and A. T. Wibowo, “Sistem Rekomendasi Film Menggunakan Metode Hybrid *Collaborative Filtering* Dan Content-based Filtering,” _eProceedings of Engineering_, vol. 9, no. 3, Jun. 2022, Accessed: Jul. 12, 2023. [Online]. Available: https://openlibrarypublications.telkomuniversity.ac.id/index.php/engineering/article/view/18066


[2] K. H. Muliadi and C. C. Lestari, “Rancang Bangun Sistem Rekomendasi Tempat Makan Menggunakan Algoritma Typicality Based *Collaborative Filtering*,” _Techno.com_, vol. 18, no. 4, pp. 275–287, Nov. 2019, doi: https://doi.org/10.33633/tc.v18i4.2515.
‌

[3] H. Februariyanti, A. D. Laksono, J. S. Wibowo, and M. S. Utomo, “IMPLEMENTASI METODE *COLLABORATIVE FILTERING* UNTUK SISTEM REKOMENDASI PENJUALAN PADA TOKO MEBEL,” _Jurnal Khatulistiwa Informatika_, vol. 9, no. 1, Jun. 2021, doi: https://doi.org/10.31294/jki.v9i1.9859.g4873.
  



  

**---Ini adalah bagian akhir laporan---**
