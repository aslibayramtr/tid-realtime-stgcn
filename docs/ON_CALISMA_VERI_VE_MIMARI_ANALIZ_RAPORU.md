# Ön Çalışma Veri ve Mimari Analiz Raporu

**Proje:** Bilgisayar Kamerası Üzerinden Gerçek Zamanlı Türk İşaret Dili (TİD) Tahmini ve Seslendirmesi Yapan Yapay Zeka Tabanlı Masaüstü Sistemi  
**Takım:** KNCO-1027 — Sena Kamiloğlu & Aslı Bayram  
**Danışman:** Dr. Öğr. Üyesi Kevser Nur Çoğalmış  
**Kurum:** Recep Tayyip Erdoğan Üniversitesi, Bilgisayar Mühendisliği  
**Rapor türü:** Güz Dönemi H9–H10 Ön Çalışma (Prototip Eğitim) öncesi teknik / veri hazırlık analizi  
**Tarih:** 17 Kasım 2026  
**Çalışma dizini:** `C:\Users\aslibayram\Desktop\on_calısma`

---

## 0. Yönetici Özet

`archive/AUTSL` altındaki RGB videolar, `train/val/test.csv` etiketleri ve **`SignList_ClassId_TR_EN.csv`** birlikte **ön çalışmaya hazırdır**. `outputs/veri_kontrol.py` çalıştırıldığında durum: **TAM HAZIR** (CSV ↔ video %100; 226/226 ClassId ↔ TR/EN eşlemesi; signer-independent split korunuyor; eksik/bozuk video yok).

Ara rapordaki üretim hedefi **133 noktanın tamamı değil**; MediaPipe Holistic’ten **46 düğüme anatomik budama** (42 el + 4 üst gövde), çok akışlı hafif ST-GCN, kayan pencere + asenkron `pyttsx3` şeklindedir. 133 düğüm ablasyon kolu olarak tutulmalıdır.

**Ön çalışma kararı:** 226 sınıf yerine **30 çekirdek TİD kelimesi** (4.829 video: train 3.737 / val 591 / test 501). Class ID listesi: `outputs/prototype_classes.txt`.

---

## 1. Görev 1 — Çalışma Alanı Envanteri

### 1.1. `on_calisma/` kök dizini

| Öğe | Durum | Not |
|---|---|---|
| `archive/AUTSL/` | Var | `train/`, `val/`, `test/` + CSV etiketleri + **SignList** |
| `KNCO-1027_O4_Gantt.xlsx` | Var | Güz + Bahar Gantt ve haftalık özet |
| `references.bib` | Var | ~92 IEEE kaynağı (Zotero dışa aktarımı) |
| `D4_Ara_Rapor_Sablonu.tex` | **Kökte yok** | Yalnızca `D4_Ara_Rapor_Sablonu.log` var |
| `.cursorignore` | Var | `archive/` ve `*.mp4` indeks dışı |

Ara rapor içeriği, masaüstündeki en güncel derlenmiş sürüm olan  
`C:\Users\aslibayram\Desktop\ö4_önçalışma\D4_Ara_Rapor_Sablonu.tex`  
üzerinden okunmuştur (aynı log’un kaynak `.tex` dosyası kökte bulunmadığı için).

### 1.2. Ara rapordan çıkarılan kapsam, mimari ve metrikler

**Kapsam:** Standart web kamerası ile gerçek zamanlı izole TİD kelime tahmini; kural tabanlı Türkçe ifade düzenleme; çevrimdışı seslendirme; yerel masaüstü (ham görüntü diske/ağa yazılmaz).

**Dört katmanlı mimari (ara rapor):**

1. **Katman 1 — Çok akışlı hafif ST-GCN:** Eklem / kemik / hareket akışları; geç füzyon; hedef ~2–3M parametre, &lt;15 MB model.
2. **Katman 2 — Anatomik budama + normalizasyon:** Holistic 543 noktadan üretim topolojisi **N = 46** (42 el + 4 üst gövde). Ablasyon: 42 ve **133**.
3. **Katman 3 — Kayan pencere + asenkron TTS:** Örtüşen pencere, güven eşiği, kararlılık/cooldown; `pyttsx3` ayrı thread + `queue`.
4. **Katman 4 — Yerel masaüstü:** OpenCV yakalama + PySide6 arayüz; mahremiyet odaklı.

**Başarı metrikleri (ara raporla hizalı):**

| Metrik | Hedef |
|---|---|
| Top-1 doğruluk (signer-independent, hedef kelime kümesi) | ≥ %85 |
| Canlı akış | 25–30 FPS |
| Model çıkarımı (kare/pencere) | &lt; 30 ms |
| Uçtan uca gecikme (işaret bitişi → ses) | &lt; 500 ms |
| Model boyutu | &lt; 15 MB |

> Not: Kullanıcı sorgusunda “133 keypoint MediaPipe Holistic” geçmektedir. Ara rapor bunu **üretim girdisi değil, ablasyon kolu** olarak tanımlar. Üretim omurgası **46 düğüm**dür. Bu rapor her iki ifadeyi de netleştirerek 46’yı önerir.

**Araç yığını (ara rapor Tablo):** Python 3.11+, PyTorch 2.x, MediaPipe 0.10.x, OpenCV 4.x, NumPy 1.26+, pyttsx3 2.90+, PySide6 6.x, scikit-learn 1.4+.

### 1.3. Gantt (`KNCO-1027_O4_Gantt.xlsx`) — H9–H10 ve Bahar kısıtları

**Güz H9 — Ön Çalışma I (Üye: Aslı Bayram):**  
Veri seti temini (AUTSL / Bosphorus22k) & MediaPipe ortamı → *Ön çalışma logu*.

**Güz H10 — Ön Çalışma II (Kaptan: Sena Kamiloğlu):**  
PyTorch ST-GCN prototip eğitimi & data pipeline → *Ön bulgular + Bahar Gantt*.

**İş bölümü özeti:**

| Dönem | Sorumlu ağırlık | Odak |
|---|---|---|
| Güz H9 | Aslı | Veri + MediaPipe boru hattı |
| Güz H10 | Sena | ST-GCN prototip + pipeline |
| Bahar H2 | Sena | Topoloji / hiperparametre |
| Bahar H3 | Aslı | Canlı OpenCV–MediaPipe akışı |
| Bahar H4 | Sena | Kural tabanlı TİD gramer motoru |
| Bahar H5 | Aslı | pyttsx3 TTS entegrasyonu |

Haftalık özet sayfasında H1–H6 tamamlanmış; H7 vize kısmi; **H9–H10 “Devam”** durumundadır. Bu analiz H9’un veri doğrulama adımını kapatmaya yöneliktir.

### 1.4. `references.bib` — ilgili standartlar

| Tema | Ana kaynaklar | Çıkarım |
|---|---|---|
| AUTSL tanımı / SI split | `sincan_autsl_2020` | 226 sınıf, 43 işaretçi, ~38k örnek; user-independent benchmark’ta RGB baseline %62,02 |
| El / üst gövde topolojisi | `ozdemir_hand_2024`, `ozdemir_cross_2025` | El odaklı çizge AUTSL/Bosphorus’ta güçlü; tam vücut şart değil |
| ST-GCN / çok akış | `parelli_spatio-temporal_2022`, `zhai_isolated_2025`, `qian_hand-aware_2025` | Komşuluk matrisi kodla tanımlanır; multi-stream AUTSL’de %96+ |
| MediaPipe + budama | `tran_sentence-level_2024`, `tran_anh_efficient_2025`, `chawla_openhands_2026` | Keypoint seçimi kritik; 46 nokta ile yüksek doğruluk / düşük gecikme raporları |
| AUTSL100 alt küme prototipi | `chawla_openhands_2026` | Ön çalışmada alt küme ile boru hattı doğrulama literatürde kabul görmüştür |

**Sonuç:** Veri setinde hazır adjacency matrix **yoktur**. Anatomik komşuluk PyTorch tarafında tanımlanmalıdır. Split standardı **signer-independent**’tır (`sincan_autsl_2020`).

### 1.5. `archive/` veri seti istatistikleri

```
archive/
└── AUTSL/
    ├── train/   (*.mp4)          # 28.142
    ├── val/     (*.mp4)          # 4.418
    ├── test/    (*.mp4)          # 3.742
    ├── train.csv
    ├── val.csv
    ├── test.csv
    └── SignList_ClassId_TR_EN.csv   # 226 satır: ClassId,TR,EN
```

#### 1.5.1. Sayısal özet

| Split | Video (`.mp4`) | CSV satırı | Sınıf | İşaretçi sayısı | Dosya eşleşmesi |
|---|---:|---:|---:|---:|---|
| train | 28.142 | 28.142 | 226 | 31 | %100 |
| val | 4.418 | 4.418 | 226 | 6 | %100 |
| test | 3.742 | 3.742 | 226 | 6 | %100 |
| **Toplam** | **36.302** | **36.302** | **226 (ID 0–225)** | **43 (örtüşmesiz)** | **Tam** |

CSV–disk farkı: 0 (eksik/fazla video yok, mükerrer yok).

#### 1.5.2. İşaretçi ayrımı (signer-independent — doğrulandı)

- **Train:** signer0,2,3,4,5,7,8,9,10,12,13,15,17,19,20,21,22,23,24,26,28,29,31,32,33,36,37,38,40,41,42  
- **Val:** signer1,11,16,18,25,35  
- **Test:** signer6,14,27,30,34,39  

Üç küme arasında **işaretçi kesişimi yok** (`set` kesişimleri boş). Bu, ara rapordaki “işaretçiden bağımsız ≥ %85” metriğiyle uyumludur.

#### 1.5.3. Video özellikleri (OpenCV ile 150 örnek)

| Özellik | Gözlem |
|---|---|
| Çözünürlük | **512 × 512** (örneklerin tamamı) |
| FPS | **30.0** |
| Süre | ~1,23–3,03 sn (medyan ~2,0 sn) |
| Kare sayısı | ~37–91 (medyan 60) |
| Dosya boyutu | tipik 0,2–0,9 MB; toplam ~14 GB |
| Modalite | Yalnızca `*_color.mp4` (RGB); Kinect depth/skeleton dosyası yok |
| SignList | **Var** — `SignList_ClassId_TR_EN.csv` (226 sınıf, boş gloss yok) |
| Diğer belgeler | `.docx` / `.pdf` yok |

Makaledeki ~38.336 ile buradaki 36.302 farkı, bu arşivin **RGB challenge split** (color) olması ve depth/iskelet modalitelerinin dahil edilmemesiyle açıklanır; proje zaten MediaPipe ile RGB’den iskelet üreteceği için bu yeterli ve uygundur.

#### 1.5.4. Etiket CSV formatı (mevcut)

Başlık satırı yok; iki sütun:

```text
signer0_sample1_color.mp4,41
signer0_sample2_color.mp4,104
```

| Sütun | Anlam |
|---|---|
| 0 | Video dosya adı (`signer{SID}_sample{N}_color.mp4`) |
| 1 | Sınıf ID (0–225 tamsayı) |

Sınıf başına toplam örnek: 121–164 (min `seker`/172 = 121; max birçok sınıf = 164; dengeli).

#### 1.5.5. SignList eşleme doğrulaması (17 Kasım 2026 — güncel)

`outputs/veri_kontrol.py` çıktısı:

| Kontrol | Sonuç |
|---|---|
| `SignList_ClassId_TR_EN.csv` | VAR — 226 satır (`ClassId,TR,EN`) |
| ClassId kapsamı | 0–225; veri setindeki tüm ID’ler SignList’te |
| Boş TR/EN gloss | 0 |
| CSV ↔ MP4 eksik/fazla | 0 / 0 (üç split) |
| Signer kesişimi | train ∩ val ∩ test = ∅ |
| Genel durum | **TAM HAZIR** |

Eşleme kuralı (doğrulandı):

```text
video_adı  --(train|val|test).csv-->  ClassId  --(SignList)-->  TR / EN
örnek: signer0_sample1_color.mp4 → 41 → cirkin / ugly
```

Üretilen yardımcı tablolar:

- `outputs/etiket_ozet.csv` — video düzeyinde eşleme (36.302 satır)
- `outputs/sinif_ozet.csv` — sınıf düzeyinde train/val/test sayıları
- `outputs/prototype_classes.csv` / `.txt` — ön çalışma 30’lu alt küme

---

## 2. Görev 2 — Beş Kritik Başlık

### 2.1. Veri seti uygunluğu ve eksik bileşenler

#### Doğrudan kullanıma uygun mu?

**Evet — hem eğitim hem TR etiket/TTS hazırlığı için uygun.**

Uygun olanlar:

- İzole TİD videoları (AUTSL RGB)
- Hazır train/val/test ayrımı (SI)
- Dosya adı ↔ sınıf ID ↔ Türkçe/İngilizce gloss eşlemesi
- Tekdüze 512² / 30 FPS (ön işleme yükünü azaltır)

Kalan (beklenen) bileşenler — artık etiket değil, boru hattı ürünleri:

| Bileşen | Zorunluluk | Açıklama |
|---|---|---|
| Hazır `.npy` iskelet | Üretilecek | MediaPipe Holistic → budama(46) → normalizasyon |
| Adjacency / graph dosyası | Kodda | Anatomik A matrisi PyTorch’ta tanımlanacak |
| `train_labels.csv` vb. alternatif adlar | Gerekmez | Mevcut `train.csv` / `val.csv` / `test.csv` yeterli |
| Bosphorus22k | İsteğe bağlı | Gantt’ta alternatif; H9 için AUTSL yeterli |

#### MediaPipe → `.npy` öncesi video ön işleme ihtiyacı

AUTSL örnekleri zaten 30 FPS / 512² olduğu için **ağır yeniden kodlama şart değil**. Yine de boru hattında şunlar tanımlanmalıdır:

| Adım | Öneri | Gerekçe |
|---|---|---|
| FPS | 30 varsay; sapmada yeniden örnekle | Zamansal kenar tutarlılığı |
| Pencere uzunluğu T | 64 kare (≈2,1 sn) — kısa videoda pad, uzunda kırp/merkez | ST-GCN sabit T ister |
| Padding | Süre &lt; T ise son kare tekrarı veya sıfır + maske | MediaPipe kayıpları için de |
| Koordinat | Bilek orijini + avuç rotasyonu + bilek–MCP ölçeği; gövde omuz ortası | `long_sign_2023`; ara rapor Katman 2 |
| Ayna artırımı | Sol/sağ el etiketlerini dikkatle değiştirerek | Genelleme; yanlış ayna SI’yi bozar |
| Güven kanalı | Görünürlük skoru C’ye ekle | Occlusion maskeleme |
| Çıktı tensör | `(C, T, V, M)` — örn. C=3 veya 4, T=64, V=46, M=1 | Yan / OpenMMLab ST-GCN geleneği |

---

### 2.2. Etiket eşleştirme ve çizge (adjacency) yapısı

#### Türkçe kelime nasıl çözümlenir?

1. `*.csv` → `ClassId`  
2. `SignList_ClassId_TR_EN.csv` → `TR` (**doğrulandı, 226/226**)  
3. Kural motoru gloss dizisini Türkçe sözdizimine çevirir  
4. `pyttsx3` seslendirir  

Ön çalışmada model çıktısı ClassId → SignList TR → (isteğe bağlı) kural motoru → TTS zinciri kurulabilir.

#### Adjacency matrix arşivde var mı?

**Hayır.** AUTSL RGB paketinde çizge topolojisi bulunmaz. ST-GCN komşuluk matrisi, MediaPipe eklem indekslerinin anatomik kemik listesinden PyTorch’ta üretilir.

Ara rapor + kaynakçaya göre doğru kurgu:

1. **Üretim topolojisi (önerilen):** N=46  
   - Sol el 21 + sağ el 21 + 4 üst gövde referansı  
   - El içi kemik kenarları + komşu parmak ucu kenarları (`ozdemir_hand_2024`)  
   - El bileği ↔ ilgili omuz / üst gövde bağlantıları  
2. **Ablasyon A:** N=42 (yalnız eller)  
3. **Ablasyon B:** N=133 (geniş; yüz/gövde daha zengin — ara rapordaki “133’ten 46’ya budama” ifadesinin geniş kolu)  
4. Holistic ham çıktı 543 noktadır; **468 yüz noktasının tamamı üretime alınmamalıdır** (gürültü + gecikme).

Uzamsal-zamansal çizge:

- Uzamsal kenar: aynı karede anatomik (ve seçilmiş ekstra el) bağlar → A  
- Zamansal kenar: ardışık karelerde aynı düğüm  
- Normalize: \(\hat{A} = \tilde{D}^{-1/2}(A+I)\tilde{D}^{-1/2}\)  
- Çok akış: joint / bone / motion (`zhai_isolated_2025`)

---

### 2.3. Ölçeklendirme ve train / val / test ayrımı

#### Ön çalışmada kaç sınıf?

| Senaryo | Sınıf | Video (gerçek) | Ne zaman |
|---|---:|---:|---|
| **Kilitlenen prototip (H9–H10)** | **30** | **4.829** (tr 3.737 / va 591 / te 501) | Ortam + pipeline + ilk Top-1 |
| Ara ölçek (Bahar erken) | 50–100 | ~8.000–16.000 | Topoloji/hiperparametre |
| Tam AUTSL | 226 | 36.302 | Asıl çalışma / nihai metrik |

#### Kilitlenen 30 sınıflık çekirdek TİD alt kümesi

Seçim ölçütleri: (i) kural motoruyla anlamlı kısa Türkçe ifadeler, (ii) nezaket/kişi/soru/eylem/yer/zaman dengesi, (iii) her sınıfta yeterli SI örnek (≥151 toplam). Dosyalar: `outputs/prototype_classes.txt`, `outputs/prototype_classes.csv`.

| # | ClassId | TR | EN | Kategori | train | val | test | toplam |
|---|---:|---|---|---|---:|---:|---:|---:|
| 1 | 173 | selam | hi | nezaket | 127 | 20 | 17 | 164 |
| 2 | 93 | hoscakal | goodbye | nezaket | 127 | 20 | 17 | 164 |
| 3 | 196 | tesekkur | thanks | nezaket | 126 | 20 | 17 | 163 |
| 4 | 65 | evet | yes | nezaket | 125 | 20 | 15 | 160 |
| 5 | 86 | hayir | no | nezaket | 121 | 20 | 17 | 158 |
| 6 | 151 | ozur_dilemek | apologize | nezaket | 127 | 20 | 17 | 164 |
| 7 | 30 | ben | I | kişi | 127 | 20 | 17 | 164 |
| 8 | 175 | sen | you | kişi | 124 | 18 | 17 | 159 |
| 9 | 34 | biz | we | kişi | 118 | 20 | 17 | 155 |
| 10 | 14 | anne | mother | kişi | 119 | 20 | 17 | 156 |
| 11 | 20 | baba | father | kişi | 125 | 19 | 16 | 160 |
| 12 | 15 | arkadas | friend | kişi | 121 | 20 | 15 | 156 |
| 13 | 111 | kim | who | soru | 126 | 20 | 17 | 163 |
| 14 | 136 | nasil | how | soru | 126 | 20 | 16 | 162 |
| 15 | 138 | nerede | where | soru | 127 | 20 | 17 | 164 |
| 16 | 137 | neden | why | soru | 127 | 20 | 17 | 164 |
| 17 | 209 | yapmak | do | eylem | 127 | 20 | 17 | 164 |
| 18 | 217 | yemek | eat | eylem | 127 | 20 | 17 | 164 |
| 19 | 94 | icmek | drink | eylem | 123 | 19 | 17 | 159 |
| 20 | 22 | bakmak | look | eylem | 122 | 20 | 17 | 159 |
| 21 | 76 | gormek | see | eylem | 122 | 19 | 17 | 158 |
| 22 | 29 | beklemek | wait | eylem | 125 | 20 | 17 | 162 |
| 23 | 211 | yardim | help | eylem | 127 | 20 | 14 | 161 |
| 24 | 64 | ev | house | yer | 127 | 20 | 17 | 164 |
| 25 | 144 | okul | school | yer | 126 | 20 | 17 | 163 |
| 26 | 100 | iyi | good | durum | 127 | 20 | 17 | 164 |
| 27 | 122 | kotu | bad | durum | 126 | 20 | 17 | 163 |
| 28 | 212 | yarin | tomorrow | zaman | 117 | 17 | 17 | 151 |
| 29 | 55 | dun | yesterday | zaman | 122 | 19 | 17 | 158 |
| 30 | 38 | cay | tea | nesne | 126 | 20 | 17 | 163 |

**Class ID listesi (kopyala-yapıştır):**

```text
173,93,196,65,86,151,30,175,34,14,20,15,111,136,138,137,209,217,94,22,76,29,211,64,144,100,122,212,55,38
```

Örnek kural motoru gloss zincirleri: `selam` → `ben` `iyi` → `tesekkur` `hoscakal`; `kim` `nerede`; `ben` `cay` `icmek`; `yarin` `okul`.

Literatür dayanağı: OpenHands AUTSL100 alt kümesiyle prototipleme (`chawla_openhands_2026`); ara rapor risk maddesi “önce AUTSL alt kümesinde boru hattı doğrulaması”.

#### Split kriteri

**Mevcut AUTSL CSV split’ini aynen koruyun.** Rastgele karıştırmayın.

- Train / val / test zaten signer-independent.  
- Alt kümede de aynı dosya listelerinden ClassId filtresi uygulayın (sızıntı olmasın).  
- Metrik: test setinde Top-1 / Top-5 + sınıf başına F1; canlıda FPS ve uçtan uca ms.

---

### 2.4. Ön çalışma öncesi teknik hazırlık listesi

#### Ortam (ara raporla uyumlu, Windows)

```text
Python 3.11.x
PyTorch 2.x          # CUDA varsa GPU eğitim; CPU çıkarım hedef
mediapipe 0.10.x
opencv-python 4.x    # (ortamda 5.x de çalışır; 4.x ara raporla uyumlu)
numpy 1.26+          # MP/torch ile pinleyin; 2.x sorun çıkarırsa 1.26’ya dönün
pyttsx3 >= 2.90
PySide6 6.x
scikit-learn, matplotlib, seaborn, tqdm, pandas
```

Kurulum doğrulama:

1. `python -c "import torch, mediapipe, cv2, numpy; print(torch.__version__, cv2.__version__)"`  
2. Webcam + MediaPipe Holistic tek kare smoke test  
3. GPU varsa `torch.cuda.is_available()`  

#### `.npy` standardı

```text
şekil: (C, T, V, M)
C = 3 (x,y,z) veya 4 (x,y,z,confidence)
T = 64 (sabit pencere)
V = 46 (üretim) / 42 / 133 (ablasyon)
M = 1  (tek kişi)
dtype: float32
dosya: {stem}.npy  +  labels.csv (stem, class_id, split, tr_gloss?)
```

#### Doğrulama adımları (H9 kontrol listesi)

- [x] `veri_kontrol.py` yeşil özet (CSV–video eşleşmesi) — **TAM HAZIR**  
- [x] `SignList` eklendi; TR eşleme + `etiket_ozet.csv` / `sinif_ozet.csv` üretildi  
- [x] 30 sınıf prototip listesi kilitlendi (`prototype_classes.txt`)  
- [ ] 100–500 videoluk MediaPipe deneme çıkarımı; NaN / sıfır oran raporu  
- [ ] `(C,T,V,M)` şekil assert’leri (V=46, T=64)  
- [ ] Dummy ST-GCN 1 epoch overfit (tek batch, 30 sınıf)  
- [ ] CPU’da pencere çıkarım süresi ölçümü (&lt;30 ms hedefi için baz çizgi)

---

### 2.5. Ara rapor / literatür karşılaştırması ve alternatif öneriler

| Karar | Ara rapor / bib dayanağı | Mühendislik önerisi |
|---|---|---|
| Girdi | MediaPipe iskelet | **Aynen** — RGB CNN elendi (SI’de sert düşüş) |
| Düğüm sayısı | 46 üretim; 42/133 ablasyon | **Önce 46**; 133’ü H10 sonu / Bahar’a bırakın |
| Model | Çok akışlı hafif ST-GCN | Prototipte önce **tek akış joint**; sonra bone/motion ekleyin (takvim riskini düşürür) |
| Veri ölçeği | AUTSL 226 | **H9–H10: 30 sınıf (4.829 video)**; 226 Bahar |
| Split | SI | **Mevcut CSV split** |
| TTS | Asenkron pyttsx3 | Ana thread’e bağlanmayın |
| Kural motoru | Hafif kural katmanı | Çekirdek kelime kümesi gloss’larıyla paralel tasarlanmalı |

**Neden 46 &gt; 133 (ön çalışma için)?**

- Ara rapor Katman 2 ve risk tablosu: 543→46 budama, CPU 25–30 FPS için kritik.  
- `tran_sentence-level_2024`: 46 nokta ile yüksek doğruluk + düşük pencere gecikmesi.  
- `ozdemir_hand_2024`: El topolojisi tam vücutla rekabet edebiliyor.  
- 133/yüz ağırlıklı graf: mikro yüz gürültüsü + parametre/gecikme artışı; H9–H10 takvimini zorlar.

**Tek akış → çok akış kademesi:** Literatürde multi-stream AUTSL’de %96+ verir; ancak ön çalışmada boru hattı doğruluğu için joint-only yeterli bir kilometre taşıdır.

---

## 3. Adım Adım Yol Haritası (H9 → H10 → Bahar)

### H9 (Aslı) — Veri & MediaPipe

1. ~~`SignList` temin~~ **tamam** (`archive/AUTSL/SignList_ClassId_TR_EN.csv`)  
2. ~~`veri_kontrol.py`~~ **tamam** (TAM HAZIR; 30’lu alt küme dışa aktarıldı)  
3. Python 3.11 ortamını kur; paket sürümlerini sabitle (`requirements.txt`)  
4. `outputs/prototype_classes.txt` ile 30 sınıfı filtrele  
5. MediaPipe Holistic → budama(46) → normalizasyon → `.npy` (önce 200–500, sonra 4.829)  
6. Kalite raporu: kaçırılan el oranı, ortalama güven

### H10 (Sena) — ST-GCN prototip

1. Anatomik A matrisi (46) kodla  
2. Dataset: `.npy` + label; **yalnızca 30 ClassId** + mevcut SI split  
3. Mini ST-GCN (joint) eğit; val Top-1 izle  
4. Test SI doğruluk not et (hedef ≥%85 yönünde erken sinyal)  
5. CPU latency ölç; Bahar Gantt’ı güncelle

### Bahar (özet)

- H2: topoloji/hiperparametre + multi-stream  
- H3: canlı OpenCV hattı  
- H4: kural motoru (çekirdek sözlük)  
- H5: asenkron TTS  
- H6–H7: bütünleşik metrikler (≥%85, &lt;500 ms, 25–30 FPS)

---

## 4. Sonuç Kararları (Kilitleme Listesi)

1. **Veri:** `archive/AUTSL` RGB + `train/val/test.csv` SI split + **SignList** — doğrulandı.  
2. **Etiket eşleme:** 226/226 ClassId → TR/EN OK; eksik video yok.  
3. **Çizge:** Arşivde yok; PyTorch’ta 46 düğümlü anatomik A (+ el ucu kenarları).  
4. **Ön çalışma ölçeği:** **30 sınıf / 4.829 video** (`prototype_classes.txt`); tam 226 Bahar.  
5. **Tensör:** `(C,T,V,M)` float32, V=46, T=64 önerisi.  
6. **133 nokta:** üretim değil; ablasyon.  
7. **Metrik hizası:** SI Top-1 ≥%85 (30’lu kümede), &lt;500 ms E2E, 25–30 FPS.

---

## 5. Kaynak Dayanakları (bu analizde kullanılan)

- Ara rapor: `ö4_önçalışma/D4_Ara_Rapor_Sablonu.tex`  
- Gantt: `KNCO-1027_O4_Gantt.xlsx`  
- Kaynakça: `references.bib` (`sincan_autsl_2020`, `ozdemir_hand_2024`, `zhai_isolated_2025`, `qian_hand-aware_2025`, `tran_sentence-level_2024`, `chawla_openhands_2026`, `parelli_spatio-temporal_2022`, …)  
- Veri taraması: `archive/AUTSL` + OpenCV örnekleme + CSV bütünlük kontrolü  
- Yardımcı betik: `outputs/veri_kontrol.py`

---

*Bu belge Ön Çalışma I–II’ye geçiş kapısı (gate) niteliğindedir. Veri/etiket kapısı **açık**tır. Sonraki kapı: MediaPipe → `.npy` boru hattı ve 30 sınıflı ST-GCN prototip eğitimi.*
