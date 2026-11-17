# Gerçek Zamanlı Türk İşaret Dili (TİD) Tahmini ve Seslendirme Sistemi

Standart bilgisayar kamerası üzerinden el, kol ve omuz eklem noktalarını (MediaPipe, V=46) gerçek zamanlı çıkararak hafif ST-GCN mimarisi ile izole TİD kelimelerini tanıyan, kural tabanlı gramer motoru ile kurallı Türkçe cümlelere dönüştüren ve çevrimdışı seslendiren (pyttsx3) masaüstü uygulaması.

## Proje Yapısı
- `configs/`: Model ve 30 sınıflık çekirdek kelime yapılandırmaları
- `scripts/`: Sıralı veri kontrol, öznitelik çıkarım ve eğitim betikleri (`01_...`, `02_...`)
- `src/`: Çizge topolojisi, normalizasyon, ST-GCN modeli, tampon yönetimi, TİD gramer kuralları, TTS ve arayüz modülleri
- `docs/`: Teknik analiz raporları, Gantt planı ve kaynakça
- `outputs/`: Sınıf özet tabloları, eğitim logları ve model çıktıları

## Kurulum

    py -3.11 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
