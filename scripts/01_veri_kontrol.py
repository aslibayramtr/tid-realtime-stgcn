#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUTSL veri bütünlük ve etiket eşleme kontrol betiği
Proje: KNCO-1027 — Gerçek Zamanlı TİD Tahmini (Ön Çalışma)

Kullanım:
    python scripts/01_veri_kontrol.py
    python scripts/01_veri_kontrol.py --root "C:/Users/aslibayram/Desktop/on_calisma"
    python scripts/01_veri_kontrol.py --sample-videos 20 --probe
    python scripts/01_veri_kontrol.py --export outputs/etiket_ozet.csv

Beklenen yapı:
    data/raw/AUTSL/
        train/  *.mp4
        val/    *.mp4
        test/   *.mp4
        train.csv
        val.csv
        test.csv
        SignList_ClassId_TR_EN.csv   # opsiyonel ama TR eşleme için kritik
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


SPLITS = ("train", "val", "test")

# Resmi AUTSL / challenge ile uyumlu olası sınıf sözlüğü dosya adları
SIGNLIST_CANDIDATES = (
    "SignList_ClassId_TR_EN.csv",
    "SignList_ClassId_TR_EN.txt",
    "class_id_tr_en.csv",
    "sign_list.csv",
    "SignList.csv",
)

# Alternatif etiket dosya adları (yoksa uyarı; mevcut train/val/test.csv yeterli)
ALT_LABEL_NAMES = (
    "train_labels.csv",
    "validation_labels.csv",
    "val_labels.csv",
    "test_labels.csv",
)


@dataclass
class SplitReport:
    name: str
    csv_path: Optional[Path] = None
    n_csv_rows: int = 0
    n_csv_unique: int = 0
    n_videos_disk: int = 0
    missing_on_disk: List[str] = field(default_factory=list)
    extra_on_disk: List[str] = field(default_factory=list)
    n_classes: int = 0
    signers: List[str] = field(default_factory=list)
    class_counts: Counter = field(default_factory=Counter)
    rows: List[Tuple[str, int]] = field(default_factory=list)  # (filename, class_id)


def find_project_root(explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    here = Path(__file__).resolve().parent
    # scripts/ -> on_calisma/
    candidate = here.parent
    if (candidate / "data" / "raw" / "AUTSL").exists():
        return candidate
    cwd = Path.cwd().resolve()
    if (cwd / "data" / "raw" / "AUTSL").exists():
        return cwd
    return candidate


def read_split_csv(path: Path) -> List[Tuple[str, int]]:
    """Başlıksız 'filename,class_id' veya başlıklı CSV'yi okur."""
    rows: List[Tuple[str, int]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for i, parts in enumerate(reader):
            if not parts or all(not str(p).strip() for p in parts):
                continue
            if len(parts) < 2:
                raise ValueError(f"{path.name} satır {i + 1}: en az 2 sütun bekleniyor → {parts}")
            fn = str(parts[0]).strip()
            cid_raw = str(parts[1]).strip()
            # Başlık satırını atla
            if i == 0 and not cid_raw.lstrip("-").isdigit():
                continue
            if not fn.lower().endswith(".mp4"):
                # Bazı etiketlerde uzantısız isim olabilir
                pass
            try:
                cid = int(float(cid_raw))
            except ValueError as exc:
                raise ValueError(f"{path.name} satır {i + 1}: sınıf ID sayı değil: {cid_raw}") from exc
            rows.append((fn, cid))
    return rows


def load_signlist(path: Path) -> Dict[int, Dict[str, str]]:
    """
    ClassId -> {TR, EN, ...} sözlüğü.
    Esnek sütun adları: ClassId/class_id/id , TR/tr/Turkish , EN/en/English
    """
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        if reader.fieldnames is None:
            raise ValueError(f"SignList boş veya başlıksız: {path}")

        def pick(names: Sequence[str], options: Sequence[str]) -> Optional[str]:
            lower = {n.lower().strip(): n for n in names if n}
            for opt in options:
                if opt in lower:
                    return lower[opt]
            return None

        fields = list(reader.fieldnames)
        id_col = pick(fields, ("classid", "class_id", "id", "label", "class"))
        tr_col = pick(fields, ("tr", "turkish", "tr_name", "gloss_tr", "sign_tr", "kelime"))
        en_col = pick(fields, ("en", "english", "en_name", "gloss_en", "sign_en"))

        # Başlık yoksa / bulunamadıysa konumsal dene: ClassId,TR,EN
        if id_col is None:
            # DictReader yanlış başlık almış olabilir; ham oku
            f.seek(0)
            raw = list(csv.reader(f, dialect=dialect))
            if not raw:
                return {}
            start = 0
            if raw[0] and not str(raw[0][0]).lstrip("-").isdigit():
                start = 1
            mapping: Dict[int, Dict[str, str]] = {}
            for parts in raw[start:]:
                if len(parts) < 2:
                    continue
                cid = int(float(parts[0]))
                mapping[cid] = {
                    "TR": parts[1].strip() if len(parts) > 1 else "",
                    "EN": parts[2].strip() if len(parts) > 2 else "",
                }
            return mapping

        mapping = {}
        for row in reader:
            cid = int(float(str(row[id_col]).strip()))
            mapping[cid] = {
                "TR": (row.get(tr_col) or "").strip() if tr_col else "",
                "EN": (row.get(en_col) or "").strip() if en_col else "",
            }
        return mapping


def find_signlist(autsl_dir: Path) -> Optional[Path]:
    for name in SIGNLIST_CANDIDATES:
        p = autsl_dir / name
        if p.exists():
            return p
    # Esnek arama
    for p in autsl_dir.glob("*"):
        if p.is_file() and "signlist" in p.name.lower() and p.suffix.lower() in {".csv", ".txt"}:
            return p
    return None


def analyze_split(autsl_dir: Path, split: str) -> SplitReport:
    report = SplitReport(name=split)
    csv_path = autsl_dir / f"{split}.csv"
    video_dir = autsl_dir / split

    if csv_path.exists():
        report.csv_path = csv_path
        report.rows = read_split_csv(csv_path)
        report.n_csv_rows = len(report.rows)
        names = [fn for fn, _ in report.rows]
        report.n_csv_unique = len(set(names))
        report.class_counts = Counter(cid for _, cid in report.rows)
        report.n_classes = len(report.class_counts)
        report.signers = sorted({fn.split("_")[0] for fn, _ in report.rows})

    if video_dir.exists():
        disk = {p.name for p in video_dir.glob("*.mp4")}
        report.n_videos_disk = len(disk)
        if report.rows:
            csv_names = {fn for fn, _ in report.rows}
            report.missing_on_disk = sorted(csv_names - disk)
            report.extra_on_disk = sorted(disk - csv_names)
    return report


def probe_videos(paths: Sequence[Path], limit: int = 20) -> List[dict]:
    try:
        import cv2  # type: ignore
    except ImportError:
        print("[UYARI] OpenCV (cv2) yok; --probe atlandı. Kurulum: pip install opencv-python")
        return []

    picks = list(paths)
    random.shuffle(picks)
    picks = picks[:limit]
    out = []
    for p in picks:
        cap = cv2.VideoCapture(str(p))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ok, _ = cap.read()
        cap.release()
        out.append(
            {
                "file": p.name,
                "split": p.parent.name,
                "width": w,
                "height": h,
                "fps": round(fps, 2),
                "frames": n,
                "duration_sec": round(n / fps, 3) if fps else None,
                "readable": bool(ok),
            }
        )
    return out


def print_table(headers: Sequence[str], rows: Sequence[Sequence[object]], max_rows: int = 30) -> None:
    cols = list(headers)
    data = [list(map(str, r)) for r in rows[:max_rows]]
    widths = [len(c) for c in cols]
    for r in data:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))
    fmt = " | ".join(f"{{:<{w}}}" for w in widths)
    line = "-+-".join("-" * w for w in widths)
    print(fmt.format(*cols))
    print(line)
    for r in data:
        print(fmt.format(*r))
    if len(rows) > max_rows:
        print(f"... ({len(rows) - max_rows} satır daha)")


def build_matched_rows(
    reports: Dict[str, SplitReport],
    signlist: Optional[Dict[int, Dict[str, str]]],
) -> List[dict]:
    matched = []
    for split, rep in reports.items():
        for fn, cid in rep.rows:
            tr = en = ""
            if signlist is not None:
                info = signlist.get(cid, {})
                tr = info.get("TR", "")
                en = info.get("EN", "")
            matched.append(
                {
                    "split": split,
                    "filename": fn,
                    "class_id": cid,
                    "TR": tr,
                    "EN": en,
                    "signer": fn.split("_")[0] if "_" in fn else "",
                }
            )
    return matched


def export_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["split", "filename", "class_id", "TR", "EN", "signer"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] Özet tablo yazıldı: {path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="AUTSL archive veri kontrol ve etiket eşleme")
    parser.add_argument("--root", type=str, default=None, help="on_calisma kök dizini")
    parser.add_argument("--sample-videos", type=int, default=12, help="Özet için örnek video sayısı")
    parser.add_argument("--probe", action="store_true", help="OpenCV ile çözünürlük/FPS örnekle")
    parser.add_argument("--export", type=str, default=None, help="Eşleşmiş özet CSV yolu")
    parser.add_argument("--preview", type=int, default=15, help="Konsolda gösterilecek eşleme satırı")
    args = parser.parse_args(argv)

    root = find_project_root(args.root)
    autsl = root / "data" / "raw" / "AUTSL"
    print("=" * 72)
    print("AUTSL VERİ KONTROL RAPORU")
    print(f"Kök     : {root}")
    print(f"AUTSL   : {autsl}")
    print("=" * 72)

    if not autsl.exists():
        print("[HATA] data/raw/AUTSL bulunamadı.")
        return 2

    # --- Eksik / mevcut dosya envanteri ---
    print("\n[1] Dosya envanteri")
    required = [f"{s}.csv" for s in SPLITS] + [f"{s}/" for s in SPLITS]
    for item in required:
        p = autsl / item.rstrip("/")
        flag = "VAR" if p.exists() else "EKSIK"
        print(f"  [{flag}] {item}")

    for name in ALT_LABEL_NAMES:
        p = autsl / name
        if p.exists():
            print(f"  [VAR] {name} (alternatif etiket adı)")
        else:
            print(f"  [YOK] {name} (zorunlu değil; {name.split('_')[0]}.csv kullanılıyor olabilir)")

    signlist_path = find_signlist(autsl)
    if signlist_path:
        print(f"  [VAR] SignList: {signlist_path.name}")
    else:
        print("  [EKSIK] SignList_ClassId_TR_EN.csv  ← Türkçe kelime eşlemesi için KRİTİK")
        print("          Beklenen minimal şema: ClassId,TR,EN")

    # --- Split analizi ---
    print("\n[2] Split bütünlüğü (CSV ↔ video)")
    reports: Dict[str, SplitReport] = {}
    summary_rows = []
    all_signers: Dict[str, set] = {}
    all_classes = set()

    for split in SPLITS:
        rep = analyze_split(autsl, split)
        reports[split] = rep
        all_signers[split] = set(rep.signers)
        all_classes.update(rep.class_counts)
        status = "OK"
        if rep.csv_path is None:
            status = "CSV YOK"
        elif rep.missing_on_disk or rep.extra_on_disk or rep.n_csv_rows != rep.n_videos_disk:
            status = "UYARI"
        summary_rows.append(
            [
                split,
                rep.n_csv_rows,
                rep.n_videos_disk,
                rep.n_classes,
                len(rep.signers),
                len(rep.missing_on_disk),
                len(rep.extra_on_disk),
                status,
            ]
        )
        if rep.missing_on_disk[:5]:
            print(f"  ! {split} CSV'de olup diskte yok (örnek): {rep.missing_on_disk[:5]}")
        if rep.extra_on_disk[:5]:
            print(f"  ! {split} diskte olup CSV'de yok (örnek): {rep.extra_on_disk[:5]}")

    print_table(
        ["split", "csv_rows", "mp4_disk", "n_class", "n_signer", "missing", "extra", "status"],
        summary_rows,
    )
    total_vids = sum(r.n_videos_disk for r in reports.values())
    print(f"\nToplam MP4: {total_vids} | Benzersiz sınıf ID: {len(all_classes)}")
    if all_classes:
        print(f"Sınıf ID aralığı: {min(all_classes)} … {max(all_classes)}")

    # Signer-independent kontrol
    print("\n[3] İşaretçi (signer) bağımsızlığı")
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    ok_si = True
    for a, b in pairs:
        inter = all_signers[a] & all_signers[b]
        if inter:
            ok_si = False
            print(f"  [Sızıntı] {a} ∩ {b} = {sorted(inter)}")
        else:
            print(f"  [OK] {a} ∩ {b} = ∅")
    if ok_si:
        print("  Sonuç: Signer-independent split korunuyor.")

    for split, rep in reports.items():
        print(f"  {split} işaretçileri ({len(rep.signers)}): {', '.join(rep.signers)}")

    # --- SignList eşleme ---
    print("\n[4] ClassId → Türkçe kelime eşlemesi")
    signlist = None
    matched: List[dict] = []
    if signlist_path:
        try:
            signlist = load_signlist(signlist_path)
            print(f"  SignList yüklendi: {len(signlist)} sınıf")
            missing_ids = sorted(all_classes - set(signlist.keys()))
            extra_ids = sorted(set(signlist.keys()) - all_classes)
            if missing_ids:
                print(f"  [UYARI] CSV'de olup SignList'te yok: {missing_ids[:20]}{'...' if len(missing_ids) > 20 else ''}")
            else:
                print("  [OK] Tüm kullanılan ClassId'ler SignList'te var.")
            if extra_ids:
                print(f"  [BİLGİ] SignList'te olup veri setinde yok: {len(extra_ids)} ID")

            matched = build_matched_rows(reports, signlist)
            # Sınıf özeti
            class_summary = {}
            for row in matched:
                cid = row["class_id"]
                if cid not in class_summary:
                    class_summary[cid] = {
                        "class_id": cid,
                        "TR": row["TR"],
                        "EN": row["EN"],
                        "train": 0,
                        "val": 0,
                        "test": 0,
                    }
                class_summary[cid][row["split"]] += 1

            preview = sorted(class_summary.values(), key=lambda x: x["class_id"])[: args.preview]
            print("\n  Sınıf özeti (ilk satırlar):")
            print_table(
                ["class_id", "TR", "EN", "train", "val", "test"],
                [[c["class_id"], c["TR"], c["EN"], c["train"], c["val"], c["test"]] for c in preview],
                max_rows=args.preview,
            )

            # Video düzeyinde örnek eşleme
            print("\n  Video ↔ etiket örnekleri:")
            sample = matched[: args.preview]
            print_table(
                ["split", "filename", "class_id", "TR", "EN"],
                [[r["split"], r["filename"], r["class_id"], r["TR"], r["EN"]] for r in sample],
                max_rows=args.preview,
            )
        except Exception as exc:
            print(f"  [HATA] SignList okunamadı: {exc}")
    else:
        matched = build_matched_rows(reports, None)
        print("  SignList yok → yalnızca ClassId ile devam edilebilir.")
        print("  TR/EN sütunları boş kalacak. Dosyayı ekleyip betiği yeniden çalıştırın.")
        sample = matched[: args.preview]
        print_table(
            ["split", "filename", "class_id", "TR", "EN"],
            [[r["split"], r["filename"], r["class_id"], r["TR"], r["EN"]] for r in sample],
            max_rows=args.preview,
        )

    # --- Video örnekleme ---
    print("\n[5] Video örnekleri")
    all_mp4: List[Path] = []
    for split in SPLITS:
        d = autsl / split
        if d.exists():
            all_mp4.extend(sorted(d.glob("*.mp4")))
    if not all_mp4:
        print("  MP4 bulunamadı.")
    else:
        random.seed(42)
        examples = random.sample(all_mp4, min(args.sample_videos, len(all_mp4)))
        for p in examples:
            print(f"  - {p.parent.name}/{p.name} ({p.stat().st_size / 1e6:.2f} MB)")
        if args.probe:
            print("\n  OpenCV probe:")
            probed = probe_videos(all_mp4, limit=min(30, max(args.sample_videos, 10)))
            if probed:
                res = Counter((r["width"], r["height"]) for r in probed)
                fps = Counter(r["fps"] for r in probed)
                durs = [r["duration_sec"] for r in probed if r["duration_sec"] is not None]
                print(f"  çözünürlük: {dict(res)}")
                print(f"  fps: {dict(fps)}")
                if durs:
                    print(
                        f"  süre (sn) min/ort/max: "
                        f"{min(durs):.2f} / {sum(durs)/len(durs):.2f} / {max(durs):.2f}"
                    )
                print_table(
                    ["split", "file", "WxH", "fps", "frames", "sec", "ok"],
                    [
                        [
                            r["split"],
                            r["file"],
                            f"{r['width']}x{r['height']}",
                            r["fps"],
                            r["frames"],
                            r["duration_sec"],
                            r["readable"],
                        ]
                        for r in probed[:10]
                    ],
                )

    # --- Export ---
    outputs_dir = root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if args.export:
        export_csv(Path(args.export), matched if matched else build_matched_rows(reports, signlist))
    elif signlist_path and matched:
        export_csv(outputs_dir / "etiket_ozet.csv", matched)

    # Sınıf özet tablosu (SignList + split sayıları)
    if signlist and matched:
        class_summary: Dict[int, dict] = {}
        for row in matched:
            cid = int(row["class_id"])
            if cid not in class_summary:
                class_summary[cid] = {
                    "class_id": cid,
                    "TR": row["TR"],
                    "EN": row["EN"],
                    "train": 0,
                    "val": 0,
                    "test": 0,
                }
            class_summary[cid][row["split"]] += 1
        sinif_path = outputs_dir / "sinif_ozet.csv"
        with sinif_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["class_id", "TR", "EN", "train", "val", "test", "total"]
            )
            writer.writeheader()
            for cid in sorted(class_summary):
                c = class_summary[cid]
                writer.writerow(
                    {
                        **c,
                        "total": c["train"] + c["val"] + c["test"],
                    }
                )
        print(f"[OK] Sınıf özeti yazıldı: {sinif_path}")

        # Ön çalışma çekirdek alt kümesi (raporla hizalı 30 sınıf)
        PROTO_IDS = [
            173, 93, 196, 65, 86, 151,  # nezaket
            30, 175, 34, 14, 20, 15,    # kişi
            111, 136, 138, 137,         # soru
            209, 217, 94, 22, 76, 29, 211,  # eylem
            64, 144,                    # yer
            100, 122,                   # durum
            212, 55,                    # zaman
            38,                         # nesne
        ]
        proto_rows = []
        for cid in PROTO_IDS:
            info = signlist.get(cid, {})
            c = class_summary.get(cid)
            if c is None:
                continue
            proto_rows.append(
                {
                    "class_id": cid,
                    "TR": info.get("TR", c["TR"]),
                    "EN": info.get("EN", c["EN"]),
                    "train": c["train"],
                    "val": c["val"],
                    "test": c["test"],
                    "total": c["train"] + c["val"] + c["test"],
                }
            )
        proto_csv = outputs_dir / "prototype_classes.csv"
        with proto_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["class_id", "TR", "EN", "train", "val", "test", "total"]
            )
            writer.writeheader()
            writer.writerows(proto_rows)
        proto_txt = outputs_dir / "prototype_classes.txt"
        proto_txt.write_text("\n".join(str(c) for c in PROTO_IDS) + "\n", encoding="utf-8")
        proto_total = sum(r["total"] for r in proto_rows)
        print(f"[OK] Prototip alt küme ({len(proto_rows)} sınıf, {proto_total} video): {proto_csv}")
        print(f"[OK] Class ID listesi: {proto_txt}")

    # --- Sonuç kodu ---
    print("\n[6] Sonuç")
    errors = []
    for split, rep in reports.items():
        if rep.csv_path is None:
            errors.append(f"{split}.csv eksik")
        if rep.missing_on_disk or rep.extra_on_disk:
            errors.append(f"{split} CSV/disk uyumsuz")
    if not signlist_path:
        print("  Durum: VERİ HAZIR (iskelet/eğitim için) — SignList EKLENMELİ (TR/TTS için)")
        return 1 if not errors else 2
    if errors:
        print("  Durum: HATALAR VAR → " + "; ".join(errors))
        return 2
    print("  Durum: TAM HAZIR (CSV + videolar + SignList eşlemesi OK)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
