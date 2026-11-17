#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MediaPipe Holistic -> 46 dugum keypoint cikarimi (prototip 30 sinif).

Kullanim:
    python scripts/02_extract_keypoints.py
    python scripts/02_extract_keypoints.py --limit 50
    python scripts/02_extract_keypoints.py --splits train val --limit 10

Cikti:
    data/processed/keypoints_v46/{split}/{stem}.npy   # (4, 64, 46, 1) float32
    outputs/logs/extract_report.md
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_prep.normalizer import normalize_sequence, pack_raw_frame  # noqa: E402
from src.graph.topology_46 import NUM_NODES  # noqa: E402

SPLITS = ("train", "val", "test")

# MediaPipe Pose landmark indeksleri
_POSE_L_SHOULDER = 11
_POSE_R_SHOULDER = 12
_POSE_L_ELBOW = 13
_POSE_R_ELBOW = 14


@dataclass
class ExtractStats:
    processed: int = 0
    skipped: int = 0
    skip_reasons: Dict[str, int] = field(default_factory=dict)
    total_frames: int = 0
    frames_missing_left: int = 0
    frames_missing_right: int = 0
    frames_missing_either: int = 0
    per_split: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def bump_skip(self, reason: str) -> None:
        self.skipped += 1
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1

    def ensure_split(self, split: str) -> None:
        if split not in self.per_split:
            self.per_split[split] = {"processed": 0, "skipped": 0}


def find_project_root(explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    here = Path(__file__).resolve().parent.parent
    if (here / "data" / "raw" / "AUTSL").exists():
        return here
    cwd = Path.cwd().resolve()
    if (cwd / "data" / "raw" / "AUTSL").exists():
        return cwd
    return here


def load_prototype_ids(path: Path) -> List[int]:
    ids: List[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        ids.append(int(s.split(",")[0].strip()))
    if not ids:
        raise ValueError(f"Prototip sinif listesi bos: {path}")
    return ids


def read_split_csv(path: Path) -> List[Tuple[str, int]]:
    rows: List[Tuple[str, int]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for i, parts in enumerate(reader):
            if not parts or all(not str(p).strip() for p in parts):
                continue
            if len(parts) < 2:
                continue
            fn = str(parts[0]).strip()
            cid_raw = str(parts[1]).strip()
            if i == 0 and not cid_raw.lstrip("-").isdigit():
                continue
            rows.append((fn, int(float(cid_raw))))
    return rows


def filter_rows(
    rows: Sequence[Tuple[str, int]],
    class_ids: Set[int],
    limit: Optional[int],
) -> List[Tuple[str, int]]:
    filtered = [(fn, cid) for fn, cid in rows if cid in class_ids]
    if limit is not None and limit >= 0:
        filtered = filtered[:limit]
    return filtered


def _lm_to_xyzc(lm, visibility: Optional[float] = None) -> List[float]:
    conf = float(visibility) if visibility is not None else float(getattr(lm, "visibility", 1.0) or 1.0)
    return [float(lm.x), float(lm.y), float(lm.z), conf]


def extract_video_keypoints(
    video_path: Path,
    holistic,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], str]:
    """
    Video -> ham (T, 46, 4), left_vis (T,), right_vis (T,).
    Basarisizsa (None, None, None, reason).
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None, None, "video_acilamadi"

    frames_raw: List[np.ndarray] = []
    left_flags: List[bool] = []
    right_flags: List[bool] = []

    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            if results.pose_landmarks is None:
                # Omuz yoksa kareyi atla (normalize edilemez)
                continue

            pose = results.pose_landmarks.landmark
            ls = _lm_to_xyzc(pose[_POSE_L_SHOULDER], pose[_POSE_L_SHOULDER].visibility)
            rs = _lm_to_xyzc(pose[_POSE_R_SHOULDER], pose[_POSE_R_SHOULDER].visibility)
            le = _lm_to_xyzc(pose[_POSE_L_ELBOW], pose[_POSE_L_ELBOW].visibility)
            re = _lm_to_xyzc(pose[_POSE_R_ELBOW], pose[_POSE_R_ELBOW].visibility)

            left_hand = None
            if results.left_hand_landmarks is not None:
                left_hand = [
                    _lm_to_xyzc(p, getattr(p, "visibility", 1.0))
                    for p in results.left_hand_landmarks.landmark
                ]

            right_hand = None
            if results.right_hand_landmarks is not None:
                right_hand = [
                    _lm_to_xyzc(p, getattr(p, "visibility", 1.0))
                    for p in results.right_hand_landmarks.landmark
                ]

            frame, l_vis, r_vis = pack_raw_frame(
                left_hand=left_hand,
                right_hand=right_hand,
                left_shoulder=ls,
                right_shoulder=rs,
                left_elbow=le,
                right_elbow=re,
            )
            frames_raw.append(frame)
            left_flags.append(l_vis)
            right_flags.append(r_vis)
    finally:
        cap.release()

    if not frames_raw:
        return None, None, None, "pose_yok_veya_bos"

    kp = np.stack(frames_raw, axis=0)
    # pack_raw_frame NaN birakti; normalize oncesi NaN -> 0 (maske zaten visibility ile)
    kp = np.nan_to_num(kp, nan=0.0).astype(np.float32)
    return kp, np.asarray(left_flags, dtype=bool), np.asarray(right_flags, dtype=bool), "ok"


def write_report(path: Path, stats: ExtractStats, limit: Optional[int], class_ids: Sequence[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    miss_l = stats.frames_missing_left / stats.total_frames if stats.total_frames else 0.0
    miss_r = stats.frames_missing_right / stats.total_frames if stats.total_frames else 0.0
    miss_e = stats.frames_missing_either / stats.total_frames if stats.total_frames else 0.0

    lines = [
        "# Keypoint Cikarim Raporu",
        "",
        f"- Prototip sinif sayisi: {len(class_ids)}",
        f"- Limit: {limit if limit is not None else 'yok (tum filtrelenmis videolar)'}",
        f"- Islenen video: **{stats.processed}**",
        f"- Elenen video: **{stats.skipped}**",
        f"- Toplam islenen kare: {stats.total_frames}",
        "",
        "## Eksik el oranlari (kare duzeyinde)",
        "",
        f"- Sol el eksik orani: {miss_l:.4f} ({stats.frames_missing_left}/{stats.total_frames})",
        f"- Sag el eksik orani: {miss_r:.4f} ({stats.frames_missing_right}/{stats.total_frames})",
        f"- En az bir el eksik orani: {miss_e:.4f} ({stats.frames_missing_either}/{stats.total_frames})",
        "",
        "## Split ozeti",
        "",
        "| split | islenen | elenen |",
        "|---|---:|---:|",
    ]
    for split in SPLITS:
        d = stats.per_split.get(split, {"processed": 0, "skipped": 0})
        lines.append(f"| {split} | {d['processed']} | {d['skipped']} |")

    if stats.skip_reasons:
        lines.extend(["", "## Eleme nedenleri", ""])
        for reason, n in sorted(stats.skip_reasons.items(), key=lambda x: -x[1]):
            lines.append(f"- `{reason}`: {n}")

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="AUTSL 46-dugum keypoint cikarimi")
    parser.add_argument("--root", type=str, default=None, help="Proje kok dizini")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Her split icin islenecek maksimum video sayisi",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(SPLITS),
        choices=list(SPLITS),
        help="Islenecek split'ler",
    )
    parser.add_argument(
        "--proto",
        type=str,
        default=None,
        help="prototype_classes.txt yolu (varsayilan: outputs/prototype_classes.txt)",
    )
    args = parser.parse_args(argv)

    root = find_project_root(args.root)
    autsl = root / "data" / "raw" / "AUTSL"
    out_root = root / "data" / "processed" / "keypoints_v46"
    proto_path = Path(args.proto) if args.proto else root / "outputs" / "prototype_classes.txt"
    report_path = root / "outputs" / "logs" / "extract_report.md"

    if not autsl.exists():
        print(f"[HATA] AUTSL bulunamadi: {autsl}")
        return 2
    if not proto_path.exists():
        print(f"[HATA] Prototip listesi yok: {proto_path}")
        return 2

    class_ids = load_prototype_ids(proto_path)
    class_set = set(class_ids)
    print(f"Prototip siniflar ({len(class_ids)}): {class_ids}")
    print(f"Cikti kok: {out_root}")
    print(f"Limit: {args.limit}")

    try:
        import mediapipe as mp
    except ImportError:
        print("[HATA] mediapipe yuklu degil. .venv icinde: pip install mediapipe")
        return 2

    mp_holistic = mp.solutions.holistic
    stats = ExtractStats()

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        for split in args.splits:
            stats.ensure_split(split)
            csv_path = autsl / f"{split}.csv"
            video_dir = autsl / split
            split_out = out_root / split
            split_out.mkdir(parents=True, exist_ok=True)

            if not csv_path.exists():
                print(f"[UYARI] {csv_path} yok, atlandi")
                continue

            rows = filter_rows(read_split_csv(csv_path), class_set, args.limit)
            print(f"\n=== {split}: {len(rows)} video ===")

            for idx, (fn, cid) in enumerate(rows, start=1):
                video_path = video_dir / fn
                stem = Path(fn).stem
                out_path = split_out / f"{stem}.npy"

                if not video_path.exists():
                    stats.bump_skip("dosya_yok")
                    stats.per_split[split]["skipped"] += 1
                    print(f"  [{idx}/{len(rows)}] ATLA dosya_yok: {fn}")
                    continue

                kp, l_vis, r_vis, reason = extract_video_keypoints(video_path, holistic)
                if kp is None or l_vis is None or r_vis is None:
                    stats.bump_skip(reason)
                    stats.per_split[split]["skipped"] += 1
                    print(f"  [{idx}/{len(rows)}] ATLA {reason}: {fn}")
                    continue

                try:
                    tensor = normalize_sequence(kp, left_visible=l_vis, right_visible=r_vis)
                except Exception as exc:
                    stats.bump_skip(f"normalize_hata:{type(exc).__name__}")
                    stats.per_split[split]["skipped"] += 1
                    print(f"  [{idx}/{len(rows)}] ATLA normalize: {fn} ({exc})")
                    continue

                expected = (4, 64, NUM_NODES, 1)
                if tensor.shape != expected or tensor.dtype != np.float32:
                    stats.bump_skip("sekil_hatali")
                    stats.per_split[split]["skipped"] += 1
                    print(f"  [{idx}/{len(rows)}] ATLA sekil={tensor.shape}: {fn}")
                    continue

                np.save(out_path, tensor)

                t_frames = int(l_vis.shape[0])
                stats.total_frames += t_frames
                miss_l = int(np.sum(~l_vis))
                miss_r = int(np.sum(~r_vis))
                stats.frames_missing_left += miss_l
                stats.frames_missing_right += miss_r
                stats.frames_missing_either += int(np.sum(~(l_vis & r_vis)))
                stats.processed += 1
                stats.per_split[split]["processed"] += 1

                if idx % 10 == 0 or idx == len(rows):
                    print(
                        f"  [{idx}/{len(rows)}] OK {stem}.npy  "
                        f"cid={cid} missL={miss_l}/{t_frames} missR={miss_r}/{t_frames}"
                    )

    write_report(report_path, stats, args.limit, class_ids)
    print("\n=== OZET ===")
    print(f"Islenen: {stats.processed} | Elenen: {stats.skipped}")
    if stats.total_frames:
        print(
            f"Eksik el (en az biri): "
            f"{stats.frames_missing_either / stats.total_frames:.4f}"
        )
    print(f"Rapor: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
