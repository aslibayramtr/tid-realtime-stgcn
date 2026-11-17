"""
46 dugumlu iskelet normalizasyonu ve sabit T pencere ornekleme.

Islem sirasi:
  1. Omuz orta noktasina gore merkezleme
  2. Omuz genisligine bolme
  3. Gorunmeyen (None / maskeli) elleri 0.0 ile maskeleme
  4. Zamansal yeniden ornekleme T=64
  5. (C, T, V, M) = (4, 64, 46, 1) float32 tensore cevirme
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from src.graph.topology_46 import (
    LEFT_ELBOW,
    LEFT_SHOULDER,
    NUM_NODES,
    RIGHT_ELBOW,
    RIGHT_SHOULDER,
)

TARGET_T = 64
NUM_CHANNELS = 4  # x, y, z, confidence
NUM_PERSONS = 1

LEFT_HAND_SLICE = slice(0, 21)
RIGHT_HAND_SLICE = slice(21, 42)

# Gorunmeyen el maskelenirken sifirlanacak dugumler (yalniz ilgili el)
LEFT_MASK_NODES = tuple(range(0, 21))
RIGHT_MASK_NODES = tuple(range(21, 42))

_EPS = 1e-6


def _as_float32(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float32)


def shoulder_center_and_width(
    frame_xyz: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Tek kare (V, 3+) icin omuz orta noktasi ve omuz genisligi.
    frame_xyz: (46, C) C>=3
    """
    ls = frame_xyz[LEFT_SHOULDER, :3]
    rs = frame_xyz[RIGHT_SHOULDER, :3]
    center = 0.5 * (ls + rs)
    width = float(np.linalg.norm(rs - ls))
    if width < _EPS:
        width = 1.0
    return center.astype(np.float32), width


def normalize_frame(
    frame_xyz: np.ndarray,
    left_visible: bool = True,
    right_visible: bool = True,
) -> np.ndarray:
    """
    Tek kareyi merkezle + olcekle; gorunmeyen elleri 0.0 yap.
    frame_xyz: (46, 3) veya (46, 4)
    Donus: (46, 4) float32 — C yoksa confidence 1.0 (maskelenenlerde 0.0)
    """
    frame = _as_float32(frame_xyz)
    if frame.shape[0] != NUM_NODES:
        raise ValueError(f"V={NUM_NODES} bekleniyor, gelen={frame.shape[0]}")
    xyz = frame[:, :3].copy()
    if frame.shape[1] >= 4:
        conf = frame[:, 3].copy()
    else:
        conf = np.ones(NUM_NODES, dtype=np.float32)

    center, width = shoulder_center_and_width(xyz)
    xyz = (xyz - center) / width

    if not left_visible:
        xyz[list(LEFT_MASK_NODES)] = 0.0
        conf[list(LEFT_MASK_NODES)] = 0.0
    if not right_visible:
        xyz[list(RIGHT_MASK_NODES)] = 0.0
        conf[list(RIGHT_MASK_NODES)] = 0.0

    out = np.concatenate([xyz, conf[:, None]], axis=1)
    return out.astype(np.float32)


def resample_temporal(seq: np.ndarray, target_t: int = TARGET_T) -> np.ndarray:
    """
    (T_in, V, C) dizisini lineer indeks ornekleme ile (target_t, V, C) yapar.
    T_in == 0 ise sifir dizi doner.
    T_in == 1 ise kareyi tekrarlar.
    """
    seq = _as_float32(seq)
    if seq.ndim != 3:
        raise ValueError(f"(T, V, C) bekleniyor, shape={seq.shape}")
    t_in, v, c = seq.shape
    if target_t <= 0:
        raise ValueError(f"target_t > 0 olmali, gelen={target_t}")
    if t_in == 0:
        return np.zeros((target_t, v, c), dtype=np.float32)
    if t_in == target_t:
        return seq.copy()
    if t_in == 1:
        return np.repeat(seq, target_t, axis=0)

    # [0, t_in-1] araliginda esit aralikli indeksler
    src_idx = np.linspace(0, t_in - 1, num=target_t)
    low = np.floor(src_idx).astype(np.int64)
    high = np.minimum(low + 1, t_in - 1)
    w = (src_idx - low).astype(np.float32)
    # Yayin: (T, 1, 1) * (T, V, C)
    w = w[:, None, None]
    out = (1.0 - w) * seq[low] + w * seq[high]
    return out.astype(np.float32)


def to_ctvm(seq_tvc: np.ndarray) -> np.ndarray:
    """
    (T, V, C) -> (C, T, V, M=1) float32.
    """
    seq = _as_float32(seq_tvc)
    if seq.ndim != 3:
        raise ValueError(f"(T, V, C) bekleniyor, shape={seq.shape}")
    # (T, V, C) -> (C, T, V) -> (C, T, V, 1)
    ctvm = np.transpose(seq, (2, 0, 1))[..., None]
    return ctvm.astype(np.float32)


def normalize_sequence(
    keypoints: np.ndarray,
    left_visible: Optional[np.ndarray] = None,
    right_visible: Optional[np.ndarray] = None,
    target_t: int = TARGET_T,
) -> np.ndarray:
    """
    Ham kare dizisini rapor standardina cevirir.

    Parametreler
    ----------
    keypoints : (T, 46, 3) veya (T, 46, 4) float
        Ham MediaPipe koordinatlari (normalize edilmemis).
    left_visible, right_visible : (T,) bool/int, opsiyonel
        False/0 ise ilgili el normalizasyon SONRASI sifirlanir.
        None ise confidence>0 veya NaN olmayan koordinata gore tahmin edilir.
    target_t : int
        Sabit pencere uzunlugu (varsayilan 64).

    Donus
    -----
    np.ndarray, shape (4, target_t, 46, 1), dtype float32
    """
    kp = _as_float32(keypoints)
    if kp.ndim != 3 or kp.shape[1] != NUM_NODES or kp.shape[2] not in (3, 4):
        raise ValueError(
            f"keypoints (T, {NUM_NODES}, 3|4) olmali, gelen={kp.shape}"
        )
    t_in = kp.shape[0]

    if left_visible is None:
        # Sol el bilek guveni / NaN
        left_visible = _infer_hand_visibility(kp, LEFT_HAND_SLICE)
    else:
        left_visible = np.asarray(left_visible, dtype=bool).reshape(t_in)

    if right_visible is None:
        right_visible = _infer_hand_visibility(kp, RIGHT_HAND_SLICE)
    else:
        right_visible = np.asarray(right_visible, dtype=bool).reshape(t_in)

    frames = []
    for t in range(t_in):
        frames.append(
            normalize_frame(
                kp[t],
                left_visible=bool(left_visible[t]),
                right_visible=bool(right_visible[t]),
            )
        )
    seq = np.stack(frames, axis=0)  # (T, 46, 4)
    seq = resample_temporal(seq, target_t=target_t)
    return to_ctvm(seq)


def _infer_hand_visibility(kp: np.ndarray, hand_slice: slice) -> np.ndarray:
    """(T,) bool — el noktalarinda NaN yoksa ve (varsa) conf ortalamasi > 0 ise True."""
    hand = kp[:, hand_slice, :]
    finite = np.isfinite(hand[..., :3]).all(axis=(1, 2))
    if hand.shape[-1] >= 4:
        conf_ok = np.nanmean(hand[..., 3], axis=1) > _EPS
        return finite & conf_ok
    # NaN yoksa ve tum sifir degilse gorunur say
    nonzero = np.any(np.abs(hand[..., :3]) > _EPS, axis=(1, 2))
    return finite & nonzero


def pack_raw_frame(
    left_hand: Optional[Sequence[Sequence[float]]],
    right_hand: Optional[Sequence[Sequence[float]]],
    left_shoulder: Sequence[float],
    right_shoulder: Sequence[float],
    left_elbow: Optional[Sequence[float]],
    right_elbow: Optional[Sequence[float]],
) -> Tuple[np.ndarray, bool, bool]:
    """
    MediaPipe Holistic ciktisindan tek kare (46, 4) ve el gorunurlugu uretir.
    El None ise ilgili 21 nokta NaN + conf=0; normalize_sequence maskeler.
    Omuz zorunlu; dirsek None ise omuz kopyasi / sifir conf.
    """
    frame = np.zeros((NUM_NODES, NUM_CHANNELS), dtype=np.float32)

    def _fill_points(start: int, points: Sequence[Sequence[float]], default_conf: float = 1.0) -> None:
        for i, p in enumerate(points):
            xyz = [float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0]
            conf = float(p[3]) if len(p) > 3 else default_conf
            frame[start + i, 0:3] = xyz
            frame[start + i, 3] = conf

    left_vis = left_hand is not None
    right_vis = right_hand is not None

    if left_vis:
        _fill_points(0, left_hand)  # type: ignore[arg-type]
    else:
        frame[LEFT_HAND_SLICE, :] = np.nan

    if right_vis:
        _fill_points(21, right_hand)  # type: ignore[arg-type]
    else:
        frame[RIGHT_HAND_SLICE, :] = np.nan

    frame[LEFT_SHOULDER, :3] = np.asarray(left_shoulder[:3], dtype=np.float32)
    frame[LEFT_SHOULDER, 3] = float(left_shoulder[3]) if len(left_shoulder) > 3 else 1.0
    frame[RIGHT_SHOULDER, :3] = np.asarray(right_shoulder[:3], dtype=np.float32)
    frame[RIGHT_SHOULDER, 3] = float(right_shoulder[3]) if len(right_shoulder) > 3 else 1.0

    if left_elbow is not None:
        frame[LEFT_ELBOW, :3] = np.asarray(left_elbow[:3], dtype=np.float32)
        frame[LEFT_ELBOW, 3] = float(left_elbow[3]) if len(left_elbow) > 3 else 1.0
    else:
        frame[LEFT_ELBOW, 3] = 0.0

    if right_elbow is not None:
        frame[RIGHT_ELBOW, :3] = np.asarray(right_elbow[:3], dtype=np.float32)
        frame[RIGHT_ELBOW, 3] = float(right_elbow[3]) if len(right_elbow) > 3 else 1.0
    else:
        frame[RIGHT_ELBOW, 3] = 0.0

    return frame, left_vis, right_vis


__all__ = [
    "TARGET_T",
    "NUM_CHANNELS",
    "NUM_PERSONS",
    "LEFT_HAND_SLICE",
    "RIGHT_HAND_SLICE",
    "normalize_frame",
    "normalize_sequence",
    "resample_temporal",
    "to_ctvm",
    "pack_raw_frame",
    "shoulder_center_and_width",
]
