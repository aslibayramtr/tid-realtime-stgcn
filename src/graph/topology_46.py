"""
46 dugumlu anatomik ST-GCN topolojisi (MediaPipe Holistic budama).

Dugum sirasi:
  0-20  Sol el (MediaPipe left_hand 0-20)
  21-41 Sag el (MediaPipe right_hand 0-20, ofset +21)
  42    Sol omuz  (pose 11)
  43    Sag omuz  (pose 12)
  44    Sol dirsek (pose 13)
  45    Sag dirsek (pose 14)

Kenarlar: el ici kemik + ardısik parmak ucu + bilek-dirsek-omuz.
Normalize: A_hat = D_tilde^{-1/2} (A + I) D_tilde^{-1/2}
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

NUM_NODES = 46

# Sol / sag el lokal indeksleri (MediaPipe Hand)
_LEFT_OFFSET = 0
_RIGHT_OFFSET = 21

LEFT_WRIST = 0
RIGHT_WRIST = 21
LEFT_SHOULDER = 42
RIGHT_SHOULDER = 43
LEFT_ELBOW = 44
RIGHT_ELBOW = 45

# MediaPipe Hand kemik kenarlari (lokal 0-20)
_HAND_BONE_EDGES: Tuple[Tuple[int, int], ...] = (
    # Basparmak
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    # Isaret
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    # Orta
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    # Yuzuk
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    # Serce
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    # Avuc (MCP zinciri)
    (5, 9),
    (9, 13),
    (13, 17),
)

# Ardısik parmak ucu kenarlari (tip): 4-8-12-16-20
_FINGERTIP_EDGES: Tuple[Tuple[int, int], ...] = (
    (4, 8),
    (8, 12),
    (12, 16),
    (16, 20),
)

# Bilek-dirsek-omuz ve omuz koprusu (global indeksler)
_ARM_EDGES: Tuple[Tuple[int, int], ...] = (
    (LEFT_WRIST, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_SHOULDER),
    (RIGHT_WRIST, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_SHOULDER),
    (LEFT_SHOULDER, RIGHT_SHOULDER),
)

NODE_NAMES: Tuple[str, ...] = tuple(
    [f"L_hand_{i}" for i in range(21)]
    + [f"R_hand_{i}" for i in range(21)]
    + ["L_shoulder", "R_shoulder", "L_elbow", "R_elbow"]
)


def _offset_edges(
    edges: Sequence[Tuple[int, int]], offset: int
) -> List[Tuple[int, int]]:
    return [(u + offset, v + offset) for u, v in edges]


def build_edge_list() -> List[Tuple[int, int]]:
    """Yonlendirilmemis kenar listesi (u, v) — her kenar bir kez."""
    edges: List[Tuple[int, int]] = []
    edges.extend(_offset_edges(_HAND_BONE_EDGES, _LEFT_OFFSET))
    edges.extend(_offset_edges(_FINGERTIP_EDGES, _LEFT_OFFSET))
    edges.extend(_offset_edges(_HAND_BONE_EDGES, _RIGHT_OFFSET))
    edges.extend(_offset_edges(_FINGERTIP_EDGES, _RIGHT_OFFSET))
    edges.extend(_ARM_EDGES)
    return edges


def build_adjacency(num_nodes: int = NUM_NODES) -> np.ndarray:
    """
    Anatomik komşuluk matrisi A (num_nodes x num_nodes), float32, simetrik, diyagonal 0.
    """
    if num_nodes != NUM_NODES:
        raise ValueError(f"Bu topoloji yalnizca V={NUM_NODES} destekler, gelen={num_nodes}")
    a = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for u, v in build_edge_list():
        if not (0 <= u < num_nodes and 0 <= v < num_nodes):
            raise IndexError(f"Kenar ({u}, {v}) V={num_nodes} disinda")
        a[u, v] = 1.0
        a[v, u] = 1.0
    return a


def normalize_adjacency(a: np.ndarray) -> np.ndarray:
    """
    ST-GCN stilinde derece-normalize komsuluk:
        A_hat = D_tilde^{-1/2} (A + I) D_tilde^{-1/2}
    """
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"A kare matris olmali, shape={a.shape}")
    a = a.astype(np.float32, copy=False)
    a_tilde = a + np.eye(a.shape[0], dtype=np.float32)
    degree = np.sum(a_tilde, axis=1)
    # Sifir derece (izolasyon) icin guvenli payda
    d_inv_sqrt = np.zeros_like(degree, dtype=np.float32)
    nonzero = degree > 0
    d_inv_sqrt[nonzero] = 1.0 / np.sqrt(degree[nonzero])
    d_mat = np.diag(d_inv_sqrt)
    a_hat = d_mat @ a_tilde @ d_mat
    return a_hat.astype(np.float32)


def get_normalized_adjacency(num_nodes: int = NUM_NODES) -> np.ndarray:
    """A olusturup Â dondurur."""
    return normalize_adjacency(build_adjacency(num_nodes))


__all__ = [
    "NUM_NODES",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "NODE_NAMES",
    "build_edge_list",
    "build_adjacency",
    "normalize_adjacency",
    "get_normalized_adjacency",
]
