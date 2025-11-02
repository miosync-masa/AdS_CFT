# Morphology & geometry utilities (no SciPy)

import numpy as np
from typing import List, Tuple

def laplacian2d(a: np.ndarray) -> np.ndarray:
    return (
        np.roll(a, 1, 0) + np.roll(a, -1, 0) +
        np.roll(a, 1, 1) + np.roll(a, -1, 1) - 4*a
    )

def perimeter_len_bool8(mask: np.ndarray) -> float:
    m = mask.astype(np.int8)
    dx = np.abs(m - np.roll(m, -1, 1))
    dy = np.abs(m - np.roll(m, -1, 0))
    d1 = np.abs(m - np.roll(np.roll(m, -1, 0), -1, 1))
    d2 = np.abs(m - np.roll(np.roll(m, -1, 0),  1, 1))
    perim = dx.sum() + dy.sum() + (np.sqrt(0.5))*(d1.sum() + d2.sum())
    return float(perim)

def corner_count(mask: np.ndarray) -> int:
    A = mask.astype(np.int8)
    B = np.roll(A, -1, 0)
    C = np.roll(A, -1, 1)
    D = np.roll(np.roll(A, -1, 0), -1, 1)
    s = A + B + C + D
    return int(np.logical_or(s == 1, s == 3).sum())

def _bfs_components_periodic(mask: np.ndarray) -> List[List[Tuple[int,int]]]:
    H, W = mask.shape
    vis = np.zeros_like(mask, dtype=bool)
    comps = []
    for i in range(H):
        for j in range(W):
            if not mask[i,j] or vis[i,j]:
                continue
            q = [(i,j)]; vis[i,j]=True; comp=[(i,j)]
            while q:
                ci,cj = q.pop()
                for di,dj in ((1,0),(-1,0),(0,1),(0,-1)):
                    ni, nj = (ci+di)%H, (cj+dj)%W
                    if mask[ni,nj] and not vis[ni,nj]:
                        vis[ni,nj]=True
                        q.append((ni,nj)); comp.append((ni,nj))
            comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps

def k_th_largest_mask(mask: np.ndarray, k: int) -> np.ndarray:
    comps = _bfs_components_periodic(mask)
    out = np.zeros_like(mask, dtype=bool)
    if len(comps) > k and len(comps[k])>0:
        for i,j in comps[k]:
            out[i,j]=True
    elif len(comps)>0:
        for i,j in comps[0]:
            out[i,j]=True
    return out

def dilate4(mask: np.ndarray, steps:int=1) -> np.ndarray:
    H, W = mask.shape
    m = mask.copy()
    for _ in range(steps):
        nb = (np.roll(m,1,0) | np.roll(m,-1,0) | np.roll(m,1,1) | np.roll(m,-1,1))
        m = (m | nb)
    return m

def erode4(mask: np.ndarray, steps:int=1) -> np.ndarray:
    inv = np.logical_not(mask)
    inv_d = dilate4(inv, steps)
    return np.logical_not(inv_d)

def inner_boundary_band(mask: np.ndarray, width:int=1) -> np.ndarray:
    er = erode4(mask, steps=width)
    return (mask & (~er))

def outer_boundary_band(mask: np.ndarray, width:int=1) -> np.ndarray:
    dl = dilate4(mask, steps=width)
    return (dl & (~mask))

def count_holes_nonperiodic(mask: np.ndarray) -> int:
    H,W = mask.shape
    inv = (~mask).astype(np.uint8)
    vis = np.zeros_like(inv, dtype=bool)
    from collections import deque
    q = deque()
    for i in range(H):
        if inv[i,0]: q.append((i,0)); vis[i,0]=True
        if inv[i,W-1]: q.append((i,W-1)); vis[i,W-1]=True
    for j in range(W):
        if inv[0,j]: q.append((0,j)); vis[0,j]=True
        if inv[H-1,j]: q.append((H-1,j)); vis[H-1,j]=True
    while q:
        ci,cj = q.popleft()
        for di,dj in ((1,0),(-1,0),(0,1),(0,-1)):
            ni, nj = ci+di, cj+dj
            if 0<=ni<H and 0<=nj<W and inv[ni,nj] and not vis[ni,nj]:
                vis[ni,nj]=True; q.append((ni,nj))
    holes = 0
    for i in range(H):
        for j in range(W):
            if inv[i,j] and not vis[i,j]:
                holes += 1
                q = deque([(i,j)]); vis[i,j]=True
                while q:
                    ci,cj = q.popleft()
                    for di,dj in ((1,0),(-1,0),(0,1),(0,-1)):
                        ni, nj = ci+di, cj+dj
                        if 0<=ni<H and 0<=nj<W and inv[ni,nj] and not vis[ni,nj]:
                            vis[ni,nj]=True; q.append((ni,nj))
    return holes
