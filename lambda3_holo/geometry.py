"""
Morphology & Geometry Utilities (no SciPy)
==========================================

Pure NumPy implementations of spatial operations for holographic lattice systems.

Functions:
- laplacian2d: Discrete Laplacian operator (5-point stencil)
- perimeter_len_bool8: Boundary length with diagonal contribution
- corner_count: Discrete curvature proxy via 2×2 neighborhoods
- dilate4, erode4: Morphological operations (4-connectivity)
- inner_boundary_band, outer_boundary_band: Extract boundary layers
- k_th_largest_mask: Select k-th largest connected component
- count_holes_nonperiodic: Euler characteristic proxy (non-periodic)

Physical Interpretation:
- Used for RT entropy calculation: S_RT ∝ perimeter + holes + curvature
- Implements discrete minimal surface detection on lattice
- Non-periodic hole counting treats grid edges as exterior boundary
"""

import numpy as np
from typing import List, Tuple
from collections import deque


def laplacian2d(a: np.ndarray) -> np.ndarray:
    """
    Compute discrete 2D Laplacian using 5-point stencil with periodic BC.
    
    ∇²f ≈ (f[i+1,j] + f[i-1,j] + f[i,j+1] + f[i,j-1] - 4f[i,j])
    
    Args:
        a: 2D array (H, W)
    
    Returns:
        Laplacian field (same shape as input)
    
    Physical Interpretation:
        Measures local deviation from neighborhood average.
        Used in diffusion equations: ∂f/∂t = D·∇²f
    """
    return (
        np.roll(a, 1, 0) + np.roll(a, -1, 0) +
        np.roll(a, 1, 1) + np.roll(a, -1, 1) - 4 * a
    )


def perimeter_len_bool8(mask: np.ndarray) -> float:
    """
    Compute perimeter length of binary mask including diagonal contribution.
    
    Length calculation:
    - Horizontal/vertical edges: weight = 1.0
    - Diagonal edges: weight = √2 ≈ 1.414
    
    Args:
        mask: Boolean 2D array
    
    Returns:
        Perimeter length (float)
    
    Physical Interpretation:
        Approximates continuous boundary length in discrete lattice.
        Standard term in Ryu-Takayanagi entropy: S_RT ∝ Area/G_N
    """
    m = mask.astype(np.int8)
    
    # Horizontal and vertical edges
    dx = np.abs(m - np.roll(m, -1, 1))
    dy = np.abs(m - np.roll(m, -1, 0))
    
    # Diagonal edges
    d1 = np.abs(m - np.roll(np.roll(m, -1, 0), -1, 1))
    d2 = np.abs(m - np.roll(np.roll(m, -1, 0), 1, 1))
    
    perim = dx.sum() + dy.sum() + np.sqrt(0.5) * (d1.sum() + d2.sum())
    return float(perim)


def corner_count(mask: np.ndarray) -> int:
    """
    Count corners in binary mask (discrete curvature proxy).
    
    A corner is detected when a 2×2 neighborhood has sum = 1 or 3:
    - sum = 1: Convex corner (isolated pixel)
    - sum = 3: Concave corner (missing pixel)
    
    Args:
        mask: Boolean 2D array
    
    Returns:
        Number of corners (int)
    
    Physical Interpretation:
        Discrete approximation of integrated curvature ∫κ ds.
        Captures geometric frustration of boundary.
        Higher corner count → more complex boundary topology.
    """
    A = mask.astype(np.int8)
    B = np.roll(A, -1, 0)
    C = np.roll(A, -1, 1)
    D = np.roll(np.roll(A, -1, 0), -1, 1)
    s = A + B + C + D
    return int(np.logical_or(s == 1, s == 3).sum())


def _bfs_components_periodic(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    """
    Find connected components via BFS with periodic boundary conditions.
    
    Uses 4-connectivity (von Neumann neighborhood).
    Components are sorted by size (largest first).
    
    Args:
        mask: Boolean 2D array
    
    Returns:
        List of components, each component is list of (i, j) coordinates
    
    Note:
        Periodic BC means grid wraps around like a torus.
        Used for finding largest coherent regions in cooperation field.
    """
    H, W = mask.shape
    vis = np.zeros_like(mask, dtype=bool)
    comps = []
    
    for i in range(H):
        for j in range(W):
            if not mask[i, j] or vis[i, j]:
                continue
            
            # BFS from (i, j)
            q = [(i, j)]
            vis[i, j] = True
            comp = [(i, j)]
            
            while q:
                ci, cj = q.pop()
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ni, nj = (ci + di) % H, (cj + dj) % W  # Periodic wrap
                    if mask[ni, nj] and not vis[ni, nj]:
                        vis[ni, nj] = True
                        q.append((ni, nj))
                        comp.append((ni, nj))
            
            comps.append(comp)
    
    # Sort by size (largest first)
    comps.sort(key=len, reverse=True)
    return comps


def k_th_largest_mask(mask: np.ndarray, k: int) -> np.ndarray:
    """
    Extract k-th largest connected component from binary mask.
    
    Args:
        mask: Boolean 2D array
        k: Component rank (0 = largest, 1 = second largest, etc.)
    
    Returns:
        Boolean mask containing only k-th largest component
        (or largest if k >= num_components)
    
    Physical Interpretation:
        Used to define region A in RT entropy calculation.
        Typically alternates between k=0 and k=1 to avoid
        boundary locking in dynamical systems.
    """
    comps = _bfs_components_periodic(mask)
    out = np.zeros_like(mask, dtype=bool)
    
    if len(comps) > k and len(comps[k]) > 0:
        # Use k-th component
        for i, j in comps[k]:
            out[i, j] = True
    elif len(comps) > 0:
        # Fallback to largest if k out of range
        for i, j in comps[0]:
            out[i, j] = True
    
    return out


def dilate4(mask: np.ndarray, steps: int = 1) -> np.ndarray:
    """
    Morphological dilation with 4-connectivity (von Neumann neighborhood).
    
    Expands True regions by including neighbors.
    Repeated application expands by Manhattan distance.
    
    Args:
        mask: Boolean 2D array
        steps: Number of dilation iterations
    
    Returns:
        Dilated mask (same shape as input)
    
    Physical Interpretation:
        Grows regions by including immediate neighbors.
        Used to extract boundary bands around regions.
    """
    H, W = mask.shape
    m = mask.copy()
    
    for _ in range(steps):
        nb = (
            np.roll(m, 1, 0) | np.roll(m, -1, 0) |
            np.roll(m, 1, 1) | np.roll(m, -1, 1)
        )
        m = (m | nb)
    
    return m


def erode4(mask: np.ndarray, steps: int = 1) -> np.ndarray:
    """
    Morphological erosion with 4-connectivity.
    
    Shrinks True regions by removing boundary pixels.
    Implemented as dilation of complement.
    
    Args:
        mask: Boolean 2D array
        steps: Number of erosion iterations
    
    Returns:
        Eroded mask (same shape as input)
    
    Physical Interpretation:
        Removes thin protrusions and isolated pixels.
        Used to extract inner boundary bands.
    """
    inv = np.logical_not(mask)
    inv_d = dilate4(inv, steps)
    return np.logical_not(inv_d)


def inner_boundary_band(mask: np.ndarray, width: int = 1) -> np.ndarray:
    """
    Extract inner boundary band of given width.
    
    Returns pixels that are:
    - Inside the mask
    - Within 'width' pixels of the boundary
    
    Args:
        mask: Boolean 2D array (region of interest)
        width: Band thickness in pixels
    
    Returns:
        Boolean mask of inner boundary band
    
    Physical Interpretation:
        Used to measure λ statistics near RT minimal surface.
        Inner band captures bulk-side quantum information.
    """
    er = erode4(mask, steps=width)
    return (mask & (~er))


def outer_boundary_band(mask: np.ndarray, width: int = 1) -> np.ndarray:
    """
    Extract outer boundary band of given width.
    
    Returns pixels that are:
    - Outside the mask
    - Within 'width' pixels of the boundary
    
    Args:
        mask: Boolean 2D array (region of interest)
        width: Band thickness in pixels
    
    Returns:
        Boolean mask of outer boundary band
    
    Physical Interpretation:
        Critical for geodesic gating: monitors λ_p99 in complement region.
        Outer band spikes → trigger delayed RT surface rewiring.
    """
    dl = dilate4(mask, steps=width)
    return (dl & (~mask))


def count_holes_nonperiodic(mask: np.ndarray) -> int:
    """
    Count holes in binary mask using non-periodic boundary conditions.
    
    Algorithm:
    1. Flood fill from all border pixels into background (~mask)
    2. Remaining unvisited background components are holes
    
    Euler characteristic relation:
        χ = #components - #holes
        For simply connected region: χ = 1 → #holes = #components - 1
    
    Args:
        mask: Boolean 2D array (foreground = True)
    
    Returns:
        Number of holes (int)
    
    Physical Interpretation:
        Topological term in RT entropy: S_RT ∝ perimeter + w_hole·holes
        Higher hole count → more complex entanglement structure.
        
    Note:
        Non-periodic BC: grid edges treated as exterior (not wraparound).
        This matches physical interpretation where boundary is "at infinity".
    
    Example:
        Solid square:           holes = 0
        Square with center hole: holes = 1
        Swiss cheese:           holes = many
    """
    H, W = mask.shape
    inv = (~mask).astype(np.uint8)  # Background
    vis = np.zeros_like(inv, dtype=bool)
    q = deque()
    
    # Mark all border background pixels as "outside" (not holes)
    for i in range(H):
        if inv[i, 0]:
            q.append((i, 0))
            vis[i, 0] = True
        if inv[i, W - 1]:
            q.append((i, W - 1))
            vis[i, W - 1] = True
    
    for j in range(W):
        if inv[0, j]:
            q.append((0, j))
            vis[0, j] = True
        if inv[H - 1, j]:
            q.append((H - 1, j))
            vis[H - 1, j] = True
    
    # Flood fill from border (mark all exterior background)
    while q:
        ci, cj = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = ci + di, cj + dj
            if 0 <= ni < H and 0 <= nj < W and inv[ni, nj] and not vis[ni, nj]:
                vis[ni, nj] = True
                q.append((ni, nj))
    
    # Count remaining unvisited background components (holes)
    holes = 0
    for i in range(H):
        for j in range(W):
            if inv[i, j] and not vis[i, j]:
                holes += 1
                # Flood fill this hole (mark as visited)
                q = deque([(i, j)])
                vis[i, j] = True
                while q:
                    ci, cj = q.popleft()
                    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ni, nj = ci + di, cj + dj
                        if 0 <= ni < H and 0 <= nj < W and inv[ni, nj] and not vis[ni, nj]:
                            vis[ni, nj] = True
                            q.append((ni, nj))
    
    return holes
