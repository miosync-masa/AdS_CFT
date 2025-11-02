import numpy as np
from lambda3_holo.geometry import perimeter_len_bool8, count_holes_nonperiodic, corner_count

def test_perimeter_and_holes_simple():
    m = np.zeros((10,10), dtype=bool)
    m[2:8, 2:8] = True
    assert perimeter_len_bool8(m) > 0
    assert count_holes_nonperiodic(m) == 0
    m[4:6,4:6] = False
    assert count_holes_nonperiodic(m) == 1

def test_curvature_nonzero():
    m = np.zeros((10,10), dtype=bool)
    m[2:8,2:8] = True
    assert corner_count(m) > 0
