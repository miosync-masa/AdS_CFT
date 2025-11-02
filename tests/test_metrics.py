import numpy as np
from lambda3_holo.metrics import crosscorr_at_lags, spearman_corr, transfer_entropy

def test_corr_shapes():
    a = np.sin(np.linspace(0, 4*np.pi, 200))
    b = np.roll(a, 5)
    lags, corrs = crosscorr_at_lags(a, b, maxlag=16)
    assert len(lags) == len(corrs)
    assert np.nanmax(corrs) > 0.5

def test_spearman_and_te():
    x = np.linspace(0,1,300) + 0.05*np.random.randn(300)
    y = x**2 + 0.05*np.random.randn(300)
    rho = spearman_corr(x,y)
    assert rho > 0.6
    te_xy = transfer_entropy(x,y,n_bins=3)
    te_yx = transfer_entropy(y,x,n_bins=3)
    assert te_xy != te_yx
