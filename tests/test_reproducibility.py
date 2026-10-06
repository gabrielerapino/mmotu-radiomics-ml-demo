import numpy as np
from sklearn.model_selection import StratifiedKFold


def test_same_seed_produces_same_splits():
    y=np.repeat(np.arange(4),15)
    def indices(seed):
        return [(a.tolist(),b.tolist()) for a,b in StratifiedKFold(5,shuffle=True,random_state=seed).split(np.zeros(len(y)),y)]
    assert indices(2026)==indices(2026)
    assert indices(2026)!=indices(2027)
