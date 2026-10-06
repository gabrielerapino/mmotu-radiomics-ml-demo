import numpy as np
from sklearn.model_selection import StratifiedKFold


def test_patient_groups_do_not_cross_folds_when_grouped():
    y=np.repeat([0,1],12); groups=np.repeat(np.arange(12),2)
    from sklearn.model_selection import StratifiedGroupKFold
    cv=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=42)
    for train,test in cv.split(np.zeros(len(y)),y,groups):
        assert not set(groups[train]) & set(groups[test])


def test_image_level_split_is_stratified_and_reproducible():
    y=np.repeat([0,1,2],20)
    a=list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(len(y)),y))
    b=list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(len(y)),y))
    for (at,av),(bt,bv) in zip(a,b):
        assert np.array_equal(at,bt) and np.array_equal(av,bv)
        assert set(y[at])==set(y[av])
