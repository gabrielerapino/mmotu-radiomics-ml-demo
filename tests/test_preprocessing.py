import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from src.pipeline import make_pipeline


def test_scaler_and_selector_fit_on_training_fold_only():
    rng=np.random.default_rng(7); X=rng.normal(size=(30,4)); y=np.repeat([0,1,2],10)
    train,test=next(StratifiedKFold(3,shuffle=True,random_state=1).split(X,y))
    X[test]+=10000
    pipe=make_pipeline(LogisticRegression(max_iter=200),seed=2,selection_percentile=100)
    pipe.fit(X[train],y[train])
    assert np.all(np.abs(pipe.named_steps["scaler"].mean_)<10)
    assert pipe.named_steps["selector"].scores_ is not None
    assert np.all(np.isfinite(pipe.predict_proba(X[test])))
