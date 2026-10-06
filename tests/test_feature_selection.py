import numpy as np
from sklearn.feature_selection import SelectPercentile
from sklearn.linear_model import LogisticRegression
from src.pipeline import make_pipeline


def test_feature_selector_uses_only_values_passed_to_fit():
    rng=np.random.default_rng(3); X=rng.normal(size=(24,8)); y=np.repeat([0,1],12)
    X[:12,0]-=2; X[12:,0]+=2
    pipe=make_pipeline(LogisticRegression(max_iter=200),selection_percentile=25)
    pipe.fit(X,y)
    selector=pipe.named_steps["selector"]
    expected=SelectPercentile(selector.score_func,percentile=25).fit(X,y).get_support()
    assert np.array_equal(selector.get_support(),expected)
