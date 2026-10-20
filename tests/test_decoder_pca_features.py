import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from decoder.research.pca_features import prepare_pca_folds


@pytest.mark.parametrize("factory", [
    lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5]),
    lambda: LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000),
    lambda: SVC(C=1, class_weight="balanced"),
])
def test_shared_projection_preserves_original_pipeline_predictions(factory):
    rng = np.random.default_rng(4)
    values = rng.normal(size=(100, 18))
    labels = (values[:, 0] + values[:, 1] > 0).astype(int)
    folds = [(values[:40], values[40:60]), (values[:60], values[60:80])]
    projected = prepare_pca_folds(folds, 5)
    for (train, test), (projected_train, projected_test) in zip(folds, projected):
        original = make_pipeline(
            StandardScaler(),
            PCA(n_components=5, svd_solver="randomized", iterated_power=3, random_state=0),
            factory(),
        ).fit(train, labels[:len(train)])
        reused = factory().fit(projected_train, labels[:len(train)])
        np.testing.assert_array_equal(reused.predict(projected_test), original.predict(test))
        np.testing.assert_allclose(reused.decision_function(projected_test), original.decision_function(test))


def test_projection_fits_once_per_fold_and_heldout_data_cannot_change_training(monkeypatch):
    rng = np.random.default_rng(8)
    values = rng.normal(size=(100, 18))
    folds = [(values[:40], values[40:60]), (values[:60], values[60:80])]
    original_fit = PCA.fit_transform
    fit_sizes = []

    def counted_fit(self, values, *args, **kwargs):
        fit_sizes.append(len(values))
        return original_fit(self, values, *args, **kwargs)

    monkeypatch.setattr(PCA, "fit_transform", counted_fit)
    projected = prepare_pca_folds(folds, 5)
    assert fit_sizes == [40, 60]
    changed = prepare_pca_folds([(train, test * 100 + 500) for train, test in folds], 5)
    for (train, test), (changed_train, changed_test) in zip(projected, changed):
        np.testing.assert_array_equal(train, changed_train)
        assert not np.allclose(test, changed_test)
