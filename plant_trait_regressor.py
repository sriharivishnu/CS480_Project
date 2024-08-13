"""
Extends sklearn's MultiOutputRegressor to support eval_set on fit


The fit method is overridden to support this change. The main body 
of the fit method is copied from MultiOutputRegressor's body.
"""

from sklearn.utils import Bunch
from sklearn.utils.validation import _check_method_params, has_fit_parameter
from sklearn.base import is_classifier, _routing_enabled
from joblib import Parallel, delayed
from sklearn.multioutput import (
    _fit_estimator,
    process_routing,
    check_classification_targets,
)
from sklearn.multioutput import MultiOutputRegressor


class PlantTraitRegressor(MultiOutputRegressor):
    def fit(self, X, y, sample_weight=None, **fit_params):
        """Fit the model to data, separately for each output variable.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        y : {array-like, sparse matrix} of shape (n_samples, n_outputs)
            Multi-output targets. An indicator matrix turns on multilabel
            estimation.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If `None`, then samples are equally weighted.
            Only supported if the underlying regressor supports sample
            weights.

        **fit_params : dict of string -> object
            Parameters passed to the ``estimator.fit`` method of each step.

            .. versionadded:: 0.23

        Returns
        -------
        self : object
            Returns a fitted instance.
        """
        if not hasattr(self.estimator, "fit"):
            raise ValueError("The base estimator should implement a fit method")

        y = self._validate_data(X="no_validation", y=y, multi_output=True)

        if is_classifier(self):
            check_classification_targets(y)

        if y.ndim == 1:
            raise ValueError(
                "y must have at least two dimensions for "
                "multi-output regression but has only one."
            )

        if _routing_enabled():
            if sample_weight is not None:
                fit_params["sample_weight"] = sample_weight
            routed_params = process_routing(
                self,
                "fit",
                **fit_params,
            )
        else:
            if sample_weight is not None and not has_fit_parameter(
                self.estimator, "sample_weight"
            ):
                raise ValueError(
                    "Underlying estimator does not support sample weights."
                )

            fit_params_validated = _check_method_params(X, params=fit_params)
            routed_params = Bunch(estimator=Bunch(fit=fit_params_validated))
            if sample_weight is not None:
                routed_params.estimator.fit["sample_weight"] = sample_weight

        eval_set = routed_params.estimator.fit.pop("eval_set")

        if type(eval_set) is list:
            X_val, Y_val = eval_set[0]
            Y_val = self._validate_data(X="no_validation", y=Y_val, multi_output=True)

            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_estimator)(
                    self.estimator,
                    X,
                    y[:, i],
                    eval_set=[(X_val, Y_val[:, i])],
                    **routed_params.estimator.fit,
                )
                for i in range(y.shape[1])
            )
        else:
            X_val, Y_val = eval_set
            Y_val = self._validate_data(X="no_validation", y=Y_val, multi_output=True)

            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_estimator)(
                    self.estimator,
                    X,
                    y[:, i],
                    eval_set=(X_val, Y_val[:, i]),
                    **routed_params.estimator.fit,
                )
                for i in range(y.shape[1])
            )

        if hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_
        if hasattr(self.estimators_[0], "feature_names_in_"):
            self.feature_names_in_ = self.estimators_[0].feature_names_in_

        return self
