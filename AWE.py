from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import _check_partial_fit_first_call
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score,  balanced_accuracy_score as bac
from imblearn.metrics import geometric_mean_score as g_mean
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import math

WEIGHT_CALCULATION = (
   "proportional_to_mse",
    "proportional_to_f1",
    "proportional_to_g-mean",
    "proportional_to_bac"
)


class AWE(BaseEnsemble, ClassifierMixin):

    def __init__(self, base_estimator=None, n_estimators=10, weighting_method="proportional_to_mse", sampling='', update=False):

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self._weighting_method = weighting_method
        self._sampling = sampling
        self._update = update
        self.ensemble_ = None
        self.weights_ = None
        self.classes_ = None
        self.X_ = None
        self.y_ = None

    def _prune(self):
        number_to_prune = len(self.ensemble_) - self.n_estimators
        for i in range(number_to_prune):
            self.ensemble_.pop(np.argmin(self.weights_))
            self.weights_ = np.delete(self.weights_, np.argmin(self.weights_))

    @staticmethod
    def _mean_squared_error(y_true, predicted_proba):
        corr_proba = predicted_proba[range(len(predicted_proba)), y_true]
        return sum((1 - corr_proba)**2) / len(y_true)

    @staticmethod
    def _random_mean_squared_error(y):
        return sum([sum(y == u) / len(y) * (1 - (sum(y == u) / len(y)))**2 for u in np.unique(y)])

    def _weight_of_random_classifier(self):
        if self._weighting_method == "proportional_to_mse":
            return 1 / self._random_mean_squared_error(self.y_)
        P = sum(self.y_ == 1)
        N = sum(self.y_ == 0)
        pP = P / len(self.y_)
        pN = N / len(self.y_)
        TP = P*pP
        TN = N*pN
        FP = N*pP
        FN = P*pN
        if self._weighting_method == "proportional_to_f1":
            return 2*TP / (2*TP + FP + FN)
        else:
            sens = TP / (TP + FP)
            spec = TN / (TN + FP)
            if self._weighting_method == "proportional_to_g-mean":
                return math.sqrt(sens * spec)
            elif self._weighting_method == "proportional_to_bac":
                return (sens + spec) / 2
            else:
                raise NotImplementedError

    def _get_weigth_for_candidate(self, candidate_clf):

        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5)
        weight = []
        for train_index, test_index in sss.split(self.X_, self.y_):
            for i in range(2):
                if self._sampling == 'over':
                    ros = RandomOverSampler(random_state=0)
                    X, y = ros.fit_resample(self.X_[train_index], self.y_[train_index])
                elif self._sampling == 'under':
                    rus = RandomUnderSampler(random_state=0)
                    X, y = rus.fit_resample(self.X_[train_index], self.y_[train_index])
                else:
                    X, y = self.X_[train_index], self.y_[train_index]
                candidate_clf.fit(X, y)
                if self._weighting_method == "proportional_to_mse" and not self._update:
                    weight.append(self._random_mean_squared_error(self.y_[test_index]) - self._mean_squared_error(
                        self.y_[test_index], candidate_clf.predict_proba(self.X_[test_index])))
                elif self._weighting_method == "proportional_to_mse" and self._update:
                    weight.append(1 / (self._mean_squared_error(self.y_[test_index], candidate_clf.predict_proba(self.X_[test_index])) + 0.001))
                elif self._weighting_method == "proportional_to_f1":
                    weight.append(f1_score(self.y_[test_index], candidate_clf.predict(self.X_[test_index])))
                elif self._weighting_method == "proportional_to_g-mean":
                    weight.append(g_mean(self.y_[test_index], candidate_clf.predict(self.X_[test_index])))
                elif self._weighting_method == "proportional_to_bac":
                    weight.append(bac(self.y_[test_index], candidate_clf.predict(self.X_[test_index])))
                else:
                    raise NotImplementedError

                train_index, test_index = test_index, train_index

            return sum(weight) / len(weight)

    def fit(self, X, y):
        """Fitting."""
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y

        candidate_clf = clone(self.base_estimator)
        candidate_clf.fit(X, y)

        self.ensemble_ = [candidate_clf]
        self.weights_ = np.array([1])
        self.classes_ = np.unique(y)

    def partial_fit(self, X, y, classes=None):
        """Partial fitting."""
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y

        # print(X[0,0])
        if _check_partial_fit_first_call(self, classes):
            self.classes_ = classes

            self.ensemble_ = []
            self.weights_ = np.array([])


        """Partial fitting"""


        # Preparing and training new candidate
        if classes is not None:
            self.classes_ = classes
        elif self.classes_ is None:
            raise Exception('Classes not specified')

        candidate_clf = clone(self.base_estimator)
        candidate_weight = self._get_weigth_for_candidate(candidate_clf)
        if self._sampling == 'over':
            ros = RandomOverSampler(random_state=0)
            X, y = ros.fit_resample(X, y)
        elif self._sampling == 'under':
            rus = RandomUnderSampler(random_state=0)
            X, y = rus.fit_resample(X, y)
        if not self._update:
            candidate_clf.fit(X, y)
        else:
            candidate_clf.partial_fit(X, y)

        self._set_weights()

        if self._update:
            random_cl_weight = self._weight_of_random_classifier()
            for i in range(len(self.ensemble_)):
                if self.weights_[i] > random_cl_weight:
                    self.ensemble_[i].partial_fit(X,y)

        self.ensemble_.append(candidate_clf)

        self.weights_ = np.append(self.weights_, np.array([candidate_weight]))

        # Post-pruning
        if len(self.ensemble_) > self.n_estimators:
            self._prune()

        # Weights normalization
        self.weights_ = self.weights_ / np.sum(self.weights_)

    def _set_weights(self):

        """Wang's weights"""
        if self._weighting_method == "proportional_to_mse" and not self._update:
            mse_rand = self._random_mean_squared_error(self.y_)
            mse_members = np.array([self._mean_squared_error(self.y_, member_clf.predict_proba(self.X_))
                                    for member_clf in self.ensemble_])
            self.weights_ = mse_rand - mse_members
        elif self._weighting_method == "proportional_to_mse" and self._update:
            self.weights_ = np.array([1 / (self._mean_squared_error(self.y_, member_clf.predict_proba(self.X_)) + 0.001)
                                    for member_clf in self.ensemble_])
        elif self._weighting_method == "proportional_to_f1":
            self.weights_ = np.array([f1_score(self.y_, member_clf.predict(self.X_)) for member_clf in self.ensemble_])
        elif self._weighting_method == "proportional_to_g-mean":
            self.weights_ = np.array([g_mean(self.y_, member_clf.predict(self.X_)) for member_clf in self.ensemble_])
        elif self._weighting_method == "proportional_to_bac":
            self.weights_ = np.array([bac(self.y_, member_clf.predict(self.X_)) for member_clf in self.ensemble_])
        else:
            raise NotImplementedError

    def score(self, X, y, sample_weight=None):
        prediction = self.predict(X)
        return sum(prediction == y) / len(y)

    def ensemble_support_matrix(self, X):
        """ESM."""
        return np.array([member_clf.predict_proba(X) for member_clf in self.ensemble_])

    def predict_proba(self, X):
        """Aposteriori probabilities."""
        # Check is fit had been called
        check_is_fitted(self, "classes_")

        # Weight support before acumulation
        weighted_support = (
               self.ensemble_support_matrix(X) * self.weights_[:, np.newaxis, np.newaxis]
        )

        # Acumulate supports
        acumulated_weighted_support = np.sum(weighted_support, axis=0)
        return acumulated_weighted_support

    def predict(self, X):
        """Hard decision."""
        # Check is fit had been called
        check_is_fitted(self, "classes_")

        # Input validation
        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("number of features does not match")

        supports = self.predict_proba(X)
        prediction = np.argmax(supports, axis=1)

        return self.classes_[prediction]

    def get_name(self):
        if self._update:
            return "AUE-" + self._weighting_method + '-' + self._sampling
        else:
            return "AWE-" + self._weighting_method + '-' + self._sampling

