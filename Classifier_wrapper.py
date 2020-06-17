import numpy as np

from skmultiflow.trees.hoeffding_tree import HoeffdingTreeClassifier


class HoeffdingTreeWrapper(HoeffdingTreeClassifier):

    def __init__(self, **kwargs):
        super(HoeffdingTreeWrapper, self).__init__(**kwargs)

    def predict_proba(self, X):
        proba = super(HoeffdingTreeWrapper, self).predict_proba(X)
        if len(proba.shape) == 2:
            return proba
        else:
            for i in range(proba.shape[0]):
                if proba[i].shape[0] == 1:
                    proba[i] = np.append(proba[i],[0.0])
            proba = np.stack(proba)
            return proba
