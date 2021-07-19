# -*- coding: utf-8 -*-
"""Naive Outlier Ensemble.
"""
# Author: Roel Bouman <roel.bouman@ru.nl>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.utils import check_array

from .base import BaseDetector
from ..models.combination import average
from ..models.lof import LOF


class Ensemble(BaseDetector):
    
    def __init__(self, estimators=[LOF()], combination_function=average, contamination=0.1, **kwargs):
        super(Ensemble, self).__init__(contamination=contamination)
        self.estimators = estimators
        self.n_estimators_ = len(estimators)
        self.combination_function = combination_function
        self.kwargs = kwargs
        
    def fit(self, X, y=None):
        X = check_array(X)
        n_samples = X.shape[0]
        
        all_scores = np.zeros((n_samples,self.n_estimators_))
        
        for i, estimator in enumerate(self.estimators):
            estimator.fit(X)
            all_scores[:,i] = estimator.decision_scores_
            
        self.decision_scores_ = self.combination_function(all_scores, **self.kwargs)
        
        return self
        
    def decision_function(self, X):
        n_samples = X.shape[0]
        
        all_scores = np.zeros((n_samples,self.n_estimators_))
        
        for i, estimator in enumerate(self.estimators):
            all_scores[:,i] = estimator.decision_function(X)
        
        return self.combination_function(all_scores, **self.kwargs)