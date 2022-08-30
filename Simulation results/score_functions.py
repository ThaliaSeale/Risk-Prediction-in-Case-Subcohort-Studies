from lifelines.utils import concordance_index
from sksurv.util import Surv

def concordance_score(n_covariates,test,model,censored = False,lifelines = False):
    
    X_test = test[range(0,n_covariates)]
    
    if censored == False:
        event_times = test["y"]
        event_observed = test["event"]
    else:
        event_times = test["time"]
        event_observed = test["event"]
    
    if lifelines:
        # test predictions
        test_preds = model.predict_partial_hazard(X_test)
        score = concordance_index(event_times, -1*test_preds, event_observed)
    else:
        y_test = Surv().from_arrays(event_observed,event_times)
        score = model.score(X_test,y_test)
    return(score)

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sksurv.metrics import integrated_brier_score
import numpy as np

def int_brier_score(cases,subcohort,cohort,test,n_covariates,model,lifelines = False):
    # cases: the case data frame
    # subcohort: the subcohort data frame
    # cohort: the cohort data frame
    # test: the test data frame
    # n_covariates: the number of covariates
    # model: the fitted model to be scored
    # lifelines: whether the model is from lifelines or not
    
    # First we get a copy of the training data to estimate the censoring distribution
    # creating case-subcohort data frame and removing duplicate entries of cases
#     case_subcohort = pd.concat([cases,subcohort])
#     case_subcohort = case_subcohort.drop_duplicates()

#     # oversampled data set
#     # "covariates"
#     X = case_subcohort[[i for i in range(0,n_covariates)]+['time']]
#     # "classes" to be oversampled. Here, cases
#     y = case_subcohort["event"]
#     ros = RandomOverSampler(sampling_strategy = {True: len(cases), False: len(cohort) - len(cases)})
#     X_resampled, y_resampled = ros.fit_resample(X, y)
#     y_train = Surv().from_arrays(y_resampled, X_resampled['time'])
    
    X_test = test[range(0,n_covariates)]
    y_train = Surv().from_arrays(cohort["event"],cohort["time"])
    y_test = Surv().from_arrays(test["event"],test["time"])
    
    if lifelines == False:
        minimum = max(min(model.event_times_),min(test["time"])) + 0.000001
        maximum = min(max(model.event_times_),max(test["time"])) - 0.000001
        
        # survival function predictions
        survs = model.predict_survival_function(X_test)

        # times at which to evaluate survival function
        times = np.arange(minimum,maximum,(maximum - minimum)/100)

        preds = np.asarray([[fn(t) for t in times] for fn in survs])

        score = integrated_brier_score(y_train, y_test, preds, times)
    else:
        # survival function predictions
        survs = model.predict_survival_function(X_test)

        # times at which to evaluate survival function
        times = survs.index[np.where((survs.index < max(test['time'])) & (survs.index > min(test['time']))) ]

        preds = np.array(survs.iloc[np.where((survs.index < max(test['time'])) & (survs.index > min(test['time']))) ]).transpose()

        score = integrated_brier_score(y_train, y_test, preds, times)
        
    return(score)