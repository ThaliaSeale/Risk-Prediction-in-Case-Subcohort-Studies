import numpy as np
from scipy.stats import weibull_min # r weibull simulation
from scipy.stats import norm # for covariate simulation
from scipy.stats import gamma # for weibull shape parameter
from scipy.stats import bernoulli # for censoring
from scipy.stats import uniform
from scipy.stats.mstats import mquantiles
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from lifelines import CoxPHFitter

def fit_cox(cases, subcohort,n_covariates):
    # cases: cases dataframe
    # cohort: cohort dataframe
    # n_covariates: the number of covariates used in the simulation
    
    # creating a single case subcohort dataframe
    case_subcohort_df = pd.concat([cases,subcohort])
    # removing unnecessary columns and duplicate rows
    case_subcohort_df = case_subcohort_df.loc[case_subcohort_df.duplicated() == False,[i for i in range(0,n_covariates)]+["time", "event"]]
    
    # creating the model and fitting the data
    cph = CoxPHFitter()
    cph.fit(case_subcohort_df, duration_col = "time", event_col = "event")
    return(cph)

def barlow_trans(cases,subcohort, n_covariates, alpha):
    # cases: cases dataframe
    # subcohort: subcohort dataframe
    # n_covariates: number of covariates used in the simulation
    # alpha: the sampling proportion used for the subcohort
    
    
    # finding the order of magnitude of data to pick the appropriate size of each "instant". We use the largest event time for this.
    order = int(np.floor(np.log(max(cases["time"]))/np.log(10))) 
    
    cases = cases.assign(
        # setting events outside subcohort to start just before they occur
        start_time = lambda df: df["time"] - 10**-(- order + 5),
        # adding appropriate weight
        weight = 1,
        subcohort = False
    )
    # setting times < 0  to 0
    cases["start_time"] = np.where(cases["start_time"] < 0, 0, cases["start_time"]) 
    
    subcohort = subcohort.assign(
        # if it is a case, the weight should be the same as the subcohort until close to the time of the event. 
        time = lambda df: np.where(df["event"], df["time"] - 10**-(- order + 5), df["time"]), 
        # the events start from the origin
        start_time = 0, 
        event = False,
        weight = 1/alpha,
        subcohort = True
    )
    # drop any rows where the start time in cases is 0, this is equivaent to time < 0 in subcohort
    subcohort = subcohort.query('time > 0')

    return(pd.concat([cases,subcohort])[[i for i in range(0,n_covariates)]+["start_time","time", "event","weight","subcohort"]])

def fit_cox_barlow(cases, subcohort,n_covariates,len_cohort):
    # cases: cases dataframe
    # subcohort: subcohort dataframe
    # n_covariates: number of covariates used in the simulation

    case_subcohort_df = barlow_trans(cases,subcohort,n_covariates,len(subcohort)/len_cohort).drop(columns = "subcohort")
    
    # creating the model and fitting the data
    cph = CoxPHFitter()
    cph.fit(case_subcohort_df, entry_col = "start_time", duration_col = "time",event_col = "event",weights_col = "weight",robust = True)
    return(cph)

def prentice_trans(cases,subcohort,n_covariates):
    # cases: cases dataframe
    # subcohort: subcohort dataframe
    # n_covariates: number of covariates used in the simulation
    
    # finding the order of magnitude of data to pick the appropriate size of each "instant". We use the largest event time for this.
    order = int(np.floor(np.log(max(cases["time"]))/np.log(10))) 
    
    
    cases = cases.assign(
        # rounding all of the 
#         time = round(cases["time"],- order + 5),
        # setting events outside subcohort to start just before they occur
        start_time = lambda df: df["time"] - 10**-(- order + 5),
        # adding appropriate weight
        weight = 1,
        subcohort = False
    )
    # setting times < 0  to 0
    cases["start_time"] = np.where(cases["start_time"] < 0, 0, cases["start_time"]) 
    
    subcohort = subcohort.assign(
        # if it is a case, the weight should be the same as the subcohort until close to the time of the event. 
        time = lambda df: np.where(df["event"], df["time"] - 10**-(- order + 5), df["time"]), 
        # the events start from the origin
        start_time = 0, 
        event = False,
        weight = 1,
        subcohort = True
    )
    # drop any rows where the start time in cases is 0, this is equivaent to time < 0 in subcohort
    subcohort = subcohort.query('time > 0')

    return(pd.concat([cases,subcohort])[[i for i in range(0,n_covariates)]+["start_time","time", "event","weight","subcohort"]])

def fit_cox_prentice(cases, subcohort,n_covariates):
    # cases: cases dataframe
    # subcohort: subcohort dataframe
    # n_covariates: number of covariates used in the simulation
    
    case_subcohort_df = prentice_trans(cases,subcohort,n_covariates).drop(columns = "subcohort")
    
    # creating the model and fitting the data
    cph = CoxPHFitter()
    cph.fit(case_subcohort_df, entry_col = "start_time", duration_col = "time",event_col = "event",weights_col = "weight",robust = True)
    return(cph)

def self_prentice_trans(cases,subcohort,n_covariates):
    # cases: cases dataframe
    # subcohort: subcohort dataframe
    # n_covariates: number of covariates used in the simulation
    
    # finding the order of magnitude of data to pick the appropriate size of each "instant". We use the largest event time for this.
    order = int(np.floor(np.log(max(cases["time"]))/np.log(10))) 
    
    # removing the cases that are in the subcohort from the cases data frame
    cases = cases[~cases.index.isin(subcohort.index)]
    # Adding the non-subcohort case weights
    cases["weight"] = 10**(-order - 5)
    cases["subcohort"] = False
    
    subcohort = subcohort.assign(
        weight = 1,
        subcohort = True
    )

    return(pd.concat([cases,subcohort])[[i for i in range(0,n_covariates)]+["time", "event","weight","subcohort"]])

def fit_cox_self_prentice(cases, subcohort,n_covariates):
    # cases: cases dataframe
    # subcohort: subcohort dataframe
    # n_covariates: number of covariates used in the simulation
    
    case_subcohort_df = self_prentice_trans(cases,subcohort,n_covariates).drop(columns = "subcohort")
    
    # creating the model and fitting the data
    cph = CoxPHFitter()
    cph.fit(case_subcohort_df, duration_col = "time",event_col = "event",weights_col = "weight",robust = True)
    return(cph)

from cox_k_fold import cox_k_fold

def fit_pen_cox_barlow(cases, subcohort,n_covariates, len_cohort, l1_ratio = 0, penalizer_show = False):
    # choosing the penaliser
    avg_score = []
    for penalizer in range(0,20):
        score = cox_k_fold(CoxPHFitter(penalizer = penalizer/10,l1_ratio = l1_ratio),cases, subcohort,n_covariates, len_cohort, barlow_trans,"time", event_col="event", k=5, scoring_method="log_likelihood", fitter_kwargs={"weights_col": "weight", "robust": True})
        avg_score.append(np.mean(score))
    penalizer = int(np.where(avg_score == max(avg_score))[0])/10
    
    # creating the model and fitting the data
    cph = CoxPHFitter(penalizer = penalizer,l1_ratio = l1_ratio)
    case_subcohort_df = barlow_trans(cases,subcohort,n_covariates, len(subcohort)/len_cohort).drop(columns = "subcohort")
    cph.fit(case_subcohort_df, entry_col = "start_time", duration_col = "time",event_col = "event",weights_col = "weight",robust = True)
    if penalizer_show:
        return(cph, penalizer)
    else:
        return(cph)
    
def fit_pen_cox_prentice(cases, subcohort,n_covariates,len_cohort, l1_ratio = 0, penalizer_show = False):
    # choosing the penaliser
    avg_score = []
    for penalizer in range(0,20):
        score = cox_k_fold(CoxPHFitter(penalizer = penalizer/10,l1_ratio = l1_ratio),cases, subcohort,n_covariates,len_cohort, prentice_trans,"time", event_col="event", k=5, scoring_method="log_likelihood", fitter_kwargs={"weights_col": "weight", "robust": True})
        avg_score.append(np.mean(score))
    penalizer = int(np.where(avg_score == max(avg_score))[0])/10
    
    # creating the model and fitting the data
    cph = CoxPHFitter(penalizer = penalizer,l1_ratio = l1_ratio)
    case_subcohort_df = prentice_trans(cases,subcohort,n_covariates,len_cohort).drop(columns = "subcohort")
    cph.fit(case_subcohort_df, entry_col = "start_time", duration_col = "time",event_col = "event",weights_col = "weight",robust = True)
    if penalizer_show:
        return(cph, penalizer)
    else:
        return(cph)
    
def fit_pen_cox_self_prentice(cases, subcohort,n_covariates,len_cohort, l1_ratio = 0, penalizer_show = False):
    # choosing the penaliser
    avg_score = []
    for penalizer in range(0,20):
        score = cox_k_fold(CoxPHFitter(penalizer = penalizer/10),cases, subcohort,n_covariates,len_cohort, self_prentice_trans,"time", event_col="event", k=5, scoring_method="log_likelihood", fitter_kwargs={"weights_col": "weight", "robust": True})
        avg_score.append(np.mean(score))
    penalizer = int(np.where(avg_score == max(avg_score))[0])/10
    
    # creating the model and fitting the data
    cph = CoxPHFitter(penalizer = penalizer,l1_ratio = l1_ratio)
    case_subcohort_df = self_prentice_trans(cases,subcohort,n_covariates,len_cohort).drop(columns = "subcohort")
    cph.fit(case_subcohort_df, duration_col = "time",event_col = "event",weights_col = "weight",robust = True)
    if penalizer_show:
        return(cph, penalizer)
    else:
        return(cph)

from imblearn.over_sampling import RandomOverSampler
    
from sksurv.tree import SurvivalTree
from sksurv.util import Surv

def unweighted_tree(cases,subcohort,n_covariates):
    # creating case-subcohort data frame and removing duplicate entries of cases
    case_subcohort = pd.concat([cases,subcohort])
    case_subcohort = case_subcohort.drop(columns = 'subcohort').drop_duplicates()
    
    # matrix of covariates
    X_train = case_subcohort[range(0,n_covariates)]
    # (event,time) response array
    y_train = Surv().from_arrays(case_subcohort['event'], case_subcohort['time'])
    
    # fitting the tree
    tree = SurvivalTree()
    tree.fit(X_train, y_train)
    
    return(tree)
    
def ros_tree(cases,subcohort,n_covariates,len_cohort):
    # creating case-subcohort data frame and removing duplicate entries of cases
    case_subcohort = pd.concat([cases,subcohort])
    case_subcohort = case_subcohort.drop(columns = 'subcohort').drop_duplicates()
    
    # oversampled data set
    # "covariates"
    X = case_subcohort[[i for i in range(0,n_covariates)]+['time']]
    # "classes" to be oversampled. Here, cases
    y = case_subcohort["event"]
    ros = RandomOverSampler(sampling_strategy = {True: len(cases), False: len_cohort - len(cases)})
    X_resampled, y_resampled = ros.fit_resample(X, y)
    
    # matrix of covariates
    X_train = X_resampled[range(0,n_covariates)]
    # (event,time) response array
    y_train = Surv().from_arrays(y_resampled, X_resampled['time'])
    
    # fitting the tree
    tree = SurvivalTree()
    tree.fit(X_train, y_train)
    
    return(tree)

from imblearn.over_sampling import SMOTENC

def smotenc_tree(cases,subcohort,n_covariates,len_cohort):
    case_subcohort = pd.concat([cases,subcohort])
    case_subcohort = case_subcohort.drop(columns = 'subcohort').drop_duplicates()

    # oversampled data set
    # "covariates"
    X = case_subcohort[[i for i in range(0,n_covariates)]+['time']]
    # "classes" to be oversampled. Here, cases
    y = case_subcohort["event"]
    categorical_features = list(np.where([sum(~(cases[i].isin([0,1]))) == 0 for i in range(0,n_covariates)])[0])
    smote_nc = SMOTENC(categorical_features=categorical_features,sampling_strategy = {True: len(cases), False: len_cohort - len(cases)})
    X_resampled, y_resampled = smote_nc.fit_resample(X, y)
    
    # matrix of covariates
    X_train = X_resampled[range(0,n_covariates)]
    # (event,time) response array
    y_train = Surv().from_arrays(y_resampled, X_resampled['time'])
    
    # fitting the tree
    tree = SurvivalTree()
    tree.fit(X_train, y_train)
    
    return(tree)

from imblearn.over_sampling import SMOTE

def smote_tree(cases,subcohort,n_covariates):
    case_subcohort = pd.concat([cases,subcohort])
    case_subcohort = case_subcohort.drop_duplicates()

    # oversampled data set
    # "covariates"
    X = case_subcohort[[i for i in range(0,n_covariates)]+['time']]
    # "classes" to be oversampled. Here, cases
    y = case_subcohort["event"]
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # matrix of covariates
    X_train = X_resampled[range(0,n_covariates)]
    # (event,time) response array
    y_train = Surv().from_arrays(y_resampled, X_resampled['time'])
    
    # fitting the tree
    tree = SurvivalTree()
    tree.fit(X_train, y_train)
    
    return(tree)

from sksurv.ensemble import RandomSurvivalForest

def unweighted_rsf(cases,subcohort,n_covariates):
    # creating case-subcohort data frame and removing duplicate entries of cases
    case_subcohort = pd.concat([cases,subcohort])
    case_subcohort = case_subcohort.drop(columns = 'subcohort').drop_duplicates()
    
    # matrix of covariates
    X_train = case_subcohort[range(0,n_covariates)]
    # (event,time) response array
    y_train = Surv().from_arrays(case_subcohort['event'], case_subcohort['time'])
    
    # fitting the random survival forest
    rsf = RandomSurvivalForest(n_estimators=1000)
    rsf.fit(X_train, y_train)
    
    return(rsf)

def ros_rsf(cases,subcohort,n_covariates,len_cohort):
    # creating case-subcohort data frame and removing duplicate entries of cases
    case_subcohort = pd.concat([cases,subcohort])
    case_subcohort = case_subcohort.drop(columns = 'subcohort').drop_duplicates()
    
    # oversampled data set
    # "covariates"
    X = case_subcohort[[i for i in range(0,n_covariates)]+['time']]
    # "classes" to be oversampled. Here, cases
    y = case_subcohort["event"]
    ros = RandomOverSampler(sampling_strategy = {True: len(cases), False: len_cohort - len(cases)})
    X_resampled, y_resampled = ros.fit_resample(X, y)
    
    # matrix of covariates
    X_train = X_resampled[range(0,n_covariates)]
    # (event,time) response array
    y_train = Surv().from_arrays(y_resampled, X_resampled['time'])
    
    # fitting the random survival forest
    rsf = RandomSurvivalForest(n_estimators=1000)
    rsf.fit(X_train, y_train)
    
    return(rsf)

def smotenc_rsf(cases,subcohort,n_covariates,len_cohort):
    case_subcohort = pd.concat([cases,subcohort])
    case_subcohort = case_subcohort.drop(columns = 'subcohort').drop_duplicates()

    # oversampled data set
    # "covariates"
    X = case_subcohort[[i for i in range(0,n_covariates)]+['time']]
    # "classes" to be oversampled. Here, cases
    y = case_subcohort["event"]
    categorical_features = list(np.where([sum(~(cases[i].isin([0,1]))) == 0 for i in range(0,n_covariates)])[0])
    smote_nc = SMOTENC(categorical_features=categorical_features,sampling_strategy = {True: len(cases), False: len_cohort - len(cases)})
    X_resampled, y_resampled = smote_nc.fit_resample(X, y)
    
    # matrix of covariates
    X_train = X_resampled[range(0,n_covariates)]
    # (event,time) response array
    y_train = Surv().from_arrays(y_resampled, X_resampled['time'])
    
    # fitting the random survival forest
    rsf = RandomSurvivalForest(n_estimators=1000)
    rsf.fit(X_train, y_train)
    
    return(rsf)

def smote_rsf(cases,subcohort,n_covariates):
    case_subcohort = pd.concat([cases,subcohort])
    case_subcohort = case_subcohort.drop_duplicates()

    # oversampled data set
    # "covariates"
    X = case_subcohort[[i for i in range(0,n_covariates)]+['time']]
    # "classes" to be oversampled. Here, cases
    y = case_subcohort["event"]
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # matrix of covariates
    X_train = X_resampled[range(0,n_covariates)]
    # (event,time) response array
    y_train = Surv().from_arrays(y_resampled, X_resampled['time'])
    
    # fitting the random survival forest
    rsf = RandomSurvivalForest(n_estimators=1000)
    rsf.fit(X_train, y_train)
    
    return(rsf)