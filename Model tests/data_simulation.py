import numpy as np
from scipy.stats import weibull_min # r weibull simulation
from scipy.stats import norm # for covariate simulation
from scipy.stats import gamma # for weibull shape parameter
from scipy.stats import bernoulli # for censoring
from scipy.stats import uniform
from scipy.stats.mstats import mquantiles
import pandas as pd

def weibull_simple_linear_sim(betas, prop_cat, obs, censor_prop, pi = 0.5):
    # betas: predictors
    # prop_cat: approximate proportion of the variables that will be categorical, determining p_cont and p_cat
    # obs: number of observations to simulate
    # censor_prop: proportion of individuals to censored
    # show_beta: prints the linear coefficients used in the simulation (for testing function)
    # pi: probabilities for simulating the probabilities in the Bernoulli disribution for the categorical variables
    
    n_beta = len(betas)
    # matrix of normal covariates
    X_norm = norm.rvs(size = obs*int(np.floor(n_beta*(1-prop_cat)))).reshape((obs,int(np.floor(n_beta*(1-prop_cat)))))
    # matrix of categorical covariates
    X_cat = bernoulli.rvs(p = pi,size = obs*int(np.ceil(n_beta*(prop_cat)))).reshape((obs,int(np.ceil(n_beta*(prop_cat)))))
    X = np.hstack([X_norm,X_cat])
    
    # shape parameter of weibull
    c = 5
    
    # calculating linear predictor
    lin_pred = np.matmul(X,betas) 
    
    # creating a dataframe for the simulation
    sim_data = pd.DataFrame(X)
    # simulating survival times from weibull distribution
    sim_data["y"] = weibull_min.rvs(c, scale = np.exp(-lin_pred/c)) 

    # proportion of censors caused by dropping out
    dropout_prop = 0.2
    # quantile above which we censor
    max_time = float(mquantiles(sim_data["y"], prob = (1 - censor_prop)/(1 - dropout_prop*censor_prop)))
    # censoring indicator invdividuals above the quantile
    sim_data["end_censor"] = np.where(sim_data["y"] > max_time,True,False)
    # dropout indicator for individuals not censored by end censoring
    sim_data["dropout"] = np.where(sim_data["end_censor"] == False, bernoulli.rvs(p = dropout_prop*censor_prop, size = len(sim_data)) == 1, False)
    # "end censoring" times
    sim_data["end_censor_time"] = np.where(sim_data["y"] > max_time, max_time, sim_data["y"])
    # simulating the dropout time
    sim_data["time"] = np.where(sim_data["dropout"], uniform.rvs(scale = sim_data["end_censor_time"]), sim_data["end_censor_time"])

    sim_data["event"] = ~(sim_data["dropout"] | sim_data["end_censor"]) 
    
    return(sim_data)

def cch_splitter(sample):
    # splits sample data into cases, subcohort, cohort and test sets
    
    cohort = sample.iloc[0:int(round(2*len(sample)/3))] # subsetting the cohort
    cases = cohort[cohort['event'] == True] # subsetting cases in the cohort
    subcohort = cohort.sample(n = len(cases))
    
    test = sample.iloc[int(np.round(2*len(sample)/3)):len(sample)] # subsetting the test set
    
    return(cases, subcohort, cohort, test)

def weibull_nonlinear_sim(phi, F_x, obs, censor_prop):
    # phi: vectorised non-linear transformation
    # obs: number of observations to simulate
    # censor_prop: proportion of individuals to censored
    # show_beta: prints the linear coefficient used in the simulation (for testing function)
    
    # matrix of normal covariates
    X = F_x
    phi_X = phi(X)
    
    # shape parameter of weibull
    c = uniform.rvs(size = 1, loc = 0.5, scale = 10) 
    
    # creating a dataframe for the simulation
    sim_data = pd.DataFrame(phi_X)
    # simulating survival times from weibull distribution
    sim_data["y"] = weibull_min.rvs(c, scale = np.exp(-phi_X/c)) 

    # proportion of censors caused by dropping out
    dropout_prop = uniform.rvs(size = 1, scale = 0.5)
    # quantile above which we censor
    max_time = float(mquantiles(sim_data["y"], prob = (1 - censor_prop)/(1 - dropout_prop*censor_prop)))
    # censoring indicator invdividuals above the quantile
    sim_data["end_censor"] = np.where(sim_data["y"] > max_time,True,False)
    # dropout indicator for individuals not censored by end censoring
    sim_data["dropout"] = np.where(sim_data["end_censor"] == False, bernoulli.rvs(p = dropout_prop*censor_prop, size = len(sim_data)) == 1, False)
    # "end censoring" times
    sim_data["end_censor_time"] = np.where(sim_data["y"] > max_time, max_time, sim_data["y"])
    # simulating the dropout time
    sim_data["time"] = np.where(sim_data["dropout"], uniform.rvs(scale = sim_data["end_censor_time"]), sim_data["end_censor_time"])

    sim_data["event"] = ~(sim_data["dropout"] | sim_data["end_censor"]) 
    
    return(sim_data)

def weibull_interaction(betas, n_cat, obs, censor_prop, show_beta = False, pi = 0.5):
    # betas: vector of coefficients
    # n_cat: approximate proportion of the variables that will be categorical, determining p_cont and p_cat
    # obs: number of observations to simulate
    # censor_prop: proportion of individuals to censored
    # show_beta: prints the linear coefficients used in the simulation (for testing function)
    # pi: probabilities for simulating the probabilities in the Bernoulli disribution for the categorical variables
    
    # matrix of normal covariates
    X_norm = norm.rvs(size = obs*(2 - n_cat)).reshape((obs,2 - n_cat))
    # matrix of categorical covariates
    X_cat = bernoulli.rvs(p = pi,size = obs*n_cat).reshape((obs,n_cat))
    X = np.hstack([X_norm,X_cat])
    X_interact = np.array(np.multiply(X[:,0],X[:,1])).reshape((obs,1))
    X = np.hstack([X, X_interact])
    
    # shape parameter of weibull
    c = 5
    
    # calculating linear predictor
    lin_pred = np.matmul(X,betas) 
    
    # creating a dataframe for the simulation
    sim_data = pd.DataFrame(X)
    # simulating survival times from weibull distribution
    sim_data["y"] = weibull_min.rvs(c, scale = np.exp(-lin_pred/c)) 

    # proportion of censors caused by dropping out
    dropout_prop = 0.2
    # quantile above which we censor
    max_time = float(mquantiles(sim_data["y"], prob = (1 - censor_prop)/(1 - dropout_prop*censor_prop)))
    # censoring indicator invdividuals above the quantile
    sim_data["end_censor"] = np.where(sim_data["y"] > max_time,True,False)
    # dropout indicator for individuals not censored by end censoring
    sim_data["dropout"] = np.where(sim_data["end_censor"] == False, bernoulli.rvs(p = dropout_prop*censor_prop, size = len(sim_data)) == 1, False)
    # "end censoring" times
    sim_data["end_censor_time"] = np.where(sim_data["y"] > max_time, max_time, sim_data["y"])
    # simulating the dropout time
    sim_data["time"] = np.where(sim_data["dropout"], uniform.rvs(scale = sim_data["end_censor_time"]), sim_data["end_censor_time"])

    sim_data["event"] = ~(sim_data["dropout"] | sim_data["end_censor"]) 
    
    if show_beta:
        print(betas)
        return(sim_data)
    else:
        return(sim_data)
    
def weibull_interaction2(betas, n_cat, obs, censor_prop, show_beta = False, pi = 0.5):
    # betas: vector of coefficients
    # n_cat: approximate proportion of the variables that will be categorical, determining p_cont and p_cat
    # obs: number of observations to simulate
    # censor_prop: proportion of individuals to censored
    # show_beta: prints the linear coefficients used in the simulation (for testing function)
    # pi: probabilities for simulating the probabilities in the Bernoulli disribution for the categorical variables
    
    # matrix of normal covariates
    X_norm = norm.rvs(size = obs*(2 - n_cat)).reshape((obs,2 - n_cat))
    # matrix of categorical covariates
    X_cat = bernoulli.rvs(p = pi,size = obs*n_cat).reshape((obs,n_cat))
    X = np.hstack([X_norm,X_cat])
    X_interact = np.array(np.multiply(X[:,0],X[:,1])).reshape((obs,1))
    X = np.hstack([X, X_interact])
    
    # shape parameter of weibull
    c = 5
    
    # calculating linear predictor
    lin_pred = np.matmul(X,betas) 
    
    # creating a dataframe for the simulation
    sim_data = pd.DataFrame(X)
    # simulating survival times from weibull distribution
    sim_data["y"] = weibull_min.rvs(c, scale = np.exp(-lin_pred/c)) 

    # proportion of censors caused by dropping out
    dropout_prop = 0.2
    # quantile above which we censor
    max_time = float(mquantiles(sim_data["y"], prob = (1 - censor_prop)/(1 - dropout_prop*censor_prop)))
    # censoring indicator invdividuals above the quantile
    sim_data["end_censor"] = np.where(sim_data["y"] > max_time,True,False)
    # dropout indicator for individuals not censored by end censoring
    sim_data["dropout"] = np.where(sim_data["end_censor"] == False, bernoulli.rvs(p = dropout_prop*censor_prop, size = len(sim_data)) == 1, False)
    # "end censoring" times
    sim_data["end_censor_time"] = np.where(sim_data["y"] > max_time, max_time, sim_data["y"])
    # simulating the dropout time
    sim_data["time"] = np.where(sim_data["dropout"], uniform.rvs(scale = sim_data["end_censor_time"]), sim_data["end_censor_time"])

    sim_data["event"] = ~(sim_data["dropout"] | sim_data["end_censor"]) 
    
    if show_beta:
        print(betas)
        return(sim_data)
    else:
        return(sim_data)

def AFT_sim(betas, prop_cat, sigma, W, obs, censor_prop, show_beta = False, pi = 0.5):
    # n_beta: vector of effect sizes
    # prop_cat: approximate proportion of the variables that will be categorical, determining p_cont and p_cat
    # W: Random variable to use for W
    # obs: number of observations to simulate
    # censor_prop: proportion of individuals to censored
    # show_beta: prints the linear coefficients used in the simulation (for testing function)
    # pi: probabilities for simulating the probabilities in the Bernoulli disribution for the categorical variables
    
    # simulating beta coefficients
    n_beta = len(betas)
    # matrix of normal covariates
    X_norm = norm.rvs(size = obs*int(np.floor(n_beta*(1-prop_cat)))).reshape((obs,int(np.floor(n_beta*(1-prop_cat)))))
    # matrix of categorical covariates
    X_cat = bernoulli.rvs(p = pi,size = obs*int(np.ceil(n_beta*(prop_cat)))).reshape((obs,int(np.ceil(n_beta*(prop_cat)))))
    X = np.hstack([X_norm,X_cat])
    
    # calculating linear predictor
    lin_pred = np.matmul(X,betas) 
    
    # creating a dataframe for the simulation
    sim_data = pd.DataFrame(X)
    # simulating survival times AFT model
    sim_data["y"] = np.exp(lin_pred + sigma*W)

    # proportion of censors caused by dropping out
    dropout_prop = uniform.rvs(size = 1, scale = 0.5)
    # quantile above which we censor
    max_time = float(mquantiles(sim_data["y"], prob = (1 - censor_prop)/(1 - dropout_prop*censor_prop)))
    # censoring indicator invdividuals above the quantile
    sim_data["end_censor"] = np.where(sim_data["y"] > max_time,True,False)
    # dropout indicator for individuals not censored by end censoring
    sim_data["dropout"] = np.where(sim_data["end_censor"] == False, bernoulli.rvs(p = dropout_prop*censor_prop, size = len(sim_data)) == 1, False)
    # "end censoring" times
    sim_data["end_censor_time"] = np.where(sim_data["y"] > max_time, max_time, sim_data["y"])
    # simulating the dropout time
    sim_data["time"] = np.where(sim_data["dropout"], uniform.rvs(scale = sim_data["end_censor_time"]), sim_data["end_censor_time"])

    sim_data["event"] = ~(sim_data["dropout"] | sim_data["end_censor"]) 
    
    if show_beta:
        print(betas)
        return(sim_data)
    else:
        return(sim_data)