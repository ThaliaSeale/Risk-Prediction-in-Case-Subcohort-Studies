import pandas as pd
import numpy as np

def weighted_concordance(event_times, predicted_scores, event_observed, samp_fraction):
    # Evaluates weighted concordance between test data and risk predictions
    
    # event_times: vector of times when events occur in the test data
    # predicted_scores: risk scores predicted by the model
    # event_observed: vector of event indicators in the test data
    # samp_fraction: the sampling fraction of the test data
    
    # data frame with all the inputs
    example_data = pd.DataFrame({
    "event_times": event_times,
    "predicted_scores": predicted_scores,
    "event_observed": event_observed}).sort_values("event_times").reset_index(drop = True)
    
    # counters for the number of:
    # concordant case-case pairs
    n_case_case = 0
    # concordant case-control pairs
    n_case_control = 0
    # total number of case-case pairs 
    total_case_pairs = 0
    # total number of case-control pairs
    total_control_pairs = 0
    
    # looping through all valid pairs
    for i in range(len(example_data)):
        if example_data.at[i,"event_observed"]: # if event observed
            for j in range(i+1,len(example_data)): # compare to all events/censored data points at a later time
                if example_data.at[j,"event_observed"]: # if subsequent individual has an event
                    if example_data.at[i,"predicted_scores"] > example_data.at[j,"predicted_scores"]: # if concordant
                        n_case_case = n_case_case + 1
                    elif example_data.at[i,"predicted_scores"] == example_data.at[j,"predicted_scores"]:
                        n_case_case = n_case_case + 0.5
                    total_case_pairs = total_case_pairs + 1
                else:  # if subsequent individual is censored
                    if example_data.at[i,"predicted_scores"] > example_data.at[j,"predicted_scores"]: # if concordant
                        n_case_control = n_case_control + 1
                    elif example_data.at[i,"predicted_scores"] == example_data.at[j,"predicted_scores"]:
                        n_case_control = n_case_control + 0.5
                    total_control_pairs = total_control_pairs + 1
    return((n_case_case + 1/samp_fraction * n_case_control )/ (total_case_pairs + 1/samp_fraction * total_control_pairs))