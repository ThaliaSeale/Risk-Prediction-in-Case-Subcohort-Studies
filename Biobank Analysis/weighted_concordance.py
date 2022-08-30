import pandas as pd
import numpy as np

def weighted_concordance(event_times, predicted_scores, event_observed, samp_fraction):

    
    example_data = pd.DataFrame({
    "event_times": np.array(event_times),
    "predicted_scores": np.array(predicted_scores),
    "event_observed": np.array(event_observed)}).sort_values("event_times").drop_duplicates().reset_index(drop = True)
    
    n_case_case = 0
    n_case_control = 0
    total_case_pairs = 0
    total_control_pairs = 0
    
    for i in range(len(example_data)):
        if example_data.at[i,"event_observed"]:
            for j in range(i+1,len(example_data)):
                if example_data.at[j,"event_observed"]:
                    if example_data.at[i,"predicted_scores"] > example_data.at[j,"predicted_scores"]:
                        n_case_case = n_case_case + 1
                    elif example_data.at[i,"predicted_scores"] == example_data.at[j,"predicted_scores"]:
                        n_case_case = n_case_case + 0.5
                    total_case_pairs = total_case_pairs + 1
                else:
                    if example_data.at[i,"predicted_scores"] > example_data.at[j,"predicted_scores"]:
                        n_case_control = n_case_control + 1
                    elif example_data.at[i,"predicted_scores"] == example_data.at[j,"predicted_scores"]:
                        n_case_control = n_case_control + 0.5
                    total_control_pairs = total_control_pairs + 1
    return((n_case_case + 1/samp_fraction * n_case_control )/ (total_case_pairs + 1/samp_fraction * total_control_pairs))
#     return(n_case_case,total_case_pairs,n_case_control,total_control_pairs)