import os
import numpy as np
from scipy.stats import norm, cauchy
import h5py
from gwpy.timeseries import TimeSeries
import time
import threading
import random

print("It's going!")

#simulate fetching large data for error checker 
def progress_indicator():
    messages = ["LOADING. LOADING. LOADING.", "No issues yet!", "Processing the process.", "Probably working on it still.", "Calculating calculations.", "Now we are cooking!", "Sorting haystack... checking for needle"
    ]
    while True:
        time.sleep(5)
        print(random.choice(messages))
progress_thread = threading.Thread(target=progress_indicator, daemon=True)
progress_thread.start()

#Check directory and make sure path is clear
base_dir = os.getcwd()

processed_folder = r"C:\Users\matts\processed"
processed_folder = processed_folder.strip().replace("\r", "").replace("\n", "")

#Load Event ID
event_id_path = os.path.join(processed_folder, "event_id.txt")
with open(event_id_path, "r") as f:
    event_id = f.read().strip()

#LOAD EM IN
entropy_hdf5_path = os.path.abspath(os.path.join(processed_folder.strip(), "entropy_results.hdf5"))

with h5py.File(entropy_hdf5_path, "r") as f:
    entropy_h1_time_shannon = f["entropy_h1_time_shannon"][:]
    entropy_l1_time_shannon = f["entropy_l1_time_shannon"][:]
    entropy_quiet_h1_time_shannon = f["entropy_quiet_h1_time_shannon"][:]
    entropy_quiet_l1_time_shannon = f["entropy_quiet_l1_time_shannon"][:]
    entropy_h1_time_renyi = f["entropy_h1_time_renyi"][:]
    entropy_l1_time_renyi = f["entropy_l1_time_renyi"][:]
    entropy_quiet_h1_time_renyi = f["entropy_quiet_h1_time_renyi"][:]   
    entropy_quiet_l1_time_renyi = f["entropy_quiet_l1_time_renyi"][:]
    entropy_h1_time_tsallis = f["entropy_h1_time_tsallis"][:]
    entropy_l1_time_tsallis = f["entropy_l1_time_tsallis"][:]
    entropy_quiet_h1_time_tsallis = f["entropy_quiet_h1_time_tsallis"][:]
    entropy_quiet_l1_time_tsallis = f["entropy_quiet_l1_time_tsallis"][:]

carlo_result_path = os.path.abspath(os.path.join(processed_folder.strip(), "monte_carlo_results.hdf5"))
print("Trying to open:", repr(carlo_result_path))

#Load our carlo sim results
with h5py.File(carlo_result_path, "r") as f:
    simulated_entropies_h1_shannon = f["simulated_entropies_h1_shannon"][:]
    simulated_entropies_l1_shannon = f["simulated_entropies_l1_shannon"][:]
    simulated_entropies_h1_renyi = f["simulated_entropies_h1_renyi"][:]
    simulated_entropies_l1_renyi = f["simulated_entropies_l1_renyi"][:]
    simulated_entropies_h1_tsallis = f["simulated_entropies_h1_tsallis"][:]
    simulated_entropies_l1_tsallis = f["simulated_entropies_l1_tsallis"][:]
    #the ones below are the ones you need. the ones up top are just mean values
    sim_entropy_h1_shannon = f["sim_entropy_h1_shannon"][:]
    sim_entropy_l1_shannon = f["sim_entropy_l1_shannon"][:]
    sim_entropy_h1_renyi = f["sim_entropy_h1_renyi"][:]
    sim_entropy_l1_renyi = f["sim_entropy_l1_renyi"][:]
    sim_entropy_h1_tsallis = f["sim_entropy_h1_tsallis"][:]
    sim_entropy_l1_tsallis = f["sim_entropy_l1_tsallis"][:]

#CREATE A SAFE BAYES FACTOR. IT LOOKS LIKE CAUCHY FIT IS WAY TOO STRONG FOR LOG METHOD
def safe_bayes_factor(log_like_model1, log_like_model2):
    #compute log bayes factor safely
    diff = log_like_model1 - log_like_model2
    max_diff = 700
    #avoid overflow by capping large differences
    diff = np.clip(diff, - max_diff, max_diff)
    return np.exp(diff)

#Define Gaussian and Cauchy models
def log_likelihood_gaussian(data, mu, sigma):
  return np.sum(norm.logpdf(data, loc=mu, scale=sigma))

def log_likelihood_cauchy(data, x0, gamma):
  return np.sum(cauchy.logpdf(data, loc=x0, scale=gamma))


#clean it up a bit
def clean_entropy(data):
    return data[np.isfinite(data)]

def evaluate_entropy_model(entropy_data):
    entropy_data = clean_entropy(entropy_data)
    mu, sigma = norm.fit(entropy_data)
    sigma = max(sigma, 1e-8)
    x0, gamma = cauchy.fit(entropy_data)
    gamma = max(gamma, 1e-8)
    log_like_gaussian = log_likelihood_gaussian(entropy_data, mu, sigma)
    log_like_cauchy = log_likelihood_cauchy(entropy_data, x0, gamma)
    K = safe_bayes_factor(log_like_cauchy, log_like_gaussian)
    if K > 10:
        evidence = "Strong evidence for Cauchy anomaly."
    elif K > 3:
        evidence = "Moderate evidence for Cauchy anomaly."
    else:
        evidence = "Gaussian model is a sufficient fit."
    return K, evidence



shannon_ent_h1 = clean_entropy(entropy_h1_time_shannon)
shannon_ent_l1 = clean_entropy(entropy_l1_time_shannon)
renyi_ent_h1 = clean_entropy(entropy_h1_time_renyi)
renyi_ent_l1 = clean_entropy(entropy_l1_time_renyi)
tsallis_ent_h1 = clean_entropy(entropy_h1_time_tsallis)
tsallis_ent_l1 = clean_entropy(entropy_l1_time_tsallis)
#fit gaussian and cauchy distributions
k_h1_s, evidence_h1_s = evaluate_entropy_model(shannon_ent_h1)
k_l1_s, evidence_l1_s = evaluate_entropy_model(shannon_ent_l1)

k_h1_r, evidence_h1_r = evaluate_entropy_model(renyi_ent_h1)
k_l1_r, evidence_l1_r = evaluate_entropy_model(renyi_ent_l1)

k_h1_t, evidence_h1_t = evaluate_entropy_model(tsallis_ent_h1)
k_l1_t, evidence_l1_t = evaluate_entropy_model(tsallis_ent_l1)


print("Shannon H1 Bayes Factor:", k_h1_s, ":", evidence_h1_s)
print("Shannon L1 Bayes Factor:", k_l1_s, ":", evidence_l1_s)

print("Renyi H1 Bayes Factor:", k_h1_r, ":", evidence_h1_r)
print("Renyi L1 Bayes Factor:", k_l1_r, ":", evidence_l1_r)

print("Tsallis H1 Bayes Factor:", k_h1_t, ":", evidence_h1_t)
print("Tsallis L1 Bayes Factor:", k_l1_t, ":", evidence_l1_t)


#save results
with h5py.File("processed/bayesian_results.hdf5", "w") as f:
    f.create_dataset("ShannonBayesFactor_H1", data=k_h1_s)
    f.create_dataset("ShannonBayesFactor_L1", data=k_l1_s)
    f.create_dataset("RenyiBayesFactor_H1", data=k_h1_r)
    f.create_dataset("RenyiBayesFactor_L1", data=k_l1_r)
    f.create_dataset("TsallisBayesFactor_H1", data=k_h1_t)
    f.create_dataset("TsallisBayesFactor_L1", data=k_l1_t)
    f.create_dataset("ShannonEvidence_H1", data=evidence_h1_s)
    f.create_dataset("ShannonEvidence_L1", data=evidence_l1_s)
    f.create_dataset("RenyiEvidence_H1", data=evidence_h1_r)
    f.create_dataset("RenyiEvidence_L1", data=evidence_l1_r)
    f.create_dataset("TsallisEvidence_H1", data=evidence_h1_t)
    f.create_dataset("TsallisEvidence_L1", data=evidence_l1_t)

print("Bayesian results saved.")
exit()
