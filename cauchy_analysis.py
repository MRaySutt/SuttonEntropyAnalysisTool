import os
from scipy.stats import kurtosis, cauchy
import numpy as np
import matplotlib.pyplot as plt
import h5py
from gwpy.timeseries import TimeSeries
import time
import threading
import random
from pathlib import Path

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

processed_folder = os.path.join(str(Path.home()), "SEAT_processed")
processed_folder = processed_folder.strip().replace("\r", "").replace("\n", "")

#Load Event ID
event_id_path = os.path.join(processed_folder, "event_id.txt")
with open(event_id_path, "r") as f:
    event_id = f.read().strip()

#LOAD EM IN
entropy_hdf5_path = os.path.abspath(os.path.join(processed_folder.strip(), "entropy_results.hdf5"))

with h5py.File(entropy_hdf5_path, "r") as f:
    shannon_h1 = f["entropy_h1_time_shannon"][:]
    shannon_l1 = f["entropy_l1_time_shannon"][:]
    entropy_quiet_h1_time_shannon = f["entropy_quiet_h1_time_shannon"][:]
    entropy_quiet_l1_time_shannon = f["entropy_quiet_l1_time_shannon"][:]
    renyi_h1 = f["entropy_h1_time_renyi"][:]
    renyi_l1 = f["entropy_l1_time_renyi"][:]
    entropy_quiet_h1_time_renyi = f["entropy_quiet_h1_time_renyi"][:]   
    entropy_quiet_l1_time_renyi = f["entropy_quiet_l1_time_renyi"][:]
    tsallis_h1 = f["entropy_h1_time_tsallis"][:]
    tsallis_l1 = f["entropy_l1_time_tsallis"][:]
    entropy_quiet_h1_time_tsallis = f["entropy_quiet_h1_time_tsallis"][:]
    entropy_quiet_l1_time_tsallis = f["entropy_quiet_l1_time_tsallis"][:]

#Fit the cauchy distribution on real entropy
x0_h1_shannon, gamma_h1_shannon = cauchy.fit([shannon_h1])
x0_l1_shannon, gamma_l1_shannon = cauchy.fit([shannon_l1])

x0_h1_renyi, gamma_h1_renyi = cauchy.fit([renyi_h1])
x0_l1_renyi, gamma_l1_renyi = cauchy.fit([renyi_l1])

x0_h1_tsallis, gamma_h1_tsallis = cauchy.fit([tsallis_h1])
x0_l1_tsallis, gamma_l1_tsallis = cauchy.fit([tsallis_l1])

#compute the log-likelihood of real entropy under Cauchy model
log_likelihood_h1_shannon = cauchy.logpdf(shannon_h1, loc=x0_h1_shannon, scale=gamma_h1_shannon)
log_likelihood_l1_shannon = cauchy.logpdf(shannon_l1, loc=x0_l1_shannon, scale=gamma_l1_shannon)

log_likelihood_h1_renyi = cauchy.logpdf(renyi_h1, loc=x0_h1_renyi, scale=gamma_h1_renyi)
log_likelihood_l1_renyi = cauchy.logpdf(renyi_l1, loc=x0_l1_renyi, scale=gamma_l1_renyi)

log_likelihood_h1_tsallis = cauchy.logpdf(tsallis_h1, loc=x0_h1_tsallis, scale=gamma_h1_tsallis)
log_likelihood_l1_tsallis = cauchy.logpdf(tsallis_l1, loc=x0_l1_tsallis, scale=gamma_l1_tsallis)

#Compute Kurtosis using real entropy
kurtosis_h1_shannon = float(np.nan_to_num(kurtosis(shannon_h1.flatten(), fisher=True), nan=0.0))
kurtosis_l1_shannon = float(np.nan_to_num(kurtosis(shannon_l1.flatten(), fisher=True), nan=0.0))

kurtosis_h1_renyi = float(np.nan_to_num(kurtosis(renyi_h1.flatten() , fisher=True), nan=0.0))
kurtosis_l1_renyi = float(np.nan_to_num(kurtosis(renyi_l1.flatten(), fisher=True), nan=0.0))

kurtosis_h1_tsallis = float(np.nan_to_num(kurtosis(tsallis_h1.flatten(), fisher=True), nan=0.0))
kurtosis_l1_tsallis = float(np.nan_to_num(kurtosis(tsallis_l1.flatten(), fisher=True), nan=0.0))

#Kurtosis interpretation for Shannon
if kurtosis_h1_shannon > 3:
  kurtosis_evidence_h1_shannon = "Shannon H1 entropy shows strong heavy-tailed behavior (Cauchy-like)."
else:
  kurtosis_evidence_h1_shannon = "Shannon H1 entropy does not strongly indicate heavy-tailed behavior."
if kurtosis_l1_shannon > 3:
  kurtosis_evidence_l1_shannon = "Shannon L1 entropy shows strong heavy-tailed behavior (Cauchy-like)."
else:
  kurtosis_evidence_l1_shannon = "Shannon L1 entropy does not strongly indicate heavy-tailed behavior."


#Kurtosis interpretation for Renyi
if kurtosis_h1_renyi > 3:
  kurtosis_evidence_h1_renyi = "Renyi H1 entropy shows strong heavy-tailed behavior (Cauchy-like)."
else:
  kurtosis_evidence_h1_renyi = "Renyi H1 entropy does not strongly indicate heavy-tailed behavior."
if kurtosis_l1_renyi > 3:
  kurtosis_evidence_l1_renyi = "Renyi L1 entropy shows strong heavy-tailed behavior (Cauchy-like)."
else:
  kurtosis_evidence_l1_renyi = "Renyi L1 entropy does not strongly indicate heavy-tailed behavior."


#Kurtosis interpretation for Tsallis
if kurtosis_h1_tsallis > 3:
  kurtosis_evidence_h1_tsallis = "Tsallis H1 entropy shows strong heavy-tailed behavior (Cauchy-like)."
else:
  kurtosis_evidence_h1_tsallis = "Tsallis H1 entropy does not strongly indicate heavy-tailed behavior."
if kurtosis_l1_tsallis > 3:
  kurtosis_evidence_l1_tsallis = "Tsallis L1 entropy shows strong heavy-tailed behavior (Cauchy-like)."
else:
  kurtosis_evidence_l1_tsallis = "Tsallis L1 entropy does not strongly indicate heavy-tailed behavior."



print(f"Cauchy Log-Likelihood for Shannon H1 Entropy: {log_likelihood_h1_shannon}")
print(f"Cauchy Log-Likelihood for Shannon L1 Entropy: {log_likelihood_l1_shannon}")

print(f"Cauchy Log-Likelihood for Renyi H1 Entropy: {log_likelihood_h1_renyi}")
print(f"Cauchy Log-Likelihood for Renyi L1 Entropy: {log_likelihood_l1_renyi}")

print(f"Cauchy Log-Likelihood for Tsallis H1 Entropy: {log_likelihood_h1_tsallis}")
print(f"Cauchy Log-Likelihood for Tsallis L1 Entropy: {log_likelihood_l1_tsallis}")

print(f"Shannon Kurtosis H1: {kurtosis_h1_shannon} ({kurtosis_evidence_h1_shannon})")
print(f"Shannon Kurtosis L1: {kurtosis_l1_shannon} ({kurtosis_evidence_l1_shannon})")

print(f"Renyi Kurtosis H1: {kurtosis_h1_renyi} ({kurtosis_evidence_h1_renyi})")
print(f"Renyi Kurtosis L1: {kurtosis_l1_renyi} ({kurtosis_evidence_l1_renyi})")

print(f"Tsallis Kurtosis H1: {kurtosis_h1_tsallis} ({kurtosis_evidence_h1_tsallis})")
print(f"Tsallis Kurtosis L1: {kurtosis_l1_tsallis} ({kurtosis_evidence_l1_tsallis})")


#save the results
with h5py.File(os.path.join(processed_folder, "cauchy_entropy_results.hdf5"), "w", driver="core") as f:
  f.create_dataset("cauchy_log_likelihood_h1_shannon", data=log_likelihood_h1_shannon)
  f.create_dataset("cauchy_log_likelihood_l1_shannon", data=log_likelihood_l1_shannon)
  f.create_dataset("cauchy_log_likelihood_h1_renyi", data=log_likelihood_h1_renyi)
  f.create_dataset("cauchy_log_likelihood_l1_renyi", data=log_likelihood_l1_renyi)
  f.create_dataset("cauchy_log_likelihood_h1_tsallis", data=log_likelihood_h1_tsallis)
  f.create_dataset("cauchy_log_likelihood_l1_tsallis", data=log_likelihood_l1_tsallis)
  f.create_dataset("kurtosis_h1_shannon", data=kurtosis_h1_shannon)
  f.create_dataset("kurtosis_l1_shannon", data=kurtosis_l1_shannon)
  f.create_dataset("kurtosis_h1_renyi", data=kurtosis_h1_renyi)
  f.create_dataset("kurtosis_l1_renyi", data=kurtosis_l1_renyi)
  f.create_dataset("kurtosis_h1_tsallis", data=kurtosis_h1_tsallis)
  f.create_dataset("kurtosis_l1_tsallis", data=kurtosis_l1_tsallis)


print("Cauchy entropy results saved.")
exit()
