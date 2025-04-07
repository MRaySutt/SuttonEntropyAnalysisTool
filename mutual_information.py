import os
import numpy as np
import h5py
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import gaussian_kde
from gwpy.timeseries import TimeSeries
import time
import threading
import random

print("It's going!")

#simulate fetching large data for error checker 
def progress_indicator():
    messages = ["LOADING. LOADING. LOADING.", "No issues yet!", "Processing the process.", "Probably working on it still.", "Calculating calculations with math and things.", "Now we are cooking!", "Sorting haystack... checking for needle"
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
    shannon_h1 = f["entropy_h1_time_shannon"][:]
    shannon_l1 = f["entropy_l1_time_shannon"][:]
    quiet_shannon_h1 = f["entropy_quiet_h1_time_shannon"][:]
    quiet_shannon_l1 = f["entropy_quiet_l1_time_shannon"][:]
    renyi_h1 = f["entropy_h1_time_renyi"][:]
    renyi_l1 = f["entropy_l1_time_renyi"][:]
    quiet_renyi_h1 = f["entropy_quiet_h1_time_renyi"][:]   
    quiet_renyi_l1 = f["entropy_quiet_l1_time_renyi"][:]
    tsallis_h1 = f["entropy_h1_time_tsallis"][:]
    tsallis_l1 = f["entropy_l1_time_tsallis"][:]
    quiet_tsallis_h1 = f["entropy_quiet_h1_time_tsallis"][:]
    quiet_tsallis_l1 = f["entropy_quiet_l1_time_tsallis"][:]
    
#ensure both signals have the same sample rate
min_length_shannon = min(len(shannon_h1), len(shannon_l1))
shannon_h1 = shannon_h1[:min_length_shannon].reshape(-1, 1)
shannon_l1 = shannon_l1[:min_length_shannon].reshape(-1, 1)

#ensure both signals have the same sample rate
min_length_renyi = min(len(renyi_h1), len(renyi_l1))
renyi_h1 = renyi_h1[:min_length_renyi].reshape(-1, 1)
renyi_l1 = renyi_l1[:min_length_renyi].reshape(-1, 1)

#ensure both signals have the same sample rate
min_length_tsallis = min(len(tsallis_h1), len(tsallis_l1))
tsallis_h1 = tsallis_h1[:min_length_tsallis].reshape(-1, 1)
tsallis_l1 = tsallis_l1[:min_length_tsallis].reshape(-1, 1)

#compute mutual information (MI) for continous signals
mi_value_shannon = mutual_info_regression(shannon_h1, shannon_l1.ravel(), discrete_features=False)[0]

mi_value_renyi = mutual_info_regression(renyi_h1, renyi_l1.ravel(), discrete_features=False)[0]

mi_value_tsallis = mutual_info_regression(tsallis_h1, tsallis_l1.ravel(), discrete_features=False)[0]


#for normalizing
h_h1_shannon = np.var(shannon_h1)
h_l1_shannon = np.var(shannon_l1)
h_h1_renyi = np.var(renyi_h1)
h_l1_renyi = np.var(renyi_l1)
h_h1_tsallis = np.var(tsallis_h1)
h_l1_tsallis = np.var(tsallis_l1)


#normalize mutual information (NMI)
normalized_mi_shannon = mi_value_shannon / np.sqrt(h_h1_shannon * h_l1_shannon)

normalized_mi_renyi = mi_value_renyi / np.sqrt(h_h1_renyi * h_l1_renyi)

normalized_mi_tsallis = mi_value_tsallis / np.sqrt(h_h1_tsallis * h_l1_tsallis)

print(f"Shannon Mutual information (H1 - L1) on Whitened Data: {mi_value_shannon}")

print(f"Shannon Normalized Mutual Information (H1 - L1): {normalized_mi_shannon}")

print(f"Renyi Mutual information (H1 - L1) on Whitened Data: {mi_value_renyi}")

print(f"Renyi Normalized Mutual Information (H1 - L1): {normalized_mi_renyi}")

print(f"Tsallis Mutual information (H1 - L1) on Whitened Data: {mi_value_tsallis}")

print(f"Tsallis Normalized Mutual Information (H1 - L1): {normalized_mi_tsallis}")

#save results 
with h5py.File("processed/mutual_information_results.hdf5", "w") as f:
  f.create_dataset("shannon_mutual_info", data=mi_value_shannon)
  f.create_dataset("renyi_mutual_info", data=mi_value_renyi)
  f.create_dataset("tsallis_mutual_info", data=mi_value_tsallis)
  f.create_dataset("shannon_normalized_mutual_info", data=normalized_mi_shannon)
  f.create_dataset("renyi_normalized_mutual_info", data=normalized_mi_renyi)
  f.create_dataset("tsallis_normalized_mutual_info", data=normalized_mi_tsallis)


print("Mutual Information Results Saved.")
exit()
