import os
from scipy.stats import ks_2samp, ttest_ind, entropy
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
import h5py
import json
import time
import threading
import random


print("It's going!")

#simulate fetching large data for error checker 
def progress_indicator():
    messages = ["Still processing... we are working on it!", "No issues yet!", "Processing the process.", "Probably working on it still.", "Calculating calculations.", "Now we are cooking!", "Sorting haystack... checking for needle"
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


print(f"{event_id}")
print(f"{base_dir}")

def compute_windowed_entropy(signal, window_size, step, entropy_func):
    entropies = []
    for start in range(0, len(signal) - window_size + 1, step):
        window = signal[start:start + window_size]
        ent = entropy_func(window)
        if not np.isnan(ent):
            entropies.append(ent)
    return np.array(entropies)



processed_folder = r"C:\Users\matts\processed"
processed_folder = processed_folder.strip().replace("\r", "").replace("\n", "")

sample_rate = 4096 #Hz

#load the data
h1_whitened_path = os.path.join(processed_folder, f"{event_id}_h1_whitened.hdf5")
l1_whitened_path = os.path.join(processed_folder, f"{event_id}_l1_whitened.hdf5")
h1_quiet_whitened_path = os.path.join(processed_folder, f"{event_id}_h1_quiet_whitened.hdf5")
l1_quiet_whitened_path = os.path.join(processed_folder, f"{event_id}_l1_quiet_whitened.hdf5")

#read the data
h1_whitened = TimeSeries.read(h1_whitened_path)
l1_whitened = TimeSeries.read(l1_whitened_path)
h1_quiet_whitened = TimeSeries.read(h1_quiet_whitened_path)
l1_quiet_whitened = TimeSeries.read(l1_quiet_whitened_path)

soft_hdf5_path = os.path.abspath(os.path.join(processed_folder.strip(), "soft_hair_results.hdf5"))
print("Trying to open:", repr(soft_hdf5_path))

#load soft hair mem results
with h5py.File(soft_hdf5_path, "r") as f:
  post_ringdown_h1 = f["post_ringdown_h1"][:]
  post_ringdown_l1 = f["post_ringdown_l1"][:]
  p_r_shan_h1 = f["p_r_shan_h1"][:]
  p_r_shan_l1 = f["p_r_shan_l1"][:]
  p_r_ren_h1 = f["p_r_ren_h1"][:]
  p_r_ren_l1 = f["p_r_ren_l1"][:]
  p_r_tsa_h1 = f["p_r_tsa_h1"][:]
  p_r_tsa_l1 = f["p_r_tsa_l1"][:]
  quiet_region_h1 = f["quiet_region_h1"][:]
  quiet_region_l1 = f["quiet_region_l1"][:]
  quiet_shan_h1 = f["quiet_shan_h1"][:]
  quiet_shan_l1 = f["quiet_shan_l1"][:]
  quiet_ren_h1 = f["quiet_ren_h1"][:]
  quiet_ren_l1 = f["quiet_ren_l1"][:]
  quiet_tsa_h1 = f["quiet_tsa_h1"][:]
  quiet_tsa_l1 = f["quiet_tsa_l1"][:]



#entropy tests again but for post-ringdown explicity
def shannon_entropy(signal):
  hist, _ = np.histogram(signal, bins=100, density=True)
  hist = hist[hist > 0] #remove zero value so no log(0)
  return entropy(hist, base=2)

def renyi_entropy(signal, alpha=0.5):
  hist, _ = np.histogram(signal, bins=100, density=True)
  hist = hist[hist > 0] 
  return 1 / (1 - alpha) * np.log2(np.sum(hist ** alpha))

def tsallis_entropy(signal, q=2):
  hist, _ = np.histogram(signal, bins=100, density=True)
  hist = hist[hist > 0]
  try:
      result = (1 - np.sum(hist ** q)) / (q - 1)
  except:
      return np.nan
  if np.isnan(result) or np.isinf(result):
      return np.nan
  return result

#windowing parameters
window_size = int(sample_rate* 0.01) #time windows
step_size = int(sample_rate * 0.005) #overlap


#compute the entropy over post ringdown windows
entropy_shannon_h1 = compute_windowed_entropy(p_r_shan_h1, window_size, step_size, shannon_entropy)
entropy_shannon_l1 = compute_windowed_entropy(p_r_shan_l1, window_size, step_size, shannon_entropy)

entropy_renyi_h1 = compute_windowed_entropy(p_r_ren_h1, window_size, step_size, renyi_entropy)
entropy_renyi_l1 = compute_windowed_entropy(p_r_ren_l1, window_size, step_size, renyi_entropy)

entropy_tsallis_h1 = compute_windowed_entropy(p_r_tsa_h1, window_size, step_size, tsallis_entropy)
entropy_tsallis_l1 = compute_windowed_entropy(p_r_tsa_l1, window_size, step_size, tsallis_entropy)

#now the quiet
shannon_quiet_h1 = compute_windowed_entropy(quiet_shan_h1, window_size, step_size, shannon_entropy)
shannon_quiet_l1 = compute_windowed_entropy(quiet_shan_l1, window_size, step_size, shannon_entropy)

renyi_quiet_h1 = compute_windowed_entropy(quiet_ren_h1, window_size, step_size, renyi_entropy)
renyi_quiet_l1 = compute_windowed_entropy(quiet_ren_l1, window_size, step_size, renyi_entropy)

tsallis_quiet_h1 = compute_windowed_entropy(quiet_tsa_h1, window_size, step_size, tsallis_entropy)
tsallis_quiet_l1 = compute_windowed_entropy(quiet_tsa_l1, window_size, step_size, tsallis_entropy)

print(f"Entropy Results for Soft Hair Region (H1): {entropy_shannon_h1}, {entropy_renyi_h1}, {entropy_tsallis_h1}")

print(f"Entropy Results for Soft Hair Region (H1 Quiet): {shannon_quiet_h1}, {renyi_quiet_h1}, {tsallis_quiet_h1}")

print(f"Entropy Results for Soft Hair Region (L1): {entropy_shannon_l1}, {entropy_renyi_l1}, {entropy_tsallis_l1}")

print(f"Entropy Results for Soft Hair Region (L1 Quiet): {shannon_quiet_l1}, {renyi_quiet_l1}, {tsallis_quiet_l1}")

#save now! 
with h5py.File("processed/soft_hair_entropy_results.hdf5", "w") as f:
  f.create_dataset("entropy_shannon_h1", data=entropy_shannon_h1)
  f.create_dataset("entropy_shannon_l1", data=entropy_shannon_l1)
  f.create_dataset("entropy_renyi_h1", data=entropy_renyi_h1)
  f.create_dataset("entropy_renyi_l1", data=entropy_renyi_l1)
  f.create_dataset("entropy_tsallis_h1", data=entropy_tsallis_h1)
  f.create_dataset("entropy_tsallis_l1", data=entropy_tsallis_l1)
  f.create_dataset("shannon_quiet_h1", data=shannon_quiet_h1)
  f.create_dataset("shannon_quiet_l1", data=shannon_quiet_l1)
  f.create_dataset("renyi_quiet_h1", data=renyi_quiet_h1)
  f.create_dataset("renyi_quiet_l1", data=renyi_quiet_l1)
  f.create_dataset("tsallis_quiet_h1", data=tsallis_quiet_h1)
  f.create_dataset("tsallis_quiet_l1", data=tsallis_quiet_l1)


print("we saved this for you.")

#normalize entropy values 
shannon_quiet_h1 /= np.max(shannon_quiet_h1)
shannon_quiet_l1 /= np.max(shannon_quiet_l1)
renyi_quiet_h1 /= np.max(renyi_quiet_h1)
renyi_quiet_l1 /= np.max(renyi_quiet_l1)
tsallis_quiet_h1 /= np.max(tsallis_quiet_h1)
tsallis_quiet_l1 /= np.max(tsallis_quiet_l1)

print(len(shannon_quiet_h1), len(entropy_shannon_h1))

def safe_ttest_ind(a,b):
    if len(a) < 2 or len(b) < 2:
        print("Warning! Not enough data points for t-test.")
        return np.nan, np.nan
    return ttest_ind(a,b)
    
def safe_ks_2samp(a,b):
    if len(a) < 2 or len(b) < 2:
        print("Warning! Not enough data points for t-test.")
        return np.nan, np.nan
    return ks_2samp(a,b)


std_quiet_h1 = np.std(shannon_quiet_h1)
std_entropy_h1 = np.std(entropy_shannon_h1)
std_quiet_l1 = np.std(shannon_quiet_l1)
std_entropy_l1 = np.std(entropy_shannon_l1)

std_results = {
    "STD_Shannon_Quiet_H1" : float(std_quiet_h1),
    "STD_Shannon_Entropy_H1" : float(std_entropy_h1),
    "STD_Shannon_Quiet_L1" : float(std_quiet_l1),
    "STD_Shannon_Entropy_L1" : float(std_entropy_l1)
}
#testing here! 
t_stat_h1_s, t_p_value_h1_s = safe_ttest_ind(shannon_quiet_h1, entropy_shannon_h1)
t_stat_l1_s, t_p_value_l1_s = safe_ttest_ind(shannon_quiet_l1, entropy_shannon_l1)
                                           
t_stat_h1_r, t_p_value_h1_r = safe_ttest_ind(renyi_quiet_h1, entropy_renyi_h1)
t_stat_l1_r, t_p_value_l1_r = safe_ttest_ind(renyi_quiet_l1, entropy_renyi_l1)

t_stat_h1_t, t_p_value_h1_t = safe_ttest_ind(tsallis_quiet_h1,entropy_shannon_h1)
t_stat_l1_t, t_p_value_l1_t = safe_ttest_ind(tsallis_quiet_l1, entropy_shannon_l1)

ks_stat_h1_s, ks_p_value_h1_s = safe_ks_2samp(shannon_quiet_h1, entropy_shannon_h1)
ks_stat_l1_s, ks_p_value_l1_s = safe_ks_2samp(shannon_quiet_l1, entropy_shannon_l1)

ks_stat_h1_r, ks_p_value_h1_r = safe_ks_2samp(renyi_quiet_h1, entropy_renyi_h1)
ks_stat_l1_r, ks_p_value_l1_r = safe_ks_2samp(renyi_quiet_l1, entropy_renyi_l1)

ks_stat_h1_t, ks_p_value_h1_t = safe_ks_2samp(tsallis_quiet_h1, entropy_shannon_h1)
ks_stat_l1_t, ks_p_value_l1_t = safe_ks_2samp(tsallis_quiet_l1, entropy_shannon_l1)

#lets print the results here
print(f"T-Test Shannon Entropy (H1): t={t_stat_h1_s:.4f}, p={t_p_value_h1_s:.4f}")
print(f"T-Test Shannon Entropy (L1): t={t_stat_l1_s:.4f}, p={t_p_value_l1_s:.4f}")
print(f"T-Test Renyi Entropy (H1): t={t_stat_h1_r:.4f}, p={t_p_value_h1_r:.4f}")
print(f"T-Test Renyi Entropy (L1): t={t_stat_l1_r:.4f}, p={t_p_value_l1_r:.4f}")
print(f"T-Test Tsallis Entropy (H1): t={t_stat_h1_t:.4f}, p={t_p_value_h1_t:.4f}")
print(f"T-Test Tsallis Entropy (L1): t={t_stat_l1_t:.4f}, p={t_p_value_l1_t:.4f}")

print(f"K-Test Shannon Entropy (H1): stat={ks_stat_h1_s:.4f}, p={t_p_value_h1_s:.4f}")
print(f"K-Test Shannon Entropy (L1): stat={ks_stat_l1_s:.4f}, p={t_p_value_l1_s:.4f}")
print(f"K-Test Renyi Entropy (H1): stat={ks_stat_h1_r:.4f}, p={t_p_value_h1_r:.4f}")
print(f"K-Test Renyi Entropy (L1): stat={ks_stat_l1_r:.4f}, p={t_p_value_l1_r:.4f}")
print(f"K-Test Tsallis Entropy (H1): stat={ks_stat_h1_t:.4f}, p={t_p_value_h1_t:.4f}")
print(f"K-Test Tsallis Entropy (L1): stat={ks_stat_l1_t:.4f}, p={t_p_value_l1_t:.4f}")

                                           

#store statistical results in a structured dictionary
stat_results = {
  "Shannon": {
    "H1_ttest_shannon": {
        "stat" : t_stat_h1_s,
        "p" : t_p_value_h1_s
    },
    "L1_ttest_shannon": {
        "stat" : t_stat_l1_s,
        "p" : t_p_value_l1_s
    },
    "H1_ks_shannon": {
        "stat" : ks_stat_h1_s,
        "p" : t_p_value_h1_s
    },
    "L1_ks_shannon": {
        "stat" : ks_stat_l1_s,
        "p" : t_p_value_l1_s
    },
    "H1_T_Interpretation": "Significant Difference" if t_p_value_h1_s < 0.05 else "No significant difference",
    "H1_T_Interpretation": "Significant Difference" if t_p_value_l1_s < 0.05 else "No significant difference",
    "H1_Ks_Interpretation": "Significant Difference" if ks_p_value_h1_s < 0.05 else "No significant difference",
    "L1_Ks_Interpretation": "Significant Difference" if ks_p_value_l1_s < 0.05 else "No significant difference"
  },
  "Renyi": {
    "H1_ttest_renyi": {
        "stat" : t_stat_h1_r,
        "p" : t_p_value_h1_r
    },
    "L1_ttest_renyi": {
        "stat" : t_stat_l1_r,
        "p" : t_p_value_l1_r
    },
    "H1_ks_renyi": {
        "stat" : ks_stat_h1_r,
        "p" : ks_p_value_h1_r
    },
    "L1_ks_renyi": {
        "stat" : ks_stat_l1_r,
        "p" : ks_p_value_l1_r
    },
    "H1_T_Interpretation": "Significant Difference" if t_p_value_h1_r < 0.05 else "No significant difference",
    "H1_T_Interpretation": "Significant Difference" if t_p_value_l1_r < 0.05 else "No significant difference",
    "H1_Ks_Interpretation": "Significant Difference" if ks_p_value_h1_r < 0.05 else "No significant difference",
    "L1_Ks_Interpretation": "Significant Difference" if ks_p_value_l1_r < 0.05 else "No significant difference"
  },
  "Tsallis": {
    "H1_ttest_tsallis": {
        "stat" : t_stat_h1_t,
        "p" : t_p_value_h1_t,   
    },
    "L1_ttest_tsallis": {
        "stat" : t_stat_l1_t,
        "p" : t_p_value_l1_t
    },
    "H1_ks_tsallis": {
        "stat" : ks_stat_h1_t,
        "p" : t_p_value_h1_t
    },
    "L1_ks_tsallis": {
        "stat" : ks_stat_l1_t,
        "p" : t_p_value_l1_t
    },
    "H1_T_Interpretation": "Significant Difference" if t_p_value_h1_t < 0.05 else "No significant difference",
    "H1_T_Interpretation": "Significant Difference" if t_p_value_l1_t < 0.05 else "No significant difference",
    "H1_Ks_Interpretation": "Significant Difference" if ks_p_value_h1_t < 0.05 else "No significant difference",
    "L1_Ks_Interpretation": "Significant Difference" if ks_p_value_l1_t < 0.05 else "No significant difference"

}}

#save results in HDF5 format
with h5py.File("processed/statistical_results.hdf5", "w") as f:
    for category, tests in stat_results.items():
        grp = f.create_group(category)
        for key, value in tests.items():
            if isinstance(value, dict):
                grp.create_dataset(key, data=json.dumps(value))
            else:
                grp.create_dataset(key, data=value)

print("Statistical Results have successfully saved.")

with h5py.File("processed/std_entropy_results_sh.hdf5", "w") as f:
    for key, value in std_results.items():
        f.create_dataset(key, data=value)

print(std_results)

exit()
