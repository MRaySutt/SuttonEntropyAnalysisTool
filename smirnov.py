import os
from scipy.stats import ks_2samp
import numpy as np
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
    renyi_h1 = f["entropy_h1_time_renyi"][:]
    renyi_l1 = f["entropy_l1_time_renyi"][:]
    tsallis_h1 = f["entropy_h1_time_tsallis"][:]
    tsallis_l1 = f["entropy_l1_time_tsallis"][:]


carlo_result_path = os.path.abspath(os.path.join(processed_folder.strip(), "monte_carlo_results.hdf5"))
print("Trying to open:", repr(carlo_result_path))

with h5py.File("processed/monte_carlo_results.hdf5", "r") as f:
  simulated_shannon_h1 = f["sim_entropy_h1_shannon"][:]
  simulated_shannon_l1 = f["sim_entropy_l1_shannon"][:]
  simulated_renyi_h1 = f["sim_entropy_h1_renyi"][:]
  simulated_renyi_l1 = f["sim_entropy_l1_renyi"][:]
  simulated_tsallis_h1 = f["sim_entropy_h1_tsallis"][:]
  simulated_tsallis_l1 = f["sim_entropy_l1_tsallis"][:]


#run the Kolmogorov-Smirnov test for deviation confidence level
ks_stat_h1_shannon, p_value_h1_shannon = ks_2samp(shannon_h1.ravel(), simulated_shannon_h1.ravel())
ks_stat_l1_shannon, p_value_l1_shannon = ks_2samp(shannon_l1.ravel(), simulated_shannon_l1.ravel())

ks_stat_h1_renyi, p_value_h1_renyi = ks_2samp(renyi_h1.ravel(), simulated_renyi_h1.ravel())
ks_stat_l1_renyi, p_value_l1_renyi = ks_2samp(renyi_l1.ravel(), simulated_renyi_l1.ravel())

ks_stat_h1_tsallis, p_value_h1_tsallis = ks_2samp(tsallis_h1.ravel(), simulated_tsallis_h1.ravel())
ks_stat_l1_tsallis, p_value_l1_tsallis = ks_2samp(tsallis_l1.ravel(), simulated_tsallis_l1.ravel())


#FOR SHANNON
print(f"KS Statistic: {ks_stat_h1_shannon}, P-Value: {p_value_h1_shannon}")

if p_value_h1_shannon < 0.05:
  print("Shannon H1 Entropy distribution is significantly different from Kerr Model (possible anomaly).")
else:
  print("Shannon H1 Entropy distribution matches the Kerr Model (no significant deviation).")

#below is for the L1

print(f"KS Statistic: {ks_stat_l1_shannon}, P-Value: {p_value_l1_shannon}")

if p_value_l1_shannon < 0.05:
  print("Shannon L1 Entropy distribution is significantly different from Kerr Model (possible anomaly).")
else:
  print("Shannon L1 Entropy distribution matches the Kerr Model (no significant deviation).")

#FOR RENYI
print(f"KS Statistic: {ks_stat_h1_renyi}, P-Value: {p_value_h1_renyi}")

if p_value_h1_renyi < 0.05:
  print("Renyi H1 Entropy distribution is significantly different from Kerr Model (possible anomaly).")
else:
  print("Renyi H1 Entropy distribution matches the Kerr Model (no significant deviation).")

#below is for the L1

print(f"KS Statistic: {ks_stat_l1_renyi}, P-Value: {p_value_l1_renyi}")

if p_value_l1_renyi < 0.05:
  print("Renyi L1 Entropy distribution is significantly different from Kerr Model (possible anomaly).")
else:
  print("Renyi L1 Entropy distribution matches the Kerr Model (no significant deviation).")

#FOR TSALLIS
print(f"KS Statistic: {ks_stat_h1_tsallis}, P-Value: {p_value_h1_tsallis}")

if p_value_h1_tsallis < 0.05:
  print("Tsallis H1 Entropy distribution is significantly different from Kerr Model (possible anomaly).")
else:
  print("Tsallis H1 Entropy distribution matches the Kerr Model (no significant deviation).")

#below is for the L1

print(f"KS Statistic: {ks_stat_l1_tsallis}, P-Value: {p_value_l1_tsallis}")

if p_value_l1_tsallis < 0.05:
  print("Tsallis L1 Entropy distribution is significantly different from Kerr Model (possible anomaly).")
else:
  print("Tsallis L1 Entropy distribution matches the Kerr Model (no significant deviation).")


#save our results
with h5py.File("processed/ks_test_results.hdf5", "w") as f:
  f.create_dataset("ks_stat_h1_shannon", data=ks_stat_h1_shannon)
  f.create_dataset("ks_stat_l1_shannon", data=ks_stat_l1_shannon)
  f.create_dataset("ks_stat_h1_renyi", data=ks_stat_h1_renyi)
  f.create_dataset("ks_stat_l1_renyi", data=ks_stat_l1_renyi)
  f.create_dataset("ks_stat_h1_tsallis", data=ks_stat_h1_tsallis)
  f.create_dataset("ks_stat_l1_tsallis", data=ks_stat_l1_tsallis)
  f.create_dataset("p_value_h1_shannon", data=p_value_h1_shannon)
  f.create_dataset("p_value_l1_shannon", data=p_value_l1_shannon)
  f.create_dataset("p_value_h1_renyi", data=p_value_h1_renyi)
  f.create_dataset("p_value_l1_renyi", data=p_value_l1_renyi)
  f.create_dataset("p_value_h1_tsallis", data=p_value_h1_tsallis)
  f.create_dataset("p_value_l1_tsallis", data=p_value_l1_tsallis)

print("KS Test Results saved to HDF5 for future reference.")
exit()
