import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, windows
from gwpy.timeseries import TimeSeries
import h5py
import time
import threading
import h5py
import random
from pathlib import Path

print("It's going!")

#THIS SECTION IS OPTIONAL DUE TO COMPUTATIONAL LOAD. THERE MAY BE SOME BUGS PRESENT. PLEASE CONTACT ME @ mattsutton9@yahoo.com and I will attempt to fix them right away

#simulate fetching large data for error checker 
def progress_indicator():
    messages = ["Still processing... we are working on it!", "No issues yet!", "Processing the process.", "Probably working on it still.", "Calculating calculations with math and things.", "Now we are cooking!", "Sorting haystack... checking for needle"
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

#load the data
h1_whitened_path = os.path.join(processed_folder, f"{event_id}_h1_whitened.hdf5")
l1_whitened_path = os.path.join(processed_folder, f"{event_id}_l1_whitened.hdf5")


#read the data
h1_whitened = TimeSeries.read(h1_whitened_path)
l1_whitened = TimeSeries.read(l1_whitened_path)


signal_pairs = [
    ("Strain", h1_whitened, l1_whitened, "blue", "red"),
    ("Shannon", shannon_h1, shannon_l1, "green", "orange"),
    ("Renyi", renyi_h1, renyi_l1, "purple", "brown"),
    ("Tsallis", tsallis_h1, tsallis_l1, "cyan", "magenta")
]

def run_echo_analysis(signal1, signal2, label, color):
    min_len = min(len(signal1), len(signal2))
    signal1 = signal1[:min_len]
    signal2 = signal2[:min_len]
    hann_window = windows.hann(len(signal1))
    windowed_1 = signal1 * hann_window
    windowed_2 = signal2 * hann_window
    auto_corr = correlate(windowed_1, windowed_2, mode="full")
    lags = np.arange(-(len(signal1)) + 1, len(signal1))
    n_perm = 1000
    random_autocorr = np.zeros((n_perm, len(auto_corr)))
    for i in range(n_perm):
        np.random.shuffle(signal1)
        np.random.shuffle(signal2)
        shuffled_corr = correlate(signal1, signal2, mode="full")
        if len(shuffled_corr) == random_autocorr.shape[1]:
            random_autocorr[i, :] = shuffled_corr
        else:
            #if there is ever a mismatch, truncate or pad as needed
            min_len = min(len(shuffled_corr), random_autocorr.shape[1])
            random_autocorr[i, :min_len] = shuffled_corr[:min_len]
            if min_len < random_autocorr.shape[1]:
                random_autocorr[i, min_len:] = 0 #zero padding
    mean_autocorr = np.mean(random_autocorr, axis=0)
    std_autocorr = np.std(random_autocorr, axis=0)
    plt.plot(lags, auto_corr, label=f"{label} Autocorrelation", linestyle="--", color=color)
    plt.fill_between(lags, mean_autocorr - 2*std_autocorr, mean_autocorr + 2*std_autocorr, color=color, alpha=0.2)
    return {
        "label" : label,
        "lags" : lags,
        "auto_corr" : auto_corr,
        "mean_autocorr": mean_autocorr,
        "std_autocorr": std_autocorr
    }

results = []

for label, signal1, signal2, c1, c2 in signal_pairs:
    result = run_echo_analysis(signal1, signal2, label, c1)
    results.append(result)



#save em! 
with h5py.File(os.path.join(processed_folder, "quantum_echo_results.hdf5"), "w", driver="core") as f:
    for res in results:
        grp = f.create_group(res["label"])
        grp.create_dataset("lags", data=res["lags"])
        grp.create_dataset("auto_corr", data=res["auto_corr"])
        grp.create_dataset("mean_autocorr", data=res["mean_autocorr"])
        grp.create_dataset("std_autocorr", data=res["std_autocorr"])

print("Quantum Echo Analysis saved to HDF5.")
exit()
