import os
import numpy as np
from ripser import Rips
import matplotlib.pyplot as plt
import h5py
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction import image
from gwpy.timeseries import TimeSeries
import time
import threading
import random
from persim import plot_diagrams
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

#normalize entropy data (important for topological analysis)
scaler = MinMaxScaler()
shannon_h1_scaled = scaler.fit_transform(shannon_h1.reshape(-1,1))
shannon_l1_scaled = scaler.fit_transform(shannon_l1.reshape(-1,1))

renyi_h1_scaled = scaler.fit_transform(renyi_h1.reshape(-1,1))
renyi_l1_scaled = scaler.fit_transform(renyi_l1.reshape(-1,1))

tsallis_h1_scaled = scaler.fit_transform(tsallis_h1.reshape(-1,1))
tsallis_l1_scaled = scaler.fit_transform(tsallis_l1.reshape(-1,1))

def sliding_window(data, window=5):
    return np.stack([data[i:i+window] for i in range(len(data) - window)])

embedded_h1_shannon = sliding_window(shannon_h1_scaled.flatten(), 5)
embedded_l1_shannon = sliding_window(shannon_l1_scaled.flatten(), 5)

embedded_h1_renyi = sliding_window(renyi_h1_scaled.flatten(), 5)
embedded_l1_renyi = sliding_window(renyi_l1_scaled.flatten(), 5)

embedded_h1_tsallis = sliding_window(tsallis_h1_scaled.flatten(), 5)
embedded_l1_tsallis = sliding_window(tsallis_l1_scaled.flatten(), 5)

#compute persistent homology
rips = Rips()
diagram_h1_shannon = rips.fit_transform(embedded_h1_shannon)
diagram_l1_shannon = rips.fit_transform(embedded_l1_shannon)

diagram_h1_renyi = rips.fit_transform(embedded_h1_renyi)
diagram_l1_renyi = rips.fit_transform(embedded_l1_renyi)

diagram_h1_tsallis = rips.fit_transform(embedded_h1_tsallis)
diagram_l1_tsallis = rips.fit_transform(embedded_l1_tsallis)


print("Persistent Homology Summary")
print(f"H1 Diagram (Shannon): H0: {diagram_h1_shannon[0].shape}, H1: {diagram_h1_shannon[1].shape}, L1 Diagram (Shannon): H0: {diagram_l1_shannon[0].shape}, H1: {diagram_l1_shannon[1].shape}")

print(f"H1 Diagram (Renyi): H0: {diagram_h1_renyi[0].shape}, H1: {diagram_h1_renyi[1].shape}, L1 Diagram (Renyi): H0: {diagram_l1_renyi[0].shape}, H1: {diagram_l1_renyi[1].shape}")

print(f"H1 Diagram (Tsallis): {diagram_h1_tsallis[0].shape}, H1: {diagram_h1_tsallis[1].shape},, L1 Diagram (Tsallis): H0: {diagram_l1_tsallis[0].shape}, H1: {diagram_l1_tsallis[1].shape}")


#print(f"L1 Diagram (H0): {diagram_h1[0].shape}, L1 Diagram (H1): {diagram_l1[1].shape}")
#print(f"L1 Diagram (H0): {diagram_h1[0].shape}, L1 Diagram (H1): {diagram_l1[1].shape}")


fig, axs = plt.subplots(1, 6, figsize=(15, 10))
rips.plot(diagram_h1_shannon[1], ax=axs[0])
axs[0].set_title("H1 Diagram (Shannon)")
rips.plot(diagram_l1_shannon[1], ax=axs[1])
axs[1].set_title("L1 Diagram (Shannon)")
rips.plot(diagram_h1_renyi[1], ax=axs[2])
axs[2].set_title("H1 Diagram (Renyi)")
rips.plot(diagram_l1_renyi[1], ax=axs[3])
axs[3].set_title("L1 Diagram (Renyi)")
if diagram_h1_tsallis[1].size > 0:
    rips.plot(diagram_h1_tsallis[1], ax=axs[4])
    axs[4].set_title("H1 Diagram (Tsallis)")
else:
    axs[4].set_title("Tsallis H1 (Empty)")
if diagram_l1_tsallis[1].size > 0:
    rips.plot(diagram_l1_tsallis[1], ax=axs[5])
    axs[5].set_title("L1 Diagram (Tsallis)")
else:
    axs[5].set_title("L1 Diagram (Empty)")
plt.tight_layout()
plt.show()

#save data of course
with h5py.File(os.path.join(processed_folder, "persistent_homology_results.hdf5"), "w", driver="core") as f:
  f.create_dataset("diagram_h1_H0_shannon", data=diagram_h1_shannon[0])
  f.create_dataset("diagram_h1_H1_shannon", data=diagram_h1_shannon[1])
  f.create_dataset("diagram_l1_L0_shannon", data=diagram_l1_shannon[0])
  f.create_dataset("diagram_l1_L1_shannon", data=diagram_l1_shannon[1])
  f.create_dataset("diagram_h1_H0_renyi", data=diagram_h1_renyi[0])
  f.create_dataset("diagram_h1_H1_renyi", data=diagram_h1_renyi[1])
  f.create_dataset("diagram_l1_L0_renyi", data=diagram_l1_renyi[0])
  f.create_dataset("diagram_l1_L1_renyi", data=diagram_l1_renyi[1])
  f.create_dataset("diagram_h1_H0_tsallis", data=diagram_h1_tsallis[0])
  f.create_dataset("diagram_h1_H1_tsallis", data=diagram_h1_tsallis[1])
  f.create_dataset("diagram_l1_L0_tsallis", data=diagram_l1_tsallis[0])
  f.create_dataset("diagram_l1_L1_tsallis", data=diagram_l1_tsallis[1])

print("Persistent homology results saved to HDF5 file for future reference.")
exit()
