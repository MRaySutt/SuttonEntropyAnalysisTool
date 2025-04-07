import os
import matplotlib.pyplot as plt
from scipy.stats import entropy
import numpy as np
import h5py
from gwpy.timeseries import TimeSeries
import time
import threading
import h5py
import random

print("It's going!")

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
processed_folder = os.path.join(base_dir, "processed")

def check_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

#Load Event ID
event_id_path = os.path.join(processed_folder, "event_id.txt")
check_file(event_id_path)
with open(event_id_path, "r") as f:
    event_id = f.read().strip()

#load sample rate 
sample_rate = 4096 #Hz

print("sorting files... sorta")


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


#Define entropy functions (Shannon, Renyi, Tsallis) 
def shannon_entropy(signal):
  hist, _ = np.histogram(signal, bins=100, density=True)
  hist = hist[hist > 0] #remove zero value so no log(0)
  return entropy(hist, base=2)

def renyi_entropy(signal, alpha=0.5):
  hist, _ = np.histogram(signal, bins=100, density=True)
  hist = hist[hist > 0]
  hist_sum = np.sum(hist ** alpha)
  hist_sum = max(hist_sum, 1e-10)
  return 1 / (1 - alpha) * np.log2(hist_sum)

def tsallis_entropy(signal, q=2):
  hist, _ = np.histogram(signal, bins=100, density=True)
  hist = hist[hist > 0]
  hist_sum = np.sum(hist ** q)
  if hist_sum > 1:
      hist_sum = 1
  return (1 - hist_sum) / (q - 1)
 

#compute entropy for event and quiet data
entropy_h1 = shannon_entropy(h1_whitened.value)
entropy_l1 = shannon_entropy(l1_whitened.value)
entropy_renyi_h1 = renyi_entropy(h1_whitened.value)
entropy_renyi_l1 = renyi_entropy(l1_whitened.value)
entropy_tsallis_h1 = tsallis_entropy(h1_whitened.value)
entropy_tsallis_l1 = tsallis_entropy(l1_whitened.value)


entropy_quiet_h1 = shannon_entropy(h1_quiet_whitened.value)
entropy_quiet_l1 = shannon_entropy(l1_quiet_whitened.value)
entropy_renyi_quiet_h1 = renyi_entropy(h1_quiet_whitened.value)
entropy_renyi_quiet_l1 = renyi_entropy(l1_quiet_whitened.value)
entropy_tsallis_quiet_h1 = tsallis_entropy(h1_quiet_whitened.value)
entropy_tsallis_quiet_l1 = tsallis_entropy(l1_quiet_whitened.value)

#compute entropy over time
def sliding_entropy_shannon(signal, window_size=1000, step_size=500):
    entropies = []
    for i in range(0, len(signal) - window_size + 1, step_size):
        window = signal[i:i + window_size]
        entropy_val = shannon_entropy(window)
        entropies.append(entropy_val)
    return np.array(entropies)

def sliding_entropy_renyi(signal, window_size=1000, step_size=500):
    entropies = []
    for i in range(0, len(signal) - window_size + 1, step_size):
        window = signal[i:i + window_size]
        entropy_val = renyi_entropy(window)
        entropies.append(entropy_val)
    return np.array(entropies)

def sliding_entropy_tsallis(signal, window_size=1000, step_size=500):
    entropies = []
    for i in range(0, len(signal) - window_size + 1, step_size):
        window = signal[i:i + window_size]
        entropy_val = tsallis_entropy(window)
        entropies.append(entropy_val)
    return np.array(entropies)

entropy_h1_time_shannon = sliding_entropy_shannon(h1_whitened.value)
entropy_l1_time_shannon = sliding_entropy_shannon(l1_whitened.value) 
entropy_quiet_l1_time_shannon = sliding_entropy_shannon(l1_quiet_whitened.value)
entropy_quiet_h1_time_shannon = sliding_entropy_shannon(h1_quiet_whitened.value)

entropy_h1_time_renyi = sliding_entropy_renyi(h1_whitened.value)
entropy_l1_time_renyi = sliding_entropy_renyi(l1_whitened.value) 
entropy_quiet_l1_time_renyi = sliding_entropy_renyi(l1_quiet_whitened.value)
entropy_quiet_h1_time_renyi = sliding_entropy_renyi(h1_quiet_whitened.value)

entropy_h1_time_tsallis = sliding_entropy_tsallis(h1_whitened.value)
entropy_l1_time_tsallis = sliding_entropy_tsallis(l1_whitened.value) 
entropy_quiet_l1_time_tsallis = sliding_entropy_tsallis(l1_quiet_whitened.value)
entropy_quiet_h1_time_tsallis = sliding_entropy_tsallis(h1_quiet_whitened.value)



entropy_file_path = os.path.join(processed_folder, "entropy_results.hdf5")

with h5py.File(entropy_file_path, "w") as f:
    f.create_dataset("entropy_h1", data=np.array([entropy_h1]))
    f.create_dataset("entropy_l1", data=np.array([entropy_l1]))
    f.create_dataset("entropy_renyi_h1", data=np.array([entropy_renyi_h1]))
    f.create_dataset("entropy_renyi_l1", data=np.array([entropy_renyi_l1]))
    f.create_dataset("entropy_tsallis_h1", data=np.array([entropy_tsallis_h1]))
    f.create_dataset("entropy_tsallis_l1", data=np.array([entropy_tsallis_l1]))
    f.create_dataset("entropy_quiet_h1", data=np.array([entropy_quiet_h1]))
    f.create_dataset("entropy_quiet_l1", data=np.array([entropy_quiet_l1]))
    f.create_dataset("entropy_renyi_quiet_h1", data=np.array([entropy_renyi_quiet_h1]))
    f.create_dataset("entropy_renyi_quiet_l1", data=np.array([entropy_renyi_quiet_l1]))
    f.create_dataset("entropy_tsallis_quiet_h1", data=np.array([entropy_tsallis_quiet_h1]))
    f.create_dataset("entropy_tsallis_quiet_l1", data=np.array([entropy_tsallis_quiet_l1]))
    f.create_dataset("entropy_h1_time_shannon", data=np.array([entropy_h1_time_shannon]))
    f.create_dataset("entropy_l1_time_shannon", data=np.array([entropy_l1_time_shannon]))
    f.create_dataset("entropy_quiet_l1_time_shannon", data=np.array([entropy_quiet_l1_time_shannon]))
    f.create_dataset("entropy_quiet_h1_time_shannon", data=np.array([entropy_quiet_h1_time_shannon]))
    f.create_dataset("entropy_h1_time_tsallis", data=np.array([entropy_h1_time_tsallis]))
    f.create_dataset("entropy_l1_time_tsallis", data=np.array([entropy_l1_time_tsallis]))
    f.create_dataset("entropy_quiet_l1_time_tsallis", data=np.array([entropy_quiet_l1_time_tsallis]))
    f.create_dataset("entropy_quiet_h1_time_tsallis", data=np.array([entropy_quiet_h1_time_tsallis]))
    f.create_dataset("entropy_h1_time_renyi", data=np.array([entropy_h1_time_renyi]))
    f.create_dataset("entropy_l1_time_renyi", data=np.array([entropy_l1_time_renyi]))
    f.create_dataset("entropy_quiet_l1_time_renyi", data=np.array([entropy_quiet_l1_time_renyi]))
    f.create_dataset("entropy_quiet_h1_time_renyi", data=np.array([entropy_quiet_h1_time_renyi]))
    f.create_dataset("h1_whitened", data=np.array([h1_whitened]))
    f.create_dataset("l1_whitened", data=np.array([l1_whitened]))
    f.create_dataset("h1_quiet_whitened", data=np.array([h1_quiet_whitened]))
    f.create_dataset("l1_quiet_whitened", data=np.array([l1_quiet_whitened]))

    
#plot event vs quiet entropy evolution
plt.figure(figsize=(12,4))
plt.plot(entropy_h1_time_shannon, label="H1 Shannon Entropy", color="blue")
plt.plot(entropy_l1_time_shannon, label="L1 Shannon Entropy", color="red")
plt.plot(entropy_quiet_h1_time_shannon, label="H1 Quiet Shannon", linestyle="dashed", color="cyan")
plt.plot(entropy_quiet_l1_time_shannon, label="L1 Quiet Shannon", linestyle="dashed", color="orange")
plt.title("Shannon Entropy Over Time: Event vs Quiet Data")
plt.xlabel("Time Window")
plt.ylabel("Entropy")
plt.legend()
plt.tight_layout()
plt.savefig(f"processed/{event_id}_entropy_shannon.png")
plt.close()

#RENYI
plt.figure(figsize=(12,4))
plt.plot(entropy_h1_time_renyi, label="H1 Renyi Entropy", color="blue")
plt.plot(entropy_l1_time_renyi, label="L1 Renyi Entropy", color="red")
plt.plot(entropy_quiet_h1_time_renyi, label="H1 Quiet Renyi", linestyle="dashed", color="cyan")
plt.plot(entropy_quiet_l1_time_renyi, label="L1 Quiet Renyi", linestyle="dashed", color="orange")
plt.title("Renyi Entropy Evolution: Event vs Quiet Data")
plt.xlabel("Time Window")
plt.ylabel("Entropy")
plt.legend()
plt.tight_layout()
plt.savefig(f"processed/{event_id}_entropy_renyi.png")
plt.close()

#TSALLIS
plt.figure(figsize=(12,4))
plt.plot(entropy_h1_time_tsallis, label="H1 Tsallis Entropy", color="blue")
plt.plot(entropy_l1_time_tsallis, label="L1 Tsallis Entropy", color="red")
plt.plot(entropy_quiet_h1_time_tsallis, label="H1 Quiet Tsallis", linestyle="dashed", color="cyan")
plt.plot(entropy_quiet_l1_time_tsallis, label="L1 Quiet Tsallis", linestyle="dashed", color="orange")
plt.title("Tsallis Entropy Evolution: Event vs Quiet Data")
plt.xlabel("Time Window")
plt.ylabel("Entropy")
plt.legend()
plt.tight_layout()
plt.savefig(f"processed/{event_id}_entropy_tsallis.png")
plt.close()

print("Entropy Results saved to HDF5. Noice!")

print("Data successfully stored")
exit()
