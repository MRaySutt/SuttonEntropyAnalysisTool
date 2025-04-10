import os
import numpy as np
import h5py
from scipy.stats import entropy
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

h1_whitened_path = os.path.join(processed_folder, f"{event_id}_h1_whitened.hdf5")
l1_whitened_path = os.path.join(processed_folder, f"{event_id}_l1_whitened.hdf5")
h1_quiet_whitened_path = os.path.join(processed_folder, f"{event_id}_h1_quiet_whitened.hdf5")
l1_quiet_whitened_path = os.path.join(processed_folder, f"{event_id}_l1_quiet_whitened.hdf5")

#read the data
h1_whitened = TimeSeries.read(h1_whitened_path)
l1_whitened = TimeSeries.read(l1_whitened_path)
h1_quiet_whitened = TimeSeries.read(h1_quiet_whitened_path)
l1_quiet_whitened = TimeSeries.read(l1_quiet_whitened_path)



entropy_hdf5_path = os.path.abspath(os.path.join(processed_folder.strip(), "entropy_results.hdf5"))

#load stored entropy values
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

shannon_quiet_entropy_h1 = entropy_quiet_h1_time_shannon
shannon_quiet_entropy_l1 = entropy_quiet_l1_time_shannon
renyi_quiet_entropy_h1 = entropy_quiet_h1_time_renyi
renyi_quiet_entropy_l1 = entropy_quiet_l1_time_renyi
tsallis_quiet_entropy_h1 = entropy_quiet_h1_time_tsallis
tsallis_quiet_entropy_l1 = entropy_quiet_l1_time_tsallis

shannon_entropy_h1 = entropy_h1_time_shannon
shannon_entropy_l1 = entropy_l1_time_shannon
renyi_entropy_h1 = entropy_h1_time_renyi
renyi_entropy_l1 = entropy_l1_time_renyi
tsallis_entropy_h1 = entropy_h1_time_tsallis
tsallis_entropy_l1 = entropy_l1_time_tsallis

#monte carlo parameters
N_SIMULATIONS = 1000 #Number of Kerr Ringdown simulations 


f_mode_range = (100, 300)

def simulate_kerr_ringdown(fs, duration=1, f_mode=(100, 300), damping_time=0.1):
  #Simulates a damped sinusoidal kerr ringdown signal
  t = np.linspace(0, duration, int(fs * duration))
  f_mode = np.random.uniform(*f_mode_range)
  phase = np.random.uniform(0, 2 * np.pi) #random phase
  amplitude = np.random.normal(1.0, 0.05) #amplitude variations 
  noise = np.random.normal(0, 0.02, len(t))
  signal_kerr = (amplitude * np.exp(-t / damping_time) * np.sin(2 * np.pi * f_mode * t + phase)) + noise
  return signal_kerr


#Inject real detector noise into simulated kerr signals 
def inject_noise_shannon(simulated_signal, real_whitened_data, detector):
  if detector == "H1":
      real_noise = h1_quiet_whitened
  elif detector == "L1":
      real_noise = l1_quiet_whitened
  else:
      raise ValueError("Detector must be 'H1' or 'L1'")
  #injects real detector noise into the Kerr signal 
  start_idx = np.random.randint(0, len(real_whitened_data) - len(simulated_signal))
  noise_segment = real_noise[start_idx:start_idx + len(simulated_signal)]
  return simulated_signal + noise_segment


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
  hist, _ = np.histogram(signal, bins=100, density=False)
  hist = hist[hist > 0]
  prob = hist / np.sum(hist)
  if q == 1:
      #Tsallis reduces to shannon if q = 1
      return -np.sum(prob * np.log(prob))
  else: 
      return (1 - np.sum(prob ** q)) / (q - 1)


#set up our entropy sliders again
def sliding_entropy_shannon(signal, window_size=1000, step_size=500):
    entropies = []
    for i in range(0, len(signal) - window_size + 1, step_size):
        window = signal[i:i + window_size]
        entropy_val = shannon_entropy(window.value)
        entropies.append(entropy_val)
    return np.array(entropies)

def sliding_entropy_renyi(signal, window_size=1000, step_size=500):
    entropies = []
    for i in range(0, len(signal) - window_size + 1, step_size):
        window = signal[i:i + window_size]
        entropy_val = renyi_entropy(window.value)
        entropies.append(entropy_val)
    return np.array(entropies)

def sliding_entropy_tsallis(signal, window_size=1000, step_size=500):
    entropies = []
    for i in range(0, len(signal) - window_size + 1, step_size):
        window = signal[i:i + window_size]
        entropy_val = tsallis_entropy(window.value)
        entropies.append(entropy_val)
    return np.array(entropies)

#run the monte carlo sim
simulated_entropies_h1_shannon = []
simulated_entropies_l1_shannon = []
simulated_entropies_h1_renyi = []
simulated_entropies_l1_renyi = []
simulated_entropies_h1_tsallis = []
simulated_entropies_l1_tsallis = []

#storing results before np.mean 
sim_entropy_h1_shannon = []
sim_entropy_l1_shannon = []
sim_entropy_h1_renyi = []
sim_entropy_l1_renyi = []
sim_entropy_h1_tsallis = []
sim_entropy_l1_tsallis = []

for _ in range(N_SIMULATIONS):
    sim_kerr_h1 = simulate_kerr_ringdown(fs=4096)
    sim_kerr_l1 = simulate_kerr_ringdown(fs=4096)
    #inject real noise into Kerr
    noisy_sim_h1 = inject_noise_shannon(sim_kerr_h1, h1_quiet_whitened, "H1")
    noisy_sim_l1 = inject_noise_shannon(sim_kerr_l1, l1_quiet_whitened, "L1")
    #Calculate Entropy
    entropy_h1_sim_shannon = sliding_entropy_shannon(noisy_sim_h1)
    entropy_l1_sim_shannon = sliding_entropy_shannon(noisy_sim_l1)
    entropy_h1_sim_renyi = sliding_entropy_renyi(noisy_sim_h1)
    entropy_l1_sim_renyi = sliding_entropy_renyi(noisy_sim_l1)
    entropy_h1_sim_tsallis = sliding_entropy_tsallis(noisy_sim_h1)
    entropy_l1_sim_tsallis = sliding_entropy_tsallis(noisy_sim_l1)
    #store results later analysis
    sim_entropy_h1_shannon.append(entropy_h1_sim_shannon)
    sim_entropy_l1_shannon.append(entropy_l1_sim_shannon)
    sim_entropy_h1_renyi.append(entropy_h1_sim_renyi)
    sim_entropy_l1_renyi.append(entropy_l1_sim_renyi)
    sim_entropy_h1_tsallis.append(entropy_h1_sim_tsallis)
    sim_entropy_l1_tsallis.append(entropy_l1_sim_tsallis)
    #store results for monte carlo
    simulated_entropies_h1_shannon.append(np.mean(entropy_h1_sim_shannon))
    simulated_entropies_l1_shannon.append(np.mean(entropy_l1_sim_shannon))
    simulated_entropies_h1_renyi.append(np.mean(entropy_h1_sim_renyi))
    simulated_entropies_l1_renyi.append(np.mean(entropy_l1_sim_renyi))
    simulated_entropies_h1_tsallis.append(np.mean(entropy_h1_sim_tsallis))
    simulated_entropies_l1_tsallis.append(np.mean(entropy_l1_sim_tsallis))

#make 'em numpier 
sim_entropy_h1_shannon = np.array(sim_entropy_h1_shannon)
sim_entropy_l1_shannon = np.array(sim_entropy_l1_shannon)

sim_entropy_h1_renyi = np.array(sim_entropy_h1_renyi)
sim_entropy_l1_renyi = np.array(sim_entropy_l1_renyi)

sim_entropy_h1_tsallis = np.array(sim_entropy_h1_tsallis)
sim_entropy_l1_tsallis = np.array(sim_entropy_l1_tsallis)



#event mean entropies
shannon_mean_h1 = np.mean(shannon_entropy_h1)
shannon_mean_l1 = np.mean(shannon_entropy_l1)

renyi_mean_h1 = np.mean(renyi_entropy_h1)
renyi_mean_l1 = np.mean(renyi_entropy_l1)

tsallis_mean_h1 = np.mean(tsallis_entropy_h1)
tsallis_mean_l1 = np.mean(tsallis_entropy_l1)

#P-value time 
p_value_h1_shannon = (np.sum(simulated_entropies_h1_shannon) >= (shannon_mean_h1)) / N_SIMULATIONS
p_value_l1_shannon = (np.sum(simulated_entropies_l1_shannon) >= (shannon_mean_l1)) / N_SIMULATIONS

p_value_h1_renyi = (np.sum(simulated_entropies_h1_renyi) >= (renyi_mean_h1)) / N_SIMULATIONS
p_value_l1_renyi = (np.sum(simulated_entropies_l1_renyi) >= (renyi_mean_l1)) / N_SIMULATIONS

p_value_h1_tsallis = (np.sum(simulated_entropies_h1_tsallis) >= (tsallis_mean_h1)) / N_SIMULATIONS
p_value_l1_tsallis = (np.sum(simulated_entropies_l1_tsallis) >= (tsallis_mean_l1)) / N_SIMULATIONS

print(f"Mean Shannon Entropy (H1): {np.mean(shannon_entropy_h1)}")
print(f"Mean Shannon Entropy (L1): {np.mean(shannon_entropy_l1)}")
print(f"Mean Simulated Shannon Entropy (H1): {np.mean(simulated_entropies_h1_shannon)}")
print(f"Mean Simulated Shannon Entropy (L1): {np.mean(simulated_entropies_l1_shannon)}")
print(f"P-Value for H1 Entropy Shannon: {p_value_h1_shannon}")
print(f"P-Value for L1 Entropy Shannon: {p_value_l1_shannon}")

print("------------------------------------")

print(f"Mean Renyi Entropy (H1): {np.mean(renyi_entropy_h1)}")
print(f"Mean Renyi Entropy (L1): {np.mean(renyi_entropy_l1)}")
print(f"Mean Simulated Renyi Entropy (H1): {np.mean(simulated_entropies_h1_renyi)}")
print(f"Mean Simulated Renyi Entropy (L1): {np.mean(simulated_entropies_l1_renyi)}")
print(f"P-Value for H1 Entropy Renyi: {p_value_h1_renyi}")
print(f"P-Value for L1 Entropy Renyi: {p_value_l1_renyi}")

print("------------------------------------")
print(f"Mean Tsallis Entropy (H1): {np.mean(tsallis_entropy_h1)}")
print(f"Mean Tsallis Entropy (L1): {np.mean(tsallis_entropy_l1)}")
print(f"Mean Simulated Tsallis Entropy (H1): {np.mean(simulated_entropies_h1_tsallis)}")
print(f"Mean Simulated Tsallis Entropy (L1): {np.mean(simulated_entropies_l1_tsallis)}")
print(f"P-Value for H1 Entropy Tsallis: {p_value_h1_tsallis}")
print(f"P-Value for L1 Entropy Tsallis: {p_value_l1_tsallis}")

#save your monte carlo results
with h5py.File(os.path.join(processed_folder, "monte_carlo_results.hdf5"), "w", driver="core") as f:
    f.create_dataset("simulated_entropies_h1_shannon", data=simulated_entropies_h1_shannon)
    f.create_dataset("simulated_entropies_l1_shannon", data=simulated_entropies_l1_shannon)
    f.create_dataset("simulated_entropies_h1_renyi", data=simulated_entropies_h1_renyi)
    f.create_dataset("simulated_entropies_l1_renyi", data=simulated_entropies_l1_renyi)
    f.create_dataset("simulated_entropies_h1_tsallis", data=simulated_entropies_h1_tsallis)
    f.create_dataset("simulated_entropies_l1_tsallis", data=simulated_entropies_l1_tsallis)
    f.create_dataset("p_value_h1_shannon", data=p_value_h1_shannon)
    f.create_dataset("p_value_l1_shannon", data=p_value_l1_shannon)
    f.create_dataset("p_value_h1_renyi", data=p_value_h1_renyi)
    f.create_dataset("p_value_l1_renyi", data=p_value_l1_renyi)
    f.create_dataset("p_value_h1_tsallis", data=p_value_h1_tsallis)
    f.create_dataset("p_value_l1_tsallis", data=p_value_l1_tsallis)
    f.create_dataset("sim_entropy_h1_shannon", data=sim_entropy_h1_shannon)
    f.create_dataset("sim_entropy_l1_shannon", data=sim_entropy_l1_shannon)
    f.create_dataset("sim_entropy_h1_renyi", data=sim_entropy_h1_renyi)
    f.create_dataset("sim_entropy_l1_renyi", data=sim_entropy_l1_renyi)
    f.create_dataset("sim_entropy_h1_tsallis", data=sim_entropy_h1_tsallis)
    f.create_dataset("sim_entropy_l1_tsallis", data=sim_entropy_l1_tsallis)



print("Monte Carlo results saved.")
exit()
