import os
from scipy.signal import welch
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
import h5py
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

#LOAD EM IN
entropy_hdf5_path = os.path.abspath(os.path.join(processed_folder.strip(), "entropy_results.hdf5"))

with h5py.File(entropy_hdf5_path, "r") as f:
    shannon_h1 = f["entropy_h1_time_shannon"][:]
    shannon_l1 = f["entropy_l1_time_shannon"][:]
    renyi_h1 = f["entropy_h1_time_renyi"][:]
    renyi_l1 = f["entropy_l1_time_renyi"][:]
    tsallis_h1 = f["entropy_h1_time_tsallis"][:]
    tsallis_l1 = f["entropy_l1_time_tsallis"][:]
    quiet_h1_shannon = f["entropy_quiet_h1_time_shannon"][:]
    quiet_l1_shannon = f["entropy_quiet_l1_time_shannon"][:]
    quiet_h1_renyi = f["entropy_quiet_h1_time_renyi"][:]
    quiet_l1_renyi = f["entropy_quiet_l1_time_renyi"][:]
    quiet_h1_tsallis = f["entropy_quiet_h1_time_tsallis"][:]
    quiet_l1_tsallis = f["entropy_quiet_l1_time_tsallis"][:]


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

#lIGO sample rate
sample_rate = 4096 #Hz

#set it up
shannon_h1 = shannon_h1.flatten()
shannon_l1 = shannon_l1.flatten()
renyi_h1 = renyi_h1.flatten()
renyi_l1 = renyi_l1.flatten()
tsallis_h1 = tsallis_h1.flatten()
tsallis_l1 = tsallis_l1.flatten()
quiet_h1_shannon = quiet_h1_shannon.flatten()
quiet_l1_shannon = quiet_l1_shannon.flatten()
quiet_h1_renyi = quiet_h1_renyi.flatten()
quiet_l1_renyi = quiet_l1_renyi.flatten()
quiet_h1_tsallis = quiet_h1_tsallis.flatten()
quiet_l1_tsallis = quiet_l1_tsallis.flatten()

#Define post-ringdown region (last 5 seconds of event data)
post_ringdown_h1 = np.array(h1_whitened.data)[-int(5 * sample_rate):]
post_ringdown_l1 = np.array(l1_whitened.data)[-int(5 * sample_rate):]
#do it for entropy too. Let's see if we can uncover anything weird
p_r_shan_h1 = np.array(shannon_h1.data)[-int(5 * sample_rate):]
p_r_shan_l1 = np.array(shannon_l1.data)[-int(5 * sample_rate):]
p_r_ren_h1 = np.array(renyi_h1.data)[-int(5 * sample_rate):]
p_r_ren_l1 = np.array(renyi_l1.data)[-int(5 * sample_rate):]
p_r_tsa_h1 = np.array(tsallis_h1.data)[-int(5 * sample_rate):]
p_r_tsa_l1 = np.array(tsallis_l1.data)[-int(5 * sample_rate):]

#define quiet data region (5 seconds before event)
quiet_region_h1 = np.array(h1_quiet_whitened.data)[-int(5 * sample_rate):]
quiet_region_l1 = np.array(l1_quiet_whitened.data)[-int(5 * sample_rate):]

#quiet for each entropy type too 
quiet_shan_h1 = np.array(quiet_h1_shannon.data)[-int(5 * sample_rate):]
quiet_shan_l1 = np.array(quiet_l1_shannon.data)[-int(5 * sample_rate):]
quiet_ren_h1 = np.array(quiet_h1_renyi.data)[-int(5 * sample_rate):]
quiet_ren_l1 = np.array(quiet_l1_renyi.data)[-int(5 * sample_rate):]
quiet_tsa_h1 = np.array(quiet_h1_tsallis.data)[-int(5 * sample_rate):]
quiet_tsa_l1 = np.array(quiet_l1_tsallis.data)[-int(5 * sample_rate):]

#compute mean residual strain
mean_residual_h1 = np.mean(post_ringdown_h1)
mean_residual_l1 = np.mean(post_ringdown_l1)

shan_mean_residual_h1 = np.mean(p_r_shan_h1)
shan_mean_residual_l1 = np.mean(p_r_shan_l1)

ren_mean_residual_h1 = np.mean(p_r_ren_h1)
ren_mean_residual_l1 = np.mean(p_r_ren_l1)

tsa_mean_residual_h1 = np.mean(p_r_tsa_h1)
tsa_mean_residual_l1 = np.mean(p_r_tsa_l1)

print(f"Mean Residual Strain (H1): {mean_residual_h1}")
print(f"Mean Residual Strain (L1): {mean_residual_l1}")
print(f"Shannon Mean Residual Strain (H1): {shan_mean_residual_h1}")
print(f"Shannon Mean Residual Strain (L1): {shan_mean_residual_l1}")
print(f"Renyi Mean Residual Strain (H1): {ren_mean_residual_h1}")
print(f"Renyi Mean Residual Strain (L1): {ren_mean_residual_l1}")
print(f"Tsallis Mean Residual Strain (H1): {tsa_mean_residual_h1}")
print(f"Tsallis Mean Residual Strain (L1): {tsa_mean_residual_l1}")

#compute power spectral density using Welch's method
nperseg = 1024 #fft segment length for spectral resolution

#post-ringdown PSD
f_h1, psd_h1 = welch(post_ringdown_h1, fs=sample_rate, nperseg=nperseg)
f_l1, psd_l1 = welch(post_ringdown_l1, fs=sample_rate, nperseg=nperseg)
#now for our entropy types
f_shan_h1, psd_shan_h1 = welch(p_r_shan_h1, fs=sample_rate, nperseg=nperseg)
f_shan_l1, psd_shan_l1 = welch(p_r_shan_l1, fs=sample_rate, nperseg=nperseg)

f_ren_h1, psd_ren_h1 = welch(p_r_ren_h1, fs=sample_rate, nperseg=nperseg)
f_ren_l1, psd_ren_l1 = welch(p_r_ren_l1, fs=sample_rate, nperseg=nperseg)

f_tsa_h1, psd_tsa_h1 = welch(p_r_tsa_h1, fs=sample_rate, nperseg=nperseg)
f_tsa_l1, psd_tsa_l1 = welch(p_r_tsa_l1, fs=sample_rate, nperseg=nperseg)

#quiet region PSD
f_h1_pre, psd_h1_pre = welch(quiet_region_h1, fs=sample_rate, nperseg=nperseg)
f_l1_pre, psd_l1_pre = welch(quiet_region_l1, fs=sample_rate, nperseg=nperseg)
#quiets for the entropy types
f_shan_h1_pre, psd_shan_h1_pre = welch(quiet_shan_h1, fs=sample_rate, nperseg=nperseg)
f_shan_l1_pre, psd_shan_l1_pre = welch(quiet_shan_l1, fs=sample_rate, nperseg=nperseg)

f_ren_h1_pre, psd_ren_h1_pre = welch(quiet_ren_h1, fs=sample_rate, nperseg=nperseg)
f_ren_l1_pre, psd_ren_l1_pre = welch(quiet_ren_l1, fs=sample_rate, nperseg=nperseg)

f_tsa_h1_pre, psd_tsa_h1_pre = welch(quiet_tsa_h1, fs=sample_rate, nperseg=nperseg)
f_tsa_l1_pre, psd_tsa_l1_pre = welch(quiet_tsa_l1, fs=sample_rate, nperseg=nperseg)
#plot PSD comparisons
plt.figure(figsize=(10, 5))
plt.semilogy(f_h1, psd_h1, label="Post-Ringdown H1", color="blue")
plt.semilogy(f_l1, psd_l1, label="Post-Ringdown L1", color="red", linestyle="--")
plt.semilogy(f_h1_pre, psd_h1_pre, label="Pre-Event (Quiet) H1", color="blue", linestyle="dotted")
plt.semilogy(f_l1_pre, psd_l1_pre, label="Pre-Event (Quiet) L1", color="red", linestyle="dotted")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density")
plt.title("PSD of Post-Ringdown Residuals vs. Quiet Data")
plt.legend()
plt.show()

#Shannon
plt.figure(figsize=(10, 5))
plt.semilogy(f_shan_h1, psd_shan_h1, label="Post-Ringdown H1", color="blue")
plt.semilogy(f_shan_l1, psd_shan_l1, label="Post-Ringdown L1", color="red", linestyle="--")
plt.semilogy(f_shan_h1_pre, psd_shan_h1_pre, label="Pre-Event (Quiet) H1", color="blue", linestyle="dotted")
plt.semilogy(f_shan_l1_pre, psd_shan_l1_pre, label="Pre-Event (Quiet) L1", color="red", linestyle="dotted")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD of Entropy Signal")
plt.title("Shannon Entropy: PSD of Post-Ringdown Residuals vs. Quiet Data")
plt.legend()
plt.show()
#Renyi
plt.figure(figsize=(10, 5))
plt.semilogy(f_ren_h1, psd_ren_h1, label="Post-Ringdown H1", color="blue")
plt.semilogy(f_ren_l1, psd_ren_l1, label="Post-Ringdown L1", color="red", linestyle="--")
plt.semilogy(f_ren_h1_pre, psd_ren_h1_pre, label="Pre-Event (Quiet) H1", color="blue", linestyle="dotted")
plt.semilogy(f_ren_l1_pre, psd_ren_l1_pre, label="Pre-Event (Quiet) L1", color="red", linestyle="dotted")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD of Entropy Signal")
plt.title("Renyi Entropy: PSD of Post-Ringdown Residuals vs. Quiet Data")
plt.legend()
plt.show()
#Tsallis
plt.figure(figsize=(10, 5))
plt.semilogy(f_tsa_h1, psd_tsa_h1, label="Post-Ringdown H1", color="blue")
plt.semilogy(f_tsa_l1, psd_tsa_l1, label="Post-Ringdown L1", color="red", linestyle="--")
plt.semilogy(f_tsa_h1_pre, psd_tsa_h1_pre, label="Pre-Event (Quiet) H1", color="blue", linestyle="dotted")
plt.semilogy(f_tsa_l1_pre, psd_tsa_l1_pre, label="Pre-Event (Quiet) L1", color="red", linestyle="dotted")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD of Entropy Signal")
plt.title("Tsallis Entropy: PSD of Post-Ringdown Residuals vs. Quiet Data")
plt.legend()
plt.show()


#compute statistical deviations between post-ringdown and quiet data 
psd_ratio_h1 = psd_h1 / (psd_h1_pre + 1e-20)
psd_ratio_l1 = psd_l1 / (psd_l1_pre + 1e-20)

psd_shan_ratio_h1 = psd_shan_h1 / (psd_shan_h1_pre + 1e-20)
psd_shan_ratio_l1 = psd_shan_l1 / (psd_shan_l1_pre + 1e-20)

psd_ren_ratio_h1 = psd_ren_h1 / (psd_ren_h1_pre + 1e-20)
psd_ren_ratio_l1 = psd_ren_l1 / (psd_ren_l1_pre + 1e-20)

psd_tsa_ratio_h1 = psd_tsa_h1 / (psd_tsa_h1_pre + 1e-20)
psd_tsa_ratio_l1 = psd_tsa_l1 / (psd_tsa_l1_pre + 1e-20)

#compute the mean PSD deviation
mean_psd_dev_h1 = np.mean(psd_ratio_h1)
mean_psd_dev_l1 = np.mean(psd_ratio_l1)

mean_shan_psd_dev_h1 = np.mean(psd_shan_ratio_h1)
mean_shan_psd_dev_l1 = np.mean(psd_shan_ratio_l1)

mean_ren_psd_dev_h1 = np.mean(psd_ren_ratio_h1)
mean_ren_psd_dev_l1 = np.mean(psd_ren_ratio_l1)

mean_tsa_psd_dev_h1 = np.mean(psd_tsa_ratio_h1)
mean_tsa_psd_dev_l1 = np.mean(psd_tsa_ratio_l1)

print(f"Mean PSD ratio (H1 Post/Pre): {mean_psd_dev_h1}")
print(f"Mean PSD ratio (L1 Post/Pre): {mean_psd_dev_l1}")

print(f"Shannon Entropy Mean PSD ratio (H1 Post/Pre): {mean_shan_psd_dev_h1}")
print(f"Shannon Entropy Mean PSD ratio (L1 Post/Pre): {mean_shan_psd_dev_l1}")

print(f"Renyi Entropy Mean PSD ratio (H1 Post/Pre): {mean_ren_psd_dev_h1}")
print(f"Renyi Entropy Mean PSD ratio (L1 Post/Pre): {mean_ren_psd_dev_l1}")

print(f"Tsallis Entropy Mean PSD ratio (H1 Post/Pre): {mean_tsa_psd_dev_h1}")
print(f"Tsallis Entropy Mean PSD ratio (L1 Post/Pre): {mean_tsa_psd_dev_l1}")

#save those results!! 
with h5py.File("processed/soft_hair_results.hdf5", "w") as f:
  f.create_dataset("mean_residual_h1", data=mean_residual_h1)
  f.create_dataset("mean_residual_l1", data=mean_residual_l1)
  f.create_dataset("shan_mean_residual_h1", data=shan_mean_residual_h1)
  f.create_dataset("shan_mean_residual_l1", data=shan_mean_residual_l1)
  f.create_dataset("ren_mean_residual_h1", data=ren_mean_residual_h1)
  f.create_dataset("ren_mean_residual_l1", data=ren_mean_residual_l1)
  f.create_dataset("tsa_mean_residual_h1", data=tsa_mean_residual_h1)
  f.create_dataset("tsa_mean_residual_l1", data=tsa_mean_residual_l1)
  f.create_dataset("psd_ratio_h1", data=psd_ratio_h1)
  f.create_dataset("psd_ratio_l1", data=psd_ratio_l1)
  f.create_dataset("psd_shan_ratio_h1", data=psd_shan_ratio_h1)
  f.create_dataset("psd_shan_ratio_l1", data=psd_shan_ratio_l1)
  f.create_dataset("psd_ren_ratio_h1", data=psd_ren_ratio_h1)
  f.create_dataset("psd_ren_ratio_l1", data=psd_ren_ratio_l1)
  f.create_dataset("psd_tsa_ratio_h1", data=psd_tsa_ratio_h1)
  f.create_dataset("psd_tsa_ratio_l1", data=psd_tsa_ratio_l1)
  f.create_dataset("mean_psd_dev_h1", data=mean_psd_dev_h1)
  f.create_dataset("mean_psd_dev_l1", data=mean_psd_dev_l1)
  f.create_dataset("mean_shan_psd_dev_h1", data=mean_shan_psd_dev_h1)
  f.create_dataset("mean_shan_psd_dev_l1", data=mean_shan_psd_dev_l1)
  f.create_dataset("mean_ren_psd_dev_h1", data=mean_ren_psd_dev_h1)
  f.create_dataset("mean_ren_psd_dev_l1", data=mean_ren_psd_dev_l1)
  f.create_dataset("mean_tsa_psd_dev_h1", data=mean_tsa_psd_dev_h1)
  f.create_dataset("mean_tsa_psd_dev_l1", data=mean_tsa_psd_dev_l1)
  f.create_dataset("post_ringdown_h1", data=post_ringdown_h1)
  f.create_dataset("post_ringdown_l1", data=post_ringdown_l1)
  f.create_dataset("p_r_shan_h1", data=p_r_shan_h1)
  f.create_dataset("p_r_shan_l1", data=p_r_shan_l1)
  f.create_dataset("p_r_ren_h1", data=p_r_ren_h1)
  f.create_dataset("p_r_ren_l1", data=p_r_ren_l1)
  f.create_dataset("p_r_tsa_h1", data=p_r_tsa_h1)
  f.create_dataset("p_r_tsa_l1", data=p_r_tsa_l1)
  f.create_dataset("quiet_region_h1", data=post_ringdown_h1)
  f.create_dataset("quiet_region_l1", data=post_ringdown_l1)
  f.create_dataset("quiet_shan_h1", data=quiet_shan_h1)
  f.create_dataset("quiet_shan_l1", data=quiet_shan_l1)
  f.create_dataset("quiet_ren_h1", data=quiet_ren_h1)
  f.create_dataset("quiet_ren_l1", data=quiet_ren_l1)
  f.create_dataset("quiet_tsa_h1", data=quiet_tsa_h1)
  f.create_dataset("quiet_tsa_l1", data=quiet_tsa_l1)

print("Soft Hair Analysis Results Saved! Hooray!")

exit()
