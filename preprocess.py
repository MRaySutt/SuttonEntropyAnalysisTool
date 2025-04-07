import os
import matplotlib.pyplot as plt
from gwpy.signal import filter_design
import numpy as np
from gwpy.timeseries import TimeSeries
import time
import threading
import h5py
import random

#This just lets us know that everything is functioning. It is my attempt at a loading screen. 
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
h1_quiet_path = os.path.join(processed_folder, f"{event_id}_h1_quiet.hdf5")
l1_quiet_path = os.path.join(processed_folder, f"{event_id}_l1_quiet.hdf5")
l1_data_path = os.path.join(processed_folder, f"{event_id}_l1_data.hdf5")
h1_data_path = os.path.join(processed_folder, f"{event_id}_h1_data.hdf5")

for file in [h1_quiet_path, l1_quiet_path, h1_data_path, l1_data_path]:
    check_file(file)

#read the data
h1_quiet = TimeSeries.read(h1_quiet_path)
l1_quiet = TimeSeries.read(l1_quiet_path)
h1_data = TimeSeries.read(h1_data_path)
l1_data = TimeSeries.read(l1_data_path)

#Bandpass filter (20 - 500 Hz)
h1_filtered = h1_data.bandpass(20, 500)
l1_filtered = l1_data.bandpass(20, 500)

h1_quiet_filtered = h1_quiet.bandpass(20, 500)
l1_quiet_filtered = l1_quiet.bandpass(20, 500)

#notch filter for known instrumental noise
h1_filtered = h1_filtered.notch(60)
h1_filtered = h1_filtered.notch(120)
h1_filtered = h1_filtered.notch(180)

l1_filtered = l1_filtered.notch(60)
l1_filtered = l1_filtered.notch(120)
l1_filtered = l1_filtered.notch(180)

h1_quiet_filtered = h1_quiet_filtered.notch(60)
h1_quiet_filtered = h1_quiet_filtered.notch(120)
h1_quiet_filtered = h1_quiet_filtered.notch(180)

#L1 Pre-Event Notching
l1_quiet_filtered = l1_quiet_filtered.notch(60)
l1_quiet_filtered = l1_quiet_filtered.notch(120)
l1_quiet_filtered = l1_quiet_filtered.notch(180)

#now whiten
h1_whitened = h1_filtered.whiten()
l1_whitened = l1_filtered.whiten()

h1_quiet_whitened = h1_quiet_filtered.whiten()
l1_quiet_whitened = l1_quiet_filtered.whiten()

psd_h1 = h1_filtered.psd()
psd_l1 = l1_filtered.psd()
psd_h1_whitened = h1_whitened.psd()
psd_l1_whitened = l1_whitened.psd()

h1_whitened.write(os.path.join(processed_folder, f"{event_id}_h1_whitened.hdf5"), format="hdf5")
l1_whitened.write(os.path.join(processed_folder, f"{event_id}_l1_whitened.hdf5"), format="hdf5")
h1_quiet_whitened.write(os.path.join(processed_folder, f"{event_id}_h1_quiet_whitened.hdf5"), format="hdf5")
l1_quiet_whitened.write(os.path.join(processed_folder, f"{event_id}_l1_quiet_whitened.hdf5"), format="hdf5")

#plot power spectral density for QC before whitening
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

axs[0].plot(np.array(psd_h1.frequencies.value), np.array(psd_h1.value), label="H1 Filtered", color="blue")
axs[0].plot(np.array(psd_l1.frequencies.value), np.array(psd_h1.value), label="L1 Filtered", color="red")
axs[0].set_title("Power Spectral Density (Post filter)")
axs[0].legend(loc = "upper left", bbox_to_anchor=(1, 1))

#plot PSD again after whitening
axs[1].plot(np.array(psd_h1_whitened.frequencies.value), np.array(psd_h1_whitened.value), label="H1 Whitened", color="blue")
axs[1].plot(np.array(psd_l1_whitened.frequencies.value), np.array(psd_l1_whitened.value), label="L1 Whitened", color="red")
axs[1].set_title("Power Spectral Density (Whitened Data)")
axs[1].legend(loc = "upper left", bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()

print("We may be done here! ---------------- Yes!")

print("Data successfully stored")
exit()
