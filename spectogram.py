import os
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
import numpy as np
import time
import threading
import h5py
import random
from pathlib import Path

print("It's going!")

 
def progress_indicator():
    messages = ["Still processing... we are working on it!", "No issues yet!", "Processing the process.", "Probably working on it still.", "Calculating calculations.", "Now we are cooking!", "Sorting haystack... checking for needle""
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

print(f"{event_id}")
print(f"{base_dir}")

#load the data
h1_whitened_path = os.path.join(processed_folder, f"{event_id}_h1_whitened.hdf5")
l1_whitened_path = os.path.join(processed_folder, f"{event_id}_l1_whitened.hdf5")
h1_quiet_whitened_path = os.path.join(processed_folder, f"{event_id}_h1_quiet_whitened.hdf5")
l1_quiet_whitened_path = os.path.join(processed_folder, f"{event_id}_l1_quiet_whitened.hdf5")

#read the data
h1_data = TimeSeries.read(h1_whitened_path)
h1_quiet = TimeSeries.read(h1_quiet_whitened_path)

l1_data = TimeSeries.read(l1_whitened_path)
l1_quiet = TimeSeries.read(l1_quiet_whitened_path)


#plot comparison for sanity check
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axs[0].plot(h1_data.times, h1_data.value, label="H1 Event Data", color="blue")
axs[0].plot(h1_quiet.times, h1_quiet.value, label="H1 Quiet Data", color="red")
axs[0].set_title("H1 Detector Strain (Event vs. Quiet)")
axs[0].legend()

axs[1].plot(l1_data.times, l1_data.value, label="L1 Event Data", color="blue")
axs[1].plot(l1_quiet.times, l1_quiet.value, label="L1 Quiet Data", color="red")
axs[1].set_title("L1 Detector Strain (Event vs. Quiet)")
axs[1].legend()

plt.xlabel("Time (s)")
plt.tight_layout()
plt.show()

exit()
