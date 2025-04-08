import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
import numpy as np
import h5py
import time
import threading
import os
import random
from pathlib import Path

print("executing script now.")

#simulate fetching large data for error checker 
def progress_indicator():
    messages = ["Still processing... we are working on it!", "No issues yet!", "Not much longer!", "Still trying!", "We are rolling!", "Now we are cooking!", "Organizing!", "Sorting haystack... checking for needle!",
    ]
    while True:
        time.sleep(5)
        print(random.choice(messages))
progress_thread = threading.Thread(target=progress_indicator, daemon=True)
progress_thread.start()

#event specifics
event_id = "GW150914" #change this and gps time for each new event
gps_time = 1126259462
duration = 40 #seconds
sample_rate = 4096 #Hz

#output folder
user_home = str(Path.home())
processed_folder = os.path.join(user_home, "SEAT_processed")
os.makedirs(processed_folder, exist_ok=True)

print(f"Going to save to {processed_folder}")

with open(os.path.join(processed_folder, "event_id.txt"), "w") as f:
    f.write(event_id)

np.save("sample_rate.npy", sample_rate)

print("Saved Event ID. We are analyzing event ", event_id, ".")

#Load 40 seconds of data centered on the event from L1 and H1 
l1_data = TimeSeries.fetch_open_data('L1', gps_time - 20, gps_time + 20)

h1_data = TimeSeries.fetch_open_data('H1', gps_time - 20, gps_time + 20)

print("LOOKED AT H1 and L1 EVENT")

quiet_time_gps = 1126259462 - 1800 #30 minutes before event #MAKE SURE YOU CHANGE THIS FOR NEW EVENT
h1_quiet = TimeSeries.fetch_open_data('H1', quiet_time_gps - 20, quiet_time_gps + 20)
l1_quiet = TimeSeries.fetch_open_data('L1', quiet_time_gps - 20, quiet_time_gps + 20)

print("we have the data now.")


#save your data
h1_data.write(os.path.join(processed_folder, f"{event_id}_h1_data.hdf5"), format="hdf5")
l1_data.write(os.path.join(processed_folder, f"{event_id}_l1_data.hdf5"), format="hdf5")
h1_quiet.write(os.path.join(processed_folder, f"{event_id}_h1_quiet.hdf5"), format="hdf5")
l1_quiet.write(os.path.join(processed_folder, f"{event_id}_l1_quiet.hdf5"), format="hdf5")

#plot comparison for sanity check
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axs[0].plot(h1_quiet.times, h1_quiet.value, label="H1 Quiet Data", color="red")
axs[0].plot(h1_data.times, h1_data.value, label="H1 Event Data", color="blue")
axs[0].set_title("H1 Detector Strain (Event vs. Quiet)")
axs[0].legend()

axs[1].plot(l1_quiet.times, l1_quiet.value, label="L1 Quiet Data", color="red")
axs[1].plot(l1_data.times, l1_data.value, label="L1 Event Data", color="blue")
axs[1].set_title("L1 Detector Strain (Event vs. Quiet)")
axs[1].legend()

plt.xlabel("Time (s)")
plt.tight_layout()
plt.show()

print("Plot successful.")

#to add new event (for example GW150914)
#gps_time = 1126259462

print("Script successfully completed.")
exit()
