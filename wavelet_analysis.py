import os
import pywt
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import numpy as np
import h5py
import time
import threading
import h5py
import random
from pathlib import Path

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

duration = 40 #seconds

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

print(f"{event_id}")
print(f"{base_dir}")


entropy_hdf5_path = os.path.abspath(os.path.join(processed_folder.strip(), "entropy_results.hdf5"))

#load stored entropy values
with h5py.File(entropy_hdf5_path, "r") as f:
    entropy_h1_time_shannon = f["entropy_h1_time_shannon"][:].flatten()
    entropy_l1_time_shannon = f["entropy_l1_time_shannon"][:].flatten()
    entropy_quiet_h1_time_shannon = f["entropy_quiet_h1_time_shannon"][:].flatten()
    entropy_quiet_l1_time_shannon = f["entropy_quiet_l1_time_shannon"][:].flatten()
    entropy_h1_time_renyi = f["entropy_h1_time_renyi"][:].flatten()
    entropy_l1_time_renyi = f["entropy_l1_time_renyi"][:].flatten()
    entropy_quiet_h1_time_renyi = f["entropy_quiet_h1_time_renyi"][:].flatten()   
    entropy_quiet_l1_time_renyi = f["entropy_quiet_l1_time_renyi"][:].flatten()
    entropy_h1_time_tsallis = f["entropy_h1_time_tsallis"][:].flatten()
    entropy_l1_time_tsallis = f["entropy_l1_time_tsallis"][:].flatten()
    entropy_quiet_h1_time_tsallis = f["entropy_quiet_h1_time_tsallis"][:].flatten()
    entropy_quiet_l1_time_tsallis = f["entropy_quiet_l1_time_tsallis"][:].flatten()


fs = 4096
freqs_of_interest = np.linspace(20, 500, num=100)
#convert frequency range to wavelet scales
scales = pywt.central_frequency('cmor1.5-1.0') * fs / freqs_of_interest

print("H1 Event Shannon Entropy:", entropy_h1_time_shannon[:10])
print("H1 Quiet Shannon Entropy::", entropy_quiet_h1_time_shannon[:10]) 


#Compute wavelet transform
cwt_result_h1_s, frequencies_h1_s = pywt.cwt(entropy_h1_time_shannon, scales, 'cmor1.5-1.0', sampling_period=1/fs)
cwt_result_l1_s, frequencies_l1_s = pywt.cwt(entropy_l1_time_shannon, scales, 'cmor1.5-1.0', sampling_period=1/fs)

cwt_result_h1_quiet_s, frequencies_h1_quiet_s = pywt.cwt(entropy_quiet_h1_time_shannon, scales, 'cmor1.5-1.0', sampling_period=1/fs)
cwt_result_l1_quiet_s, frequencies_l1_quiet_s = pywt.cwt(entropy_quiet_l1_time_shannon, scales, 'cmor1.5-1.0', sampling_period=1/fs)


#renyi
cwt_result_h1_r, frequencies_h1_r = pywt.cwt(entropy_h1_time_renyi, scales, 'cmor1.5-1.0', sampling_period=1/fs)
cwt_result_l1_r, frequencies_l1_r = pywt.cwt(entropy_l1_time_renyi, scales, 'cmor1.5-1.0', sampling_period=1/fs)

cwt_result_h1_quiet_r, frequencies_h1_quiet_r = pywt.cwt(entropy_quiet_h1_time_renyi, scales, 'cmor1.5-1.0', sampling_period=1/fs)
cwt_result_l1_quiet_r, frequencies_l1_quiet_r = pywt.cwt(entropy_quiet_l1_time_renyi, scales, 'cmor1.5-1.0', sampling_period=1/fs)

#tsallis
cwt_result_h1_t, frequencies_h1_t = pywt.cwt(entropy_h1_time_tsallis, scales, 'cmor1.5-1.0', sampling_period=1/fs)
cwt_result_l1_t, frequencies_l1_t = pywt.cwt(entropy_l1_time_tsallis, scales, 'cmor1.5-1.0', sampling_period=1/fs)

cwt_result_h1_quiet_t, frequencies_h1_quiet_t = pywt.cwt(entropy_quiet_h1_time_tsallis, scales, 'cmor1.5-1.0', sampling_period=1/fs)
cwt_result_l1_quiet_t, frequencies_l1_quiet_t = pywt.cwt(entropy_quiet_l1_time_tsallis, scales, 'cmor1.5-1.0', sampling_period=1/fs)


print("H1 Event Max:", np.max(cwt_result_h1_s))
print("H1 Quiet Max:", np.max(cwt_result_h1_quiet_s))


#Plot H1 Shannon
fig1, ax1 = plt.subplots(1, 2, figsize=(14, 6))

ax1[0].imshow(np.abs(cwt_result_h1_s), aspect='auto', extent= [0, cwt_result_h1_s.shape[1], frequencies_h1_s[-1], frequencies_h1_s[0]], origin='lower', cmap='magma')
ax1[0].set_title("Shannon H1 Event Scalogram")

ax1[1].imshow(np.abs(cwt_result_h1_quiet_s), aspect='auto', extent= [0, cwt_result_h1_quiet_s.shape[1], frequencies_h1_quiet_s[-1], frequencies_h1_quiet_s[0]], origin='lower', cmap='magma')
ax1[1].set_title("Shannon H1 Quiet Scalogram")

plt.tight_layout()
plt.show()

#then again for L1 Shannon

fig2, ax2 = plt.subplots(1, 2, figsize=(14, 6))

ax2[0].imshow(np.abs(cwt_result_l1_s), aspect='auto', extent=[0, cwt_result_l1_s.shape[1], frequencies_l1_s[-1], frequencies_l1_s[0]], origin='lower', cmap='magma')
ax2[0].set_title("Shannon L1 Event Scalogram")

ax2[1].imshow(np.abs(cwt_result_l1_quiet_s), aspect='auto', extent=[0, cwt_result_l1_quiet_s.shape[1], frequencies_l1_quiet_s[-1], frequencies_l1_quiet_s[0]], origin='lower', cmap='magma')
ax2[1].set_title("Shannon L1 Quiet Scalogram")

plt.tight_layout()
plt.show()

#------------------------PLOT RENYI
fig3, ax3 = plt.subplots(1, 2, figsize=(14, 6))

ax3[0].imshow(np.abs(cwt_result_h1_r), aspect='auto', extent= [0, cwt_result_h1_r.shape[1], frequencies_h1_r[-1], frequencies_h1_r[0]], origin='lower', cmap='magma')
ax3[0].set_title("Renyi H1 Event Scalogram")

ax3[1].imshow(np.abs(cwt_result_h1_quiet_r), aspect='auto', extent= [0, cwt_result_h1_quiet_r.shape[1], frequencies_h1_quiet_r[-1], frequencies_h1_quiet_r[0]], origin='lower', cmap='magma')
ax3[1].set_title("Renyi H1 Quiet Scalogram")

plt.tight_layout()
plt.show()

#then again for L1 Renyi

fig4, ax4 = plt.subplots(1, 2, figsize=(14, 6))

ax4[0].imshow(np.abs(cwt_result_l1_r), aspect='auto', extent=[0, cwt_result_l1_r.shape[1], frequencies_l1_r[-1], frequencies_l1_r[0]], origin='lower', cmap='magma')
ax4[0].set_title("Renyi L1 Event Scalogram")

ax4[1].imshow(np.abs(cwt_result_l1_quiet_r), aspect='auto', extent=[0, cwt_result_l1_quiet_r.shape[1], frequencies_l1_quiet_r[-1], frequencies_l1_quiet_r[0]], origin='lower', cmap='magma')
ax4[1].set_title("Renyi L1 Quiet Scalogram")

plt.tight_layout()
plt.show()

#------------------------PLOT Tsallis
fig5, ax5 = plt.subplots(1, 2, figsize=(14, 6))

ax5[0].imshow(np.abs(cwt_result_h1_t), aspect='auto', extent= [0, cwt_result_h1_t.shape[1], frequencies_h1_t[-1], frequencies_h1_t[0]], origin='lower', cmap='magma')
ax5[0].set_title("Tsallis H1 Event Scalogram")

ax5[1].imshow(np.abs(cwt_result_h1_quiet_t), aspect='auto', extent= [0, cwt_result_h1_quiet_t.shape[1], frequencies_h1_quiet_t[-1], frequencies_h1_quiet_t[0]], origin='lower', cmap='magma')
ax5[1].set_title("Tsallis H1 Quiet Scalogram")

plt.tight_layout()
plt.show()

#then again for Tsallis

fig6, ax6 = plt.subplots(1, 2, figsize=(14, 6))

ax6[0].imshow(np.abs(cwt_result_l1_t), aspect='auto', extent=[0, cwt_result_l1_t.shape[1], frequencies_l1_t[-1], frequencies_l1_t[0]], origin='lower', cmap='magma')
ax6[0].set_title("Tsallis L1 Event Scalogram")

ax6[1].imshow(np.abs(cwt_result_l1_quiet_s), aspect='auto', extent=[0, cwt_result_l1_quiet_s.shape[1], frequencies_l1_quiet_s[-1], frequencies_l1_quiet_s[0]], origin='lower', cmap='magma')
ax6[1].set_title("Tsallis L1 Quiet Scalogram")

plt.tight_layout()
plt.show()



#save your results
with h5py.File(os.path.join(processed_folder, "wavelet_results.hdf5"), "w", driver="core") as f::
  f.create_dataset("cwt_h1_s", data=cwt_result_h1_s)
  f.create_dataset("cwt_l1_s", data=cwt_result_l1_s)
  f.create_dataset("cwt_quiet_h1_s", data=cwt_result_h1_quiet_s)
  f.create_dataset("cwt_quiet_l1_s", data=cwt_result_l1_quiet_s)
  f.create_dataset("cwt_h1_r", data=cwt_result_h1_r)
  f.create_dataset("cwt_l1_r", data=cwt_result_l1_r)
  f.create_dataset("cwt_quiet_h1_r", data=cwt_result_h1_quiet_r)
  f.create_dataset("cwt_quiet_l1_r", data=cwt_result_l1_quiet_r)
  f.create_dataset("cwt_h1_t", data=cwt_result_h1_t)
  f.create_dataset("cwt_l1_t", data=cwt_result_l1_t)
  f.create_dataset("cwt_quiet_h1_t", data=cwt_result_h1_quiet_t)
  f.create_dataset("cwt_quiet_l1_t", data=cwt_result_l1_quiet_t)


print("Wavelet Transform results saved.")
print("Data successfully stored")
exit()
