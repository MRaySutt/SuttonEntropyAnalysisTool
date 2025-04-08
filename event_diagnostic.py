import os
from scipy.stats import ks_2samp, ttest_ind, entropy
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
import h5py
import json
import time
import threading
import random
import sys
from pathlib import Path

base_dir = os.getcwd()
processed_folder = os.path.join(str(Path.home()), "SEAT_processed")
processed_folder = processed_folder.strip().replace("\r", "").replace("\n", "")

#Load Event ID
event_id_path = os.path.join(processed_folder, "event_id.txt")
with open(event_id_path, "r") as f:
    event_id = f.read().strip()

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, message):
        for s in self.streams:
            s.write(message)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

log_file = open(f"{event_id}_event_diagnostic_output.txt", "w")
sys.stdout = Tee(sys.stdout, log_file)


print(event_id)
print("Sample Rate: 4096 Hz")
print("--------EVENT DIAGNOSTIC--------")

#ENTROPY RESULTS LOADING

entropy_hdf5_path = os.path.abspath(os.path.join(processed_folder.strip(), "entropy_results.hdf5"))

#load stored entropy values
with h5py.File(entropy_hdf5_path, "r") as f:
    entropy_shannon_h1 = f["entropy_h1"][:]
    entropy_renyi_h1 = f["entropy_renyi_h1"][:]
    entropy_tsallis_h1 = f["entropy_tsallis_h1"][:]

with h5py.File(entropy_hdf5_path, "r") as f:    
    entropy_shannon_l1 = f["entropy_l1"][:]
    entropy_renyi_l1 = f["entropy_renyi_l1"][:]
    entropy_tsallis_l1 = f["entropy_tsallis_l1"][:]

with h5py.File(entropy_hdf5_path, "r") as f:
    shannon_quiet_h1 = f["entropy_quiet_h1"][:]
    renyi_quiet_h1 = f["entropy_renyi_quiet_h1"][:]
    tsallis_quiet_h1 = f["entropy_tsallis_quiet_h1"][:]

with h5py.File(entropy_hdf5_path, "r") as f:    
    shannon_quiet_l1 = f["entropy_quiet_l1"][:]
    renyi_quiet_l1 = f["entropy_renyi_quiet_l1"][:]
    tsallis_quiet_l1 = f["entropy_tsallis_quiet_l1"][:]


print("Entropy values for ", event_id, " are listed below:")

print(f"Entropy H1 (Shannon): {entropy_shannon_h1}")
print(f"Entropy H1 (Renyi): {entropy_renyi_h1}")
print(f"Entropy H1 (Tsallis): {entropy_tsallis_h1}")
print(f"Entropy Quiet H1 (Shannon): {shannon_quiet_h1}")
print(f"Entropy Quiet H1 (Renyi): {renyi_quiet_h1}")
print(f"Entropy Quiet H1 (Tsallis): {tsallis_quiet_h1}")
print("-----------------L1------------------")
print(f"Entropy L1 (Shannon): {entropy_shannon_l1}")
print(f"Entropy L1 (Renyi): {entropy_renyi_l1}")
print(f"Entropy L1 (Tsallis): {entropy_tsallis_l1}")
print(f"Entropy Quiet L1 (Shannon): {shannon_quiet_l1}")
print(f"Entropy Quiet L1 (Renyi): {renyi_quiet_l1}")
print(f"Entropy Quiet L1 (Tsallis): {tsallis_quiet_l1}")

print("This section calculates entropy over post-ringdown and quiet windows to detect structure or informational deviation.")

print("------------------------------------")

#Monte Carlo for the Doctor 
mc_path = os.path.join(processed_folder.strip(), "monte_carlo_results.hdf5")

with h5py.File(mc_path, "r") as f:
    shan_sim_h1 = f["simulated_entropies_h1_shannon"][:]
    shan_sim_l1 = f["simulated_entropies_l1_shannon"][:]
    ren_sim_h1 = f["simulated_entropies_h1_renyi"][:]
    ren_sim_l1 = f["simulated_entropies_l1_renyi"][:]
    tsa_sim_h1 = f["simulated_entropies_h1_tsallis"][:]
    tsa_sim_l1 = f["simulated_entropies_l1_tsallis"][:]
    mc_shan_p_value_h1 = f["p_value_h1_shannon"][()]
    mc_shan_p_value_l1 = f["p_value_l1_shannon"][()]
    mc_ren_p_value_h1 = f["p_value_h1_renyi"][()]
    mc_ren_p_value_l1 = f["p_value_l1_renyi"][()]
    mc_tsa_p_value_h1 = f["p_value_h1_tsallis"][()]
    mc_tsa_p_value_l1 = f["p_value_l1_tsallis"][()]

print("MONTE CARLO ANALYSIS")
print("This section we inject synthetic entropy data into quiet baselines for p-value distribution and significance.")
print(f"Simulated Entropies Shannon H1: {shan_sim_h1}")
print(f"Simulated Entropies Shannon L1: {shan_sim_l1}")
print(f"Shannon P-Value H1: {mc_shan_p_value_h1}")
print(f"Shannon P-Value L1: {mc_shan_p_value_l1}")
print("-------------------------")
print(f"Simulated Entropies Renyi H1: {ren_sim_h1}")
print(f"Simulated Entropies Renyi L1: {ren_sim_l1}")
print(f"Renyi P-Value H1: {mc_ren_p_value_h1}")
print(f"Renyi P-Value L1: {mc_ren_p_value_l1}")
print("-------------------------")
print(f"Simulated Entropies Tsallis H1: {tsa_sim_h1}")
print(f"Simulated Entropies Tsallis L1: {tsa_sim_l1}")
print(f"Tsallis P-Value H1: {mc_tsa_p_value_h1}")
print(f"Tsallis P-Value L1: {mc_tsa_p_value_l1}")

print("------------------------------------")
#SOFT HAIR RESULTS LOADING
 
soft_hdf5_path = os.path.abspath(os.path.join(processed_folder.strip(), "soft_hair_results.hdf5"))

#load soft hair mem results
with h5py.File(soft_hdf5_path, "r") as f:
    post_ringdown_h1 = f["post_ringdown_h1"][:]
    post_ringdown_l1 = f["post_ringdown_l1"][:]
    p_r_shan_h1 = f["p_r_shan_h1"][:]
    p_r_shan_l1 = f["p_r_shan_l1"][:]
    p_r_ren_h1 = f["p_r_ren_h1"][:]
    p_r_ren_l1 = f["p_r_ren_l1"][:]
    p_r_tsa_h1 = f["p_r_tsa_h1"][:]
    p_r_tsa_l1 = f["p_r_tsa_l1"][:]
    psd_ratio_h1 = f["psd_ratio_h1"][:]
    psd_ratio_l1 = f["psd_ratio_l1"][:]
    psd_shan_ratio_h1 = f["psd_shan_ratio_h1"][:]
    psd_shan_ratio_l1 = f["psd_shan_ratio_l1"][:]
    psd_ren_ratio_h1 = f["psd_ren_ratio_h1"][:]
    psd_ren_ratio_l1 = f["psd_ren_ratio_l1"][:]
    psd_tsa_ratio_h1 = f["psd_tsa_ratio_h1"][:]
    psd_tsa_ratio_l1 = f["psd_tsa_ratio_l1"][:]
    mean_psd_dev_h1 = f["mean_psd_dev_h1"][()]
    mean_psd_dev_l1 = f["mean_psd_dev_l1"][()]
    mean_shan_psd_dev_h1 = f["mean_shan_psd_dev_h1"][()]
    mean_shan_psd_dev_l1 = f["mean_shan_psd_dev_l1"][()]
    mean_ren_psd_dev_h1 = f["mean_ren_psd_dev_h1"][()]
    mean_ren_psd_dev_l1 = f["mean_ren_psd_dev_l1"][()]
    mean_tsa_psd_dev_h1 = f["mean_tsa_psd_dev_h1"][()]
    mean_tsa_psd_dev_l1 = f["mean_tsa_psd_dev_l1"][()]
print("Soft Hair Analysis:")
print("This section computes entropy change post-ringdown to investigate 'soft hair' quantum structure at the horizon.")
print("Soft Hair Memory Results are the following:")
print(f"Post-Ringdown H1: {post_ringdown_h1}")
print(f"Post-Ringdown L1: {post_ringdown_l1}")
print(f"PSD Ratio H1: {psd_ratio_h1}")
print(f"PSD Ratio L1: {psd_ratio_l1}")
print(f"Mean PSD Deviation H1: {mean_psd_dev_h1}")
print(f"Mean PSD Deviation L1: {mean_psd_dev_l1}")
print("Soft Hair Memory Results for Different Entropy Methods")
print(f"Shannon Post-Ringdown H1: {p_r_shan_h1}")
print(f"Shannon Post-Ringdown L1: {p_r_shan_l1}")
print(f"Shannon PSD Ratio H1: {psd_shan_ratio_h1}")
print(f"Shannon PSD Ratio L1: {psd_shan_ratio_l1}")
print(f"Shannon Mean PSD Deviation H1: {mean_shan_psd_dev_h1}")
print(f"Shannon Mean PSD Deviation L1: {mean_shan_psd_dev_l1}")
print(f"Renyi Post-Ringdown H1: {p_r_ren_h1}")
print(f"Renyi Post-Ringdown L1: {p_r_ren_l1}")
print(f"Renyi PSD Ratio H1: {psd_ren_ratio_h1}")
print(f"Renyi PSD Ratio L1: {psd_ren_ratio_l1}")
print(f"Renyi Mean PSD Deviation H1: {mean_ren_psd_dev_h1}")
print(f"Renyi Mean PSD Deviation L1: {mean_ren_psd_dev_l1}")
print(f"Tsallis Post-Ringdown H1: {p_r_tsa_h1}")
print(f"Tsallis Post-Ringdown L1: {p_r_tsa_l1}")
print(f"Tsallis PSD Ratio H1: {psd_tsa_ratio_h1}")
print(f"Tsallis PSD Ratio L1: {psd_tsa_ratio_l1}")
print(f"Tsallis Mean PSD Deviation H1: {mean_tsa_psd_dev_h1}")
print(f"Tsallis Mean PSD Deviation L1: {mean_tsa_psd_dev_l1}")


print("------------------------------------")

#Quantum hair
stat_path = os.path.abspath(os.path.join(processed_folder.strip(), "statistical_results.hdf5"))

with h5py.File(stat_path, "r") as f:
    results = {}
    for category in f:
        results[category] = {}
        for key in f[category]:
            raw = f[category][key][()]
            try:
                decoded = raw.decode("utf-8")
                results[category][key] = json.loads(decoded)
            except:
                results[category][key] = raw

print("STATISTICAL RESULTS OF SOFT HAIR ENTROPY VALUE VS QUIET DATA")
for category, tests in results.items():
    print(f"\n---{category} ---")
    for test_name, metrics in tests.items():
        if isinstance(metrics, dict) and "stat" in metrics and "p" in metrics:
            stat = metrics["stat"]
            p_val = metrics["p"]
            print(f"{test_name}: Statistic = {stat:.4f}, P-Value = {p_val:.4f}")


print("------------------------------------")


#KS Test
ks_path = os.path.join(processed_folder.strip(), "ks_test_results.hdf5")

with h5py.File(ks_path, "r") as f:
    ks_stat_h1_shannon = f["ks_stat_h1_shannon"][()]
    ks_stat_l1_shannon = f["ks_stat_l1_shannon"][()]
    ks_stat_h1_renyi = f["ks_stat_h1_renyi"][()]
    ks_stat_l1_renyi = f["ks_stat_l1_renyi"][()]
    ks_stat_h1_tsallis = f["ks_stat_h1_tsallis"][()]
    ks_stat_l1_tsallis = f["ks_stat_l1_tsallis"][()]
    ks_p_value_h1_shannon = f["p_value_h1_shannon"][()]
    ks_p_value_l1_shannon = f["p_value_l1_shannon"][()]
    ks_p_value_h1_renyi = f["p_value_h1_renyi"][()]
    ks_p_value_l1_renyi = f["p_value_l1_renyi"][()]
    ks_p_value_h1_tsallis = f["p_value_h1_tsallis"][()]
    ks_p_value_l1_tsallis = f["p_value_l1_tsallis"][()]

print("\nKS TEST RESULTS")
print("This section compares entropy distributions before and after merger to detect statistically significant deviations in shape or structure.")
print(f"Shannon KS Statistic H1: {ks_stat_h1_shannon}, P-Value: {ks_p_value_h1_shannon}")
print(f"Shannon KS Statistic L1: {ks_stat_l1_shannon}, P-Value: {ks_p_value_l1_shannon}")

print(f"Renyi KS Statistic H1: {ks_stat_h1_renyi}, P-Value: {ks_p_value_h1_renyi}")
print(f"Renyi KS Statistic L1: {ks_stat_l1_renyi}, P-Value: {ks_p_value_l1_renyi}")

print(f"Tsallis KS Statistic H1: {ks_stat_h1_tsallis}, P-Value: {ks_p_value_h1_tsallis}")
print(f"Tsallis KS Statistic L1: {ks_stat_l1_tsallis}, P-Value: {ks_p_value_l1_tsallis}")

print("------------------------------------")

#Mutual information results 
mi_path = os.path.join(processed_folder.strip(), "mutual_information_results.hdf5")

with h5py.File(mi_path, "r") as f:
    shan_mi_value = f["shannon_mutual_info"][()]
    ren_mi_value = f["renyi_mutual_info"][()]
    tsa_mi_value = f["tsallis_mutual_info"][()]
    shan_norm_mi = f["shannon_normalized_mutual_info"][()]
    ren_norm_mi = f["renyi_normalized_mutual_info"][()]
    tsa_norm_mi = f["tsallis_normalized_mutual_info"][()]

print("\nMUTUAL INFORMATION RESULTS")
print("This section measures correlation between L1 and H1 entropy streams, indicating cross-detector entanglement or coherence.")
print(f"Shannon Mutual Information: {shan_mi_value}")
print(f"Shannon Normalized MI: {shan_norm_mi}")

print(f"Renyi Mutual Information: {ren_mi_value}")
print(f"Renyi Normalized MI: {ren_norm_mi}")

print(f"Tsallis Mutual Information: {tsa_mi_value}")
print(f"Tsallis Normalized MI: {tsa_norm_mi}")

print("------------------------------------")
#Quantum Echo
echo_path = os.path.abspath(os.path.join(processed_folder.strip(), "quantum_echo_results.hdf5"))

if os.path.exists(echo_path):
    print("---QUANTUM ECHO RESULTS---")
    print("This section looks for non-random structure persistent in post-ringdown signals that cannot be explained by classical noise or detector artifacts.")
    with h5py.File(echo_path, "r") as f:
        for label in f:
            grp = f[label]
            mean_autocorr = grp["mean_autocorr"][()]
            std_autocorr = grp["std_autocorr"][()]
            print(f"\n{label}")
            print(f"Mean Autocorrelation: {mean_autocorr:.6f}")
            print(f"Std Autocorrelation: {std_autocorr:.6f}")
else:
    print("Quantum Echo Results not available. Skipping echo summary.")
    

print("------------------------------------")

bayes_path = os.path.join(processed_folder.strip(), "bayesian_results.hdf5")

with h5py.File(bayes_path, "r") as f:
    shan_k_h1 = f["ShannonBayesFactor_H1"][()]
    shan_k_l1 = f["ShannonBayesFactor_L1"][()]
    ren_k_h1 = f["RenyiBayesFactor_H1"][()]
    ren_k_l1 = f["RenyiBayesFactor_L1"][()]
    tsa_k_h1 = f["TsallisBayesFactor_H1"][()]
    tsa_k_l1 = f["TsallisBayesFactor_L1"][()]
    shan_evidence_h1 = f["ShannonEvidence_H1"][()]
    shan_evidence_l1 = f["ShannonEvidence_L1"][()]
    ren_evidence_h1 = f["RenyiEvidence_H1"][()]
    ren_evidence_l1 = f["RenyiEvidence_L1"][()]
    tsa_evidence_h1 = f["TsallisEvidence_H1"][()]
    tsa_evidence_l1 = f["TsallisEvidence_L1"][()]

print("--- Bayesian Results ---")
print("This section evaluates which statistical model best fits the entropy data, highlighting potential non-Gaussian behavior.")
print(f"Shannon H1 Bayes Factor: {shan_k_h1}, Evidence: {shan_evidence_h1}")
print(f"Shannon L1 Bayes Factor: {shan_k_l1}, Evidence: {shan_evidence_l1}")

print(f"Renyi H1 Bayes Factor: {ren_k_h1}, Evidence: {ren_evidence_h1}")
print(f"Renyi L1 Bayes Factor: {ren_k_l1}, Evidence: {ren_evidence_l1}")

print(f"Tsallis H1 Bayes Factor: {tsa_k_h1}, Evidence: {tsa_evidence_h1}")
print(f"Tsallis L1 Bayes Factor: {tsa_k_l1}, Evidence: {tsa_evidence_l1}")

print("------------------------------------")

cauchy_path = os.path.join(processed_folder.strip(), "cauchy_entropy_results.hdf5")

with h5py.File(cauchy_path, "r") as f:
    log_shan_h1 = f["cauchy_log_likelihood_h1_shannon"][()]
    log_shan_l1 = f["cauchy_log_likelihood_l1_shannon"][()]
    log_ren_h1 = f["cauchy_log_likelihood_h1_renyi"][()]
    log_ren_l1 = f["cauchy_log_likelihood_l1_renyi"][()]
    log_tsa_h1 = f["cauchy_log_likelihood_h1_tsallis"][()]
    log_tsa_l1 = f["cauchy_log_likelihood_l1_tsallis"][()]
    kurt_shan_h1 = f["kurtosis_h1_shannon"][()]
    kurt_shan_l1 = f["kurtosis_l1_shannon"][()]
    kurt_ren_h1 = f["kurtosis_h1_renyi"][()]
    kurt_ren_l1 = f["kurtosis_l1_renyi"][()]
    kurt_tsa_h1 = f["kurtosis_h1_tsallis"][()]
    kurt_tsa_l1 = f["kurtosis_l1_tsallis"][()]

print("--- Cauchy Entropy Results ---")
print("This section tests for heavy-tailed entropy behavior, indicating rare but extreme deviations that may signal information retention.")
print(f"Shannon Cauchy Log-Likelihood (H1): {log_shan_h1}")
print(f"Shannon Cauchy Log-Likelihood (L1): {log_shan_l1}")
print(f"Shannon Kurtosis (H1): {kurt_shan_h1}")
print(f"Shannon Kurtosis (L1): {kurt_shan_l1}")

print(f"Renyi Cauchy Log-Likelihood (H1): {log_ren_h1}")
print(f"Renyi Cauchy Log-Likelihood (L1): {log_ren_l1}")
print(f"Renyi Kurtosis (H1): {kurt_ren_h1}")
print(f"Renyi Kurtosis (L1): {kurt_ren_l1}")

print(f"Tsallis Cauchy Log-Likelihood (H1): {log_tsa_h1}")
print(f"Tsallis Cauchy Log-Likelihood (L1): {log_tsa_l1}")
print(f"Tsallis Kurtosis (H1): {kurt_tsa_h1}")
print(f"Tsallis Kurtosis (L1): {kurt_tsa_l1}")

print("------------------------------------")

#homology results down here! 
homology_path = os.path.abspath(os.path.join(processed_folder.strip(), "persistent_homology_results.hdf5"))
if os.path.exists(homology_path):
    with h5py.File(homology_path, "r") as f:
        print("Persistent Homology Summary")
        print("This section uses topological data analysis to detect persistent structure features in entropy space that differ from noise.")
        print(f"H1 Diagram (Shannon): H0: {f['diagram_h1_H0_shannon'].shape}, H1: {f['diagram_h1_H1_shannon'].shape}, L1 Diagram (Shannon): H0: {f['diagram_l1_L0_shannon'].shape}, H1: {f['diagram_l1_L1_shannon'].shape}")
        print(f"H1 Diagram (Renyi): H0: {f['diagram_h1_H0_renyi'].shape}, H1: {f['diagram_h1_H1_renyi'].shape}, L1 Diagram (Renyi): H0: {f['diagram_l1_L0_renyi'].shape}, H1: {f['diagram_l1_L1_renyi'].shape}")
        print(f"H1 Diagram (Tsallis): H0: {f['diagram_h1_H0_tsallis'].shape}, H1: {f['diagram_h1_H1_tsallis'].shape},, L1 Diagram (Tsallis): H0: {f['diagram_l1_L0_tsallis'].shape}, H1: {f['diagram_l1_L1_tsallis'].shape}")
else:
    print("\nPersistent Homology results not found... skipping for now!")
        


if hasattr(sys.stdout, 'streams'):
    original_stdout = sys.stdout.streams[0]
    sys.stdout = original_stdout
log_file.close()
