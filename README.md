# SEAT – Sutton Entropy Analysis Tool

SEAT is an advanced gravitational wave analysis toolkit designed to test for information retention in black hole mergers through entropy-based signal analysis.

It incorporates classical preprocessing and novel quantum-inspired metrics across multiple entropy models (Shannon, Renyi, Tsallis), enabling high-resolution investigation of potential memory effects, quantum echoes, and topological structures in LIGO strain data.

> **Built from scratch by Matthew R. Sutton — for scientists, by a curious mind.**

---

## Features

- Full entropy decomposition: Shannon, Renyi, and Tsallis methods
- Monte Carlo event injection system
- Entropy model selection via Bayesian factor analysis
- PSD deviation and ratio testing
- Wavelet entropy visualization
- Mutual information (H1 ↔ L1 correlation)
- Quantum echo detection (across all entropy types)
- Persistent homology and topological tracking (H0, H1 diagrams)
- Soft hair memory estimation with entropy deltas
- Unified event diagnostic output

---

## Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/SEAT.git
cd SEAT
+ ```

### 2. Set up your processed data folder 

Throughout the software, I refer to my designated file path that was set in the fetching_mechanism.py. I titled this the processed folder and stored it in my user account. I’d recommend following a similar process.

### 3. Install dependencies

pip install -r requirements.txt

This runs on Python out of terminal or command prompt. Python 3.9+ is recommended.

### 4. Set your target event in fetching_mechanism.py 

Replace the filler GPS time that is included 

### 5. Run through each module in this order as some build off of each other

1.) fetching_mechanism.py
2.) preprocess.py
3.) spectrogram.py
4.) entropy_analysis.py
5.) wavelet_analysis.py
6.) monte_carlo.py
7.) bayesian_model.py
8.) cauchy_analysis.py
9.) mutual_information.py
10.) smirnov.py
11.) homology.py
12.) echo.py***
13.) soft_hair_memory.py
14.) soft_hair_entropy.py
15.) event_diagnostic.py

### 6. The Quantum echo is optional.

This step is extremely computationally intensive. It took hours on my machine to run just one. This section is explained in detail below.

Measures autocorrelation within post-merger signals to detect potential “echoes”—low-amplitude, delayed repetitions of the original waveform. These echoes are hypothesized signatures of quantum-scale corrections near the event horizon.

• The analysis is run across multiple domains: standard whitened strain data, and entropy-transformed data (Shannon, Renyi, Tsallis).

• Autocorrelation functions are computed and statistically summarized (mean, std) for each channel and method.

• The test looks for non-random structure—persistent post-ringdown signals that cannot be explained by classical noise or detector artifacts.

• Entropy-based echoes offer a novel probe: if echoes persist more clearly in Renyi or Shannon domains than in raw strain, it suggests the information retention may be encoded in statistical structure, not amplitude.

### 7 Results

After completing the analysis, rename any output files that do not already include the event ID and store them in a dedicated folder to preserve results.

## Licensing

SEAT is released under the MIT License for academic and non-commercial use.

**Commercial use is prohibited without written permission.**

To request a commercial license, contact the author:
- **Email**: [mattsutton9@yahoo.com]

---

### TL;DR
- You can use SEAT for academic papers, coursework, and non-profit research
- You can’t package or sell SEAT commercially without permission

### Citation 
@misc{seat2025,
author = {Matthew R. Sutton},
title = {SEAT - Sutton Entropy Analysis Tool},
year = {2025},
url = {https://github.com/MRaySutt/SuttonEntropyAnalysisTool},
note = {Open-source software for gravitational-wave entropy analysis}
}
