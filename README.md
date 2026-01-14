# Heart–Lung–Brain Coupling in Sleep  
**CAP Sleep Database (PhysioNet)**

This repository contains analysis code for studying heart–lung–brain coupling during sleep using the CAP Sleep Database from PhysioNet. The project focuses on cardio–respiratory and heart–brain interactions derived from polysomnography data, with relevance for digital biomarkers in sleep and neurodegenerative disorders.

## Dataset
- **Source:** PhysioNet – CAP Sleep Database (v1.0.0)
- **Data type:** Polysomnography (EEG, ECG, respiration, EOG, EMG)
- **Annotations:** Cyclic Alternating Pattern (CAP) sleep microstructure

## Objectives
- Explore heart–lung coupling using cardio-pulmonary coupling (CPC) metrics
- Analyze heart–brain interactions via heartbeat-evoked potentials (HEP)
- Characterize coupling patterns during sleep microstructure (CAP vs non-CAP)

## Repository Structure
```

data/            # Local data only (not tracked by git)
├── raw/
├── processed/
└── metadata/
notebooks/       # Exploratory and analysis notebooks
src/             # Reusable processing and feature extraction code
figures/         # Generated figures
results/         # Tables and metrics
docs/            # Notes and references

````

## Tools
- EDF visualization: EDFbrowser
- Signal loading and preprocessing: MNE-Python
- EDF I/O utilities: pyEDFlib
- ECG and physiological feature extraction: NeuroKit2

## Reproducibility
1. Download the CAP Sleep Database from PhysioNet  
2. Place EDF files in `data/raw/`
3. Create the Python environment:
   ```bash
   conda env create -f environment.yml
````

4. Run notebooks in numerical order
```

---
