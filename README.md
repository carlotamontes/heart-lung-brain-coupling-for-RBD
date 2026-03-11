# Heart–Lung–Brain Coupling in Sleep  
**CAP Sleep Database (PhysioNet)**

This repository contains analysis code for studying heart–lung–brain coupling during sleep using the CAP Sleep Database from PhysioNet. The project focuses on cardio–respiratory and heart–brain interactions derived from polysomnography data, with relevance for digital biomarkers in sleep and neurodegenerative disorders.

## Dataset
- **Source:** PhysioNet – CAP Sleep Database (v1.0.0)
- **Data type:** Polysomnography (EEG, ECG)

## Objectives
- Explore heart–lung coupling using cardio-pulmonary coupling (CPC) metrics
- Analyze heart–brain interactions via heartbeat-evoked potentials (HEP)

## Repository Structure
```

data/            # Local data only (not tracked by git)
├── raw/
notebooks/       # Exploratory and analysis notebooks
src/             # Reusable processing and feature extraction code

````

## Tools
- EDF visualization: EDFbrowser
- Signal loading and preprocessing: MNE-Python
- EDF I/O utilities: pyEDFlib
- ECG and physiological feature extraction: NeuroKit2

---
