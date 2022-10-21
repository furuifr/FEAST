# FEAST

## Introduction
FEAST is a federated feature selection framework under VFL setting, who considers conditional mutual information (CMI) based feature selection, and utilizes CMI to identify features that are highly correlated with the label while having low redundancy between each other. The workflow of FEAST mainly consists of four stages: namely data pre-processing, statistical variable generation, feature score calculation, and feature ranking and selection.

The files in the project are described as follows:
- dataset: The four datasets introduced in the paper.
- discretization: Implementation of data discretization (binning) methods.
- filter: Implementation of statistical variable generation and feature score calculation.
- multi-party-real: Implementation of FEAST in real multi-party scenarios.
- multi-party-simulation: Implementation of FEAST in simulated multi-party scenarios.
- single-party: Implementation of CFEAST.
- classification.py: Implementation of different classifier.
- featureSelectionInMultiParties.py: The workflow of FEAST.
- featureSelectionInSingleParty.py: The workflow of CFEAST.
- preprocessing.py: Implementation of data pre-processing.

## Environments
```
python 3.6.6
numpy 1.19.2
pandas 0.22.0
scikit-learn 0.24.2
xgboost 1.3.3
grpcio 1.14.1
protobuf 3.17.2
```

## Quick Start
### FEAST in real multi-party scenarios: 
The users needs to prepare multiple machines (one is active party and the others are passive parties). Then, placing the file whose filename with 'active' on the active party, and the file whose filename with 'passive' on the passive party. The rest of the files are required by all parties. Next, The users can run FEAST with the following commands:
```
python multi-party-real/feature_selection_passive_selectall.py
python multi-party-real/feature_selection_active_selectall.py
```

### FEAST in simulated multi-party scenarios: 
In this scenario, The users can simulate the multi-party feature selection process by modifying the profile.
Taking mimic dataset as an example, the users can run FEAST with the following command:
```
python multi-party-simulation/mimic/mimic_FEAST.py
```

### CFEAST:
Taking mimic dataset as an example, the users can run CFEAST with the following command:
```
python single-party/mimic/mimic_FEAST.py
```
