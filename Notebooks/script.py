# Create a Script to create HDF5 files and two CSV file for every patient 
# Controls: n01, n02, n03, n04, n05, n09, n10, n11, ins1, ins2, ins3, ins4, ins5, ins6, ins7, ins8, ins9
# RBD patients: rbd02, rbd03, rbd05, rbd07, rbd08, rbd09, rbd10, rbd11, rbd12, rbd13, rbd17, rbd18, rbd19, rbd21, rbd22

# CSV File should have the following columns:
# Patient ID
# Group (Control or RBD)
# each row should be a epoch of 30 seconds with the following columns:
# Epoch Start Time
# Epoch End Time
# sleep stage (Wake, N1, N2, N3, REM)
# n_beats (using hrv_per_epoch function) 
# hr_mean_bpm (using hrv_per_epoch function) 
# rmssd_ms (using hrv_per_epoch function) 
# sdnn_ms (using hrv_per_epoch function) 
# pnn50_pct (using hrv_per_epoch function) 
# HFC (using hpc_metric function)
# LFC (using hpc_metric function)
# LFC/HFC (using hpc_metric function)

# The second CSV file should have the following columns:
# Patient ID
# Group (Control or RBD)
# Sleep stage (Wake, N1, N2, N3, REM) per row with the following columns:
# Using hep_metric and delta_power_1s for all epoch of one sleep stage
# Pearson correlation pearson_r  
# Pearson correlation pearson_p
# Spearman correlation spearman_r
# Spearman correlation spearman_p

# HDF5 file (one per patient) should have the following structure:
# /Patient_ID
#     /Epochs
#         /Epoch_1
#             /Epoch Start Time
#             /Epoch End Time   
#             /sleep stage
#             /HEP values (30,)
#             /delta power values (30,) 
#         /Epoch_2
#             /Epoch Start Time
#             /Epoch End Time   
#             /sleep stage
#             /HEP values (30,)
#             /delta power values (30,) 

