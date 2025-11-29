# =============================================================================
# STEP 2.5: DATASET ENHANCEMENT PIPELINE
#
# This script builds a high-quality, balanced dataset for training by:
# 1. Loading multiple arrhythmia-heavy records from the MIT-BIH database.
# 2. Processing each record to extract beat segments, labels, and RR-intervals.
# 3. Performing a stratified train-test split to ensure both sets have the
#    same original class distribution.
# 4. Applying signal augmentation ONLY to the abnormal beats in the TRAINING set
#    to create a balanced training environment.
# 5. Saving the final, separated training and testing datasets to a new folder.
# =============================================================================

import os
import json
import numpy as np
import pandas as pd
import wfdb
import neurokit2 as nk
from scipy.signal import resample
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- 1. CONFIGURATION ---
print("--- Step 1: Initializing Configuration ---")
# Define the output directory for the new balanced dataset
OUTPUT_DIR = r"V:\Projects\MP\Ver3.0\processed_data_balanced"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# List of arrhythmia-heavy records from the MIT-BIH Arrhythmia Database
RECORDS = ['101', '106', '108', '109', '112', '114', '116', '118', '119', '122',
           '200', '201', '202', '203', '207', '208', '210', '213', '214', '215', '221', '223', '230']
print(f"Target records for processing: {len(RECORDS)}")

# Beat segmentation and resampling parameters
WINDOW_SIZE = 0.8  # seconds
RESAMPLE_LENGTH = 187 # A common length for MIT-BIH datasets

# Augmentation parameters
# We will augment until the ratio of Normal:Abnormal in the training set is ~2:1
TARGET_RATIO = 2.0


# --- 2. AUGMENTATION FUNCTION ---
def augment_signal(signal):
    """Applies realistic noise, time shift, and amplitude scaling to a signal."""
    # Add Gaussian noise
    noise = np.random.normal(0, 0.01, len(signal))
    signal_noisy = signal + noise
    
    # Apply a random time shift
    shift_amount = np.random.randint(-10, 10)
    signal_shifted = np.roll(signal_noisy, shift_amount)
    
    # Apply random amplitude scaling
    scale_factor = np.random.uniform(0.9, 1.1)
    signal_scaled = signal_shifted * scale_factor
    
    return signal_scaled


# --- 3. DATA COLLECTION AND PREPROCESSING ---
print("\n--- Step 2: Collecting and Preprocessing Data from all records ---")
all_beat_segments = []
all_beat_labels = []
all_beat_features = []

for record_name in tqdm(RECORDS, desc="Processing Records"):
    try:
        # Load record and annotations
        record = wfdb.rdrecord(record_name, pn_dir='mitdb')
        annotation = wfdb.rdann(record_name, 'atr', pn_dir='mitdb')
        
        signal = record.p_signal[:, 0]
        fs = record.fs
        
        # Get R-peaks
        _, rpeaks_info = nk.ecg_peaks(signal, sampling_rate=fs)
        rpeaks = rpeaks_info['ECG_R_Peaks']
        
        # Generate labels (0 for Normal 'N', 1 for others)
        # Note: We align our detected R-peaks to the nearest expert annotation
        symbols = annotation.symbol
        beat_indices = annotation.sample
        labels = []
        for rpeak in rpeaks:
            # Find the closest annotation symbol to this rpeak
            closest_symbol_idx = np.argmin(np.abs(beat_indices - rpeak))
            symbol = symbols[closest_symbol_idx]
            labels.append(0 if symbol == 'N' else 1)
        
        # Segment beats
        half_window = int(WINDOW_SIZE * fs / 2)
        beats = []
        valid_labels = []
        valid_rpeaks = []
        
        for i, r in enumerate(rpeaks):
            start, end = r - half_window, r + half_window
            if start >= 0 and end < len(signal):
                beat = signal[start:end]
                beat_resampled = resample(beat, RESAMPLE_LENGTH)
                beats.append(beat_resampled)
                valid_labels.append(labels[i])
                valid_rpeaks.append(r)
        
        # Calculate RR-intervals for the valid beats
        rr_intervals = np.diff(valid_rpeaks) / fs * 1000
        beat_features = [{"RR_ms": float(rr)} for rr in rr_intervals]
        
        # Align all data to the minimum length (usually len(rr_intervals))
        min_len = len(beat_features)
        all_beat_segments.extend(beats[:min_len])
        all_beat_labels.extend(valid_labels[:min_len])
        all_beat_features.extend(beat_features)

    except Exception as e:
        print(f"Could not process record {record_name}. Error: {e}")

# Convert lists to numpy arrays
X_all = np.array(all_beat_segments)
y_all = np.array(all_beat_labels)
features_df_all = pd.DataFrame(all_beat_features)

print("\n--- Initial Data Collection Summary ---")
unique, counts = np.unique(y_all, return_counts=True)
print(f"Total beats collected: {len(y_all)}")
print(f"Label distribution - 0 (Normal): {counts[0]}, 1 (Abnormal): {counts[1]}")


# --- 4. STRATIFIED TRAIN-TEST SPLIT (BEFORE AUGMENTATION) ---
print("\n--- Step 3: Performing Stratified Train-Test Split ---")
# We split the data BEFORE augmenting to ensure the test set is a pure, real-world sample
X_train, X_test, y_train, y_test, features_train, features_test = train_test_split(
    X_all, y_all, features_df_all,
    test_size=0.2,
    random_state=42,
    stratify=y_all  # <<< CRITICAL: a stratified split maintains class ratios
)
print("Data split complete.")
print(f"Train set size: {len(y_train)}, Test set size: {len(y_test)}")
unique_train, counts_train = np.unique(y_train, return_counts=True)
print(f"Train set labels - 0 (Normal): {counts_train[0]}, 1 (Abnormal): {counts_train[1]}")


# --- 5. AUGMENT THE TRAINING SET ---
print("\n--- Step 4: Augmenting Abnormal Beats in the Training Set ---")
# Identify abnormal beats in the training set
abnormal_indices = np.where(y_train == 1)[0]
normal_count = counts_train[0]
abnormal_count = counts_train[1]

# Calculate how many new samples we need to generate
num_new_samples = int((normal_count / TARGET_RATIO) - abnormal_count)
print(f"Need to generate {num_new_samples} new abnormal samples to reach a ~{int(TARGET_RATIO)}:1 ratio.")

if num_new_samples > 0:
    augmented_beats = []
    # Select abnormal beats to augment, allowing for repeats if necessary
    beats_to_augment_indices = np.random.choice(abnormal_indices, size=num_new_samples, replace=True)
    
    for idx in tqdm(beats_to_augment_indices, desc="Augmenting Signals"):
        original_beat = X_train[idx]
        augmented_beat = augment_signal(original_beat)
        augmented_beats.append(augmented_beat)
        
    # Add augmented beats and labels to the training set
    X_train = np.vstack([X_train, np.array(augmented_beats)])
    y_train = np.hstack([y_train, np.ones(num_new_samples, dtype=int)])
    
    # NOTE: We don't augment the tabular 'features' data as RR intervals for synthetic
    # beats would be harder to realistically generate. The primary goal here is to
    # balance the signal-based dataset for the CNN.
    
    print("Augmentation complete.")

print("\n--- Final Balanced Training Set Summary ---")
unique_final, counts_final = np.unique(y_train, return_counts=True)
print(f"Total training beats: {len(y_train)}")
print(f"Final train labels - 0 (Normal): {counts_final[0]}, 1 (Abnormal): {counts_final[1]}")


# --- 6. SAVE THE FINAL DATASETS ---
print("\n--- Step 5: Saving Final Datasets ---")
# Save the balanced training data
np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
features_train.to_json(os.path.join(OUTPUT_DIR, "features_train.json"), orient='records', indent=4)

# Save the original, imbalanced testing data
np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)
features_test.to_json(os.path.join(OUTPUT_DIR, "features_test.json"), orient='records', indent=4)

print(f"\n✅ All datasets saved successfully to:\n{OUTPUT_DIR}")
# Cell 2a: Dataset Sanity Check & Insights (Corrected)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# This cell assumes you have already loaded:
# X_train, y_train, features_train
# X_test, y_test, features_test

# Create DataFrames from the loaded feature lists/dicts
features_train_df = pd.DataFrame(features_train)
features_test_df = pd.DataFrame(features_test)


print("="*60)
print("              DATASET SANITY CHECK & INSIGHTS")
print("="*60)

# --- 1. Overall Shapes and Features ---
print("\n--- [1] Overall Shapes and Features ---")
print(f"Total Training Records (Signals): {X_train.shape[0]}")
print(f"Total Testing Records (Signals):  {X_test.shape[0]}")
print(f"Signal Feature Vector Length:     {X_train.shape[1]} samples per beat")
print(f"Tabular Features Available:         {list(features_train_df.columns)}")


# --- 2. Training Set Analysis ---
print("\n--- [2] Training Set Analysis (Balanced via Augmentation) ---")
train_norm_count = np.count_nonzero(y_train == 0)
train_abnorm_count = np.count_nonzero(y_train == 1)
train_total = len(y_train)
print(f"No. of records: {train_total}")
print(f"  - Normal (Class 0) Records:     {train_norm_count} ({train_norm_count/train_total:.2%})")
print(f"  - Abnormal (Class 1) Records:   {train_abnorm_count} ({train_abnorm_count/train_total:.2%})")

# --- 3. Testing Set Analysis ---
print("\n--- [3] Testing Set Analysis (Original Imbalance) ---")
test_norm_count = np.count_nonzero(y_test == 0)
test_abnorm_count = np.count_nonzero(y_test == 1)
test_total = len(y_test)
print(f"No. of records: {test_total}")
print(f"  - Normal (Class 0) Records:     {test_norm_count} ({test_norm_count/test_total:.2%})")
print(f"  - Abnormal (Class 1) Records:   {test_abnorm_count} ({test_abnorm_count/test_total:.2%})")


# --- 4. Visualizing the Class Balance ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Class Distribution Comparison', fontsize=16)

# Training set plot
sns.countplot(x=y_train, ax=axes[0])
axes[0].set_title(f'Training Set (Balanced)')
axes[0].set_xticklabels(['Normal (0)', 'Abnormal (1)'])
for p in axes[0].patches:
    axes[0].annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', xytext=(0, 9), textcoords='offset points')

# Testing set plot
sns.countplot(x=y_test, ax=axes[1])
axes[1].set_title(f'Testing Set (Original Imbalance)')
axes[1].set_xticklabels(['Normal (0)', 'Abnormal (1)'])
for p in axes[1].patches:
    axes[1].annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', xytext=(0, 9), textcoords='offset points')
plt.show()


# --- 5. Feature-Level Insight: RR Interval Distribution ---
print("\n--- [4] Feature-Level Insight: RR Interval Distribution by Class ---")
# FIX: We create the plot DataFrame using only the original, non-augmented
# training data, as we don't have RR features for the synthetic beats.
original_train_len = len(features_train_df)
plot_df = features_train_df.copy()
# Use only the slice of y_train that corresponds to the original data
plot_df['label'] = y_train[:original_train_len]

plt.figure(figsize=(12, 6))
sns.histplot(data=plot_df, x='RR_ms', hue='label', kde=True, bins=50)
plt.title('Distribution of RR Intervals for Normal vs. Abnormal Beats (Original Training Data)')
plt.legend(title='Class', labels=['Abnormal (1)', 'Normal (0)'])
plt.show()


# --- 6. Signal-Level Insight: Waveform Comparison ---
print("\n--- [5] Signal-Level Insight: Waveform Comparison ---")
# Find the first instance of a normal and an abnormal beat in the training data
normal_idx = np.where(y_train == 0)[0][0]
abnormal_idx = np.where(y_train == 1)[0][0]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Example Waveform Comparison', fontsize=16)

# Plot Normal Beat
axes[0].plot(X_train[normal_idx])
axes[0].set_title(f'Example Normal Beat (Class 0)')
axes[0].grid(True)

# Plot Abnormal Beat
axes[1].plot(X_train[abnormal_idx], color='orangered')
axes[1].set_title(f'Example Abnormal Beat (Class 1)')
axes[1].grid(True)
plt.show()