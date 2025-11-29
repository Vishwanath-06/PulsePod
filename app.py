import streamlit as st
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pandas as pd
from collections import deque
from sklearn.metrics import classification_report, confusion_matrix
from joblib import load

# Conditionally import SNNTorch
try:
    import snntorch as snn
    from snntorch import surrogate
    SNNTORCH_AVAILABLE = True
except ImportError:
    SNNTORCH_AVAILABLE = False
    st.warning("snnTorch not found. SNN model will be unavailable.")

# =============================================================================
# --- PATH CONFIGURATION ---
# =============================================================================
DATA_DIR = r"....\unified_test_set"
MODELS_DIR = r"....\models"
LOG_FILE = 'logs/streamlit_events.log'

# Ensure log directory exists
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# =============================================================================
# --- LOGGER CONFIGURATION ---
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    filename=LOG_FILE,
    filemode='w'
)

# =============================================================================
# --- MODEL DEFINITIONS ---
# (This section is unchanged)
# =============================================================================
class ECG_CNN(nn.Module):
    def __init__(self, input_length=187):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2); self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(3, 2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2); self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(3, 2)
        self._to_linear = None; self._get_to_linear_size(input_length)
        self.fc1 = nn.Linear(self._to_linear, 128); self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)

    def _get_to_linear_size(self, l_in):
        with torch.no_grad():
            x = torch.zeros(1, 1, l_in); x = self.pool1(self.relu1(self.conv1(x))); x = self.pool2(self.relu2(self.conv2(x))); self._to_linear = x.shape[1] * x.shape[2]
            
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x))); x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, self._to_linear); x = self.relu3(self.fc1(x))
        x = self.dropout(x); x = self.fc2(x); return x

class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_length=187):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2); self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(3, 2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2); self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(3, 2)
        self._to_linear = None; self._get_to_linear_size(input_length)
        self.fc1 = nn.Linear(self._to_linear, 128); self.relu3 = nn.ReLU()
        
    def _get_to_linear_size(self, l_in):
        with torch.no_grad():
            x = torch.zeros(1, 1, l_in); x = self.pool1(self.relu1(self.conv1(x))); x = self.pool2(self.relu2(self.conv2(x))); self._to_linear = x.shape[1] * x.shape[2]
            
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x))); x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, self._to_linear); embeddings = self.relu3(self.fc1(x)); return embeddings

class CNN_LSTM_Model(nn.Module):
    def __init__(self, cnn_feature_extractor, lstm_hidden_size=128, lstm_layers=2):
        super().__init__()
        self.cnn = cnn_feature_extractor
        self.lstm = nn.LSTM(128, lstm_hidden_size, lstm_layers, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, 2)

    def forward(self, x):
        b, s, c, h = x.size(); ci = x.view(b * s, c, h); co = self.cnn(ci)
        ri = co.view(b, s, -1); lo, _ = self.lstm(ri); l = lo[:, -1, :]
        g = self.fc(l); return g

if SNNTORCH_AVAILABLE:
    class SNN_Model(nn.Module):
        def __init__(self, beta=0.95, spike_grad=surrogate.atan()):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 32, 5, padding=2); self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
            self.pool1 = nn.MaxPool1d(3, 2); self.conv2 = nn.Conv1d(32, 64, 5, padding=2)
            self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad); self.pool2 = nn.MaxPool1d(3, 2)
            self._to_linear = 64 * 46; self.fc1 = nn.Linear(self._to_linear, 128)
            self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad); self.fc2 = nn.Linear(128, 2)
            self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)
            
        def forward(self, x):
            mem1, mem2, mem3, mem4 = self.lif1.init_leaky(), self.lif2.init_leaky(), self.lif3.init_leaky(), self.lif4.init_leaky()
            mem4_rec = []
            for _ in range(50):
                cur1 = self.pool1(self.conv1(x)); spk1, mem1 = self.lif1(cur1, mem1)
                cur2 = self.pool2(self.conv2(spk1)); spk2, mem2 = self.lif2(cur2, mem2)
                flat = spk2.view(x.size(0), -1); cur3 = self.fc1(flat)
                spk3, mem3 = self.lif3(cur3, mem3); cur4 = self.fc2(spk3)
                spk4, mem4 = self.lif4(cur4, mem4); mem4_rec.append(mem4)
            return torch.stack(mem4_rec, dim=0).sum(dim=0)

# =============================================================================
# --- HELPER FUNCTIONS ---
# =============================================================================

@st.cache_resource(show_spinner="Loading model...")
def load_cached_model(model_choice, device, models_dir):
    # (This function is unchanged)
    model_file_map = {
        'CNN': 'final_robust_cnn.pt',
        'CNN (Focal Loss)': 'final_robust_cnn_focal_loss.pt',
        'CNN+LSTM': 'final_robust_cnn_lstm.pt',
        'SNN': 'final_robust_snn.pt'
    }
    model_file = model_file_map.get(model_choice)
    if not model_file:
        raise FileNotFoundError(f"No model file defined for '{model_choice}'")

    if 'CNN+LSTM' in model_choice:
        cnn_base = CNNFeatureExtractor(input_length=187)
        model = CNN_LSTM_Model(cnn_base).to(device)
    elif 'SNN' in model_choice:
        if not SNNTORCH_AVAILABLE:
            raise ImportError("snnTorch not installed. SNN model cannot be loaded.")
        model = SNN_Model().to(device)
    else:
        model = ECG_CNN(input_length=187).to(device)
    
    full_path = os.path.join(models_dir, model_file)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Model file not found at: {full_path}. Please check MODELS_DIR path.")
        
    model.load_state_dict(torch.load(full_path, map_location=device))
    model.eval()
    return model

@st.cache_data(show_spinner="Loading test data...")
def load_cached_test_data(data_dir):
    # (This function is unchanged)
    x_path = os.path.join(data_dir, "X_test.npy")
    y_path = os.path.join(data_dir, "y_test.npy")
    
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"Test data not found in '{data_dir}'. Please check DATA_DIR path.")
        
    X_test = np.load(x_path)
    y_test = np.load(y_path)
    return X_test, y_test

def handle_alert(event, conf_thresh=0.85, uncert_thresh=0.4):
    """
    Checks an event, logs it, and returns an alert message for the UI.
    Thresholds are now handled by the function's defaults.
    """
    alert_message = None
    if event["predicted_label"] == 1 and event["confidence"] >= conf_thresh:
        alert_message = f"🚨 **ABNORMAL BEAT DETECTED** (Prob: {event['confidence']:.2f})"
    elif event["uncertainty_entropy"] > uncert_thresh:
        alert_message = f"❓ **UNCERTAIN PREDICTION** (Entropy: {event['uncertainty_entropy']:.2f})"
    
    if alert_message:
        # Note: event['index'] is now the *actual* data index
        full_log = f"ALERT on Index {event['index']}: {alert_message} | True Label={event['true_label']}"
        logging.info(full_log) # Write to log file
        print(full_log) # Write to console
        return alert_message
    return None

def create_beat_plot(event):
    # (This function is unchanged)
    fig, ax = plt.subplots(figsize=(10, 4))
    
    if not event:
        # Default plot before simulation starts
        ax.set_title("Waiting for simulation to start...")
        ax.set_ylim(-2, 2)
        ax.grid(True)
        plt.close(fig)
        return fig

    # Determine color based on prediction and correctness
    color = 'green'
    if event["predicted_label"] == 1:
        color = 'red' # Abnormal prediction
    if event["predicted_label"] != event["true_label"]:
        color = 'orange' # Incorrect prediction

    ax.plot(event["signal"], color=color)
    
    # Note: event['index'] is the *actual* data index
    title = (
        f"Index: {event['index']} | True Label: {event['true_label']}\n"
        f"Prediction: {event['predicted_label']} | Confidence: {event['confidence']:.2f} | Entropy: {event['uncertainty_entropy']:.2f}"
    )
    ax.set_title(title)
    ax.set_ylim(-2, 2)
    ax.grid(True)
    plt.close(fig) # Close the figure to prevent memory leaks in Streamlit
    return fig

def create_confusion_matrix_plot(y_true, y_pred, model_choice):
    # (This function is unchanged)
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Abnormal'], 
                yticklabels=['Normal', 'Abnormal'], ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'Confusion Matrix ({model_choice})')
    plt.close(fig)
    return fig

# =============================================================================
# --- STREAMLIT APP LAYOUT & LOGIC (VERSION 9) ---
# =============================================================================

st.set_page_config(page_title="Project PulsePod", page_icon="❤️", layout="wide")
st.title("❤️ Project PulsePod: Real-Time Cardiac Simulation")

# --- Constants ---
SEQUENCE_LENGTH = 10
DEVICE = torch.device("cpu")
START_DELAY_SECONDS = 3

# --- Session State Initialization ---
if 'simulation_running' not in st.session_state:
    # Use states: False (idle), "PREPARING", True (running)
    st.session_state.simulation_running = False
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'all_events' not in st.session_state:
    st.session_state.all_events = []
if 'sequence_buffer' not in st.session_state:
    st.session_state.sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
if 'selected_model_name' not in st.session_state:
    st.session_state.selected_model_name = "CNN"
if 'model' not in st.session_state:
    st.session_state.model = None
if 'last_valid_event' not in st.session_state:
    st.session_state.last_valid_event = None # To hold last plot
if 'simulation_indices' not in st.session_state:
    st.session_state.simulation_indices = [] 
if 'num_beats_to_run' not in st.session_state:
    st.session_state.num_beats_to_run = 200
if 'abnormal_log' not in st.session_state:      # <-- NEW: Initialize log
    st.session_state.abnormal_log = []


# --- Sidebar Controls ---
st.sidebar.title("🔬 Simulation Controls")
st.sidebar.info("This app simulates the performance of PulsePod models on real-world ECG data.")

model_options = ['CNN', 'CNN (Focal Loss)', 'CNN+LSTM']
if SNNTORCH_AVAILABLE:
    model_options.append('SNN')

selected_model_name = st.sidebar.selectbox(
    "Select Model",
    options=model_options,
    key='selected_model_name',
    disabled=(st.session_state.simulation_running != False) # Disable if preparing or running
)

# 'num_beats' is the slider value
num_beats = st.sidebar.slider(
    "Number of Beats to Simulate", 
    min_value=1, 
    max_value=250,
    value=200,
    disabled=(st.session_state.simulation_running != False)
)
delay = st.sidebar.slider(
    "Simulation Delay (s)", 0.0, 1.0, 0.1, 0.05
)

# --- REMOVED THRESHOLD SLIDERS ---

col1, col2 = st.sidebar.columns(2)
if col1.button("Start Simulation", type="primary", use_container_width=True, disabled=(st.session_state.simulation_running != False)):
    # --- Start Button Logic ---
    st.session_state.simulation_running = "PREPARING" 
    st.session_state.current_index = 0
    st.session_state.all_events = []
    st.session_state.sequence_buffer.clear()
    st.session_state.last_valid_event = None
    st.session_state.abnormal_log = [] # <-- NEW: Clear log on start
    
    # Load model on start
    try:
        st.session_state.model = load_cached_model(selected_model_name, DEVICE, MODELS_DIR)
        st.toast(f"Model '{selected_model_name}' loaded successfully!")
    except Exception as e:
        st.error(f"🚨 ERROR loading model: {e}")
        st.error("Please check your model paths and environment.")
        st.session_state.simulation_running = False
    
    # Load data on start
    try:
        X_test, y_test = load_cached_test_data(DATA_DIR)
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        
        # --- Generate random indices ---
        total_beats_available = len(y_test)
        num_beats_requested = num_beats # Get value from slider
        
        # Ensure we don't request more beats than available
        num_to_sample = min(num_beats_requested, total_beats_available) 
        
        # Sample *without replacement*
        st.session_state.simulation_indices = np.random.choice(
            total_beats_available, num_to_sample, replace=False
        )
        # Store the actual number of beats we will run
        st.session_state.num_beats_to_run = num_to_sample 
        
        st.toast(f"Loaded {total_beats_available} test beats. Will sample {num_to_sample} random beats.")

    except Exception as e:
        st.error(f"🚨 ERROR loading data: {e}")
        st.error("Please check your data paths.")
        st.session_state.simulation_running = False
    st.rerun()

if col2.button("Stop / Reset", use_container_width=True, disabled=(st.session_state.simulation_running == False)):
    # --- Stop Button Logic ---
    # We keep the log intact when stopping, only clear on a new run
    st.session_state.simulation_running = False
    st.session_state.last_valid_event = None
    st.rerun()


# --- Main App Body ---
st.header(f"📈 Live Feed: {st.session_state.selected_model_name}")

# We draw the plot immediately, using the 'last_valid_event' from the previous
# run. This ensures the plot area is *always* filled.
plot_fig = create_beat_plot(st.session_state.last_valid_event)
st.pyplot(plot_fig)

# We now create a single placeholder for ALL content *below* the plot.
content_placeholder = st.empty()
# --- END OF FIXED LAYOUT ---


if st.session_state.simulation_running == False:
    # --- IDLE / FINAL REPORT STATE ---
    with content_placeholder.container():
        if not st.session_state.all_events:
            # App is idle, waiting to start
            st.info("Select a model and simulation parameters from the sidebar and click 'Start Simulation'.")
        else:
            # Simulation finished, show final report
            st.success("✅ Simulation Complete!")
            st.header("📊 Final Performance Report")
            
            y_true = [e['true_label'] for e in st.session_state.all_events]
            y_pred = [e['predicted_label'] for e in st.session_state.all_events]
            
            # --- SPLIT LAYOUT FOR REPORT ---
            col1, col2 = st.columns([1, 1]) # Two equal columns
            
            with col1:
                st.subheader("Classification Metrics")
                report = classification_report(y_true, y_pred, target_names=['Normal (0)', 'Abnormal (1)'])
                st.text(report) # Use st.text to preserve formatting
            
            with col2:
                st.subheader("Confusion Matrix")
                fig_cm = create_confusion_matrix_plot(y_true, y_pred, st.session_state.selected_model_name)
                st.pyplot(fig_cm)
            
            # --- NEW: Show final log ---
            st.subheader("Abnormal Beat Log")
            st.markdown("All beats predicted as **Abnormal (Label 1)** during the simulation:")
            log_text = "\n".join(st.session_state.abnormal_log)
            st.text_area("Logged Alerts", log_text, height=200, key="log_area_final")

                
elif st.session_state.simulation_running == "PREPARING":
    # --- PRE-SIMULATION DELAY STATE ---
    with content_placeholder.container():
        # Hold the spacer
        st.markdown("<div style='height: 69px;'></div>", unsafe_allow_html=True)
        # Show a countdown
        with st.spinner(f"🚀 Preparing simulation... Starting in {START_DELAY_SECONDS} seconds..."):
            time.sleep(START_DELAY_SECONDS)
        
        # Transition to the running state
        st.session_state.simulation_running = True
        st.rerun()

else:
    # --- SIMULATION IS RUNNING (state == True) ---
    
    # Check if model and data are loaded
    if st.session_state.model is None or 'X_test' not in st.session_state:
        st.warning("Model or data not loaded. Please Start Simulation.")
        st.session_state.simulation_running = False
        st.rerun()

    # Get the loop counter (e.g., 0, 1, 2, ...)
    idx = st.session_state.current_index 
    # Get the total number of beats we are simulating
    total_beats_to_run = st.session_state.num_beats_to_run 
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    
    # Check if we have simulated all requested beats
    if idx < total_beats_to_run:
        
        # --- Get the random data index for this step ---
        current_data_index = st.session_state.simulation_indices[idx]
        signal, label = X_test[current_data_index], y_test[current_data_index]
        
        event = None
        probs = None
        
        # --- Inference Logic ---
        with torch.no_grad():
            if st.session_state.selected_model_name == 'CNN+LSTM':
                st.session_state.sequence_buffer.append(signal)
                if len(st.session_state.sequence_buffer) == SEQUENCE_LENGTH:
                    seq_tensor = torch.tensor(
                        np.array(st.session_state.sequence_buffer), dtype=torch.float32
                    ).unsqueeze(0).unsqueeze(2).to(DEVICE)
                    logits = st.session_state.model(seq_tensor)
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            else:
                signal_tensor = torch.tensor(
                    signal, dtype=torch.float32
                ).unsqueeze(0).unsqueeze(0).to(DEVICE)
                logits = st.session_state.model(signal_tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        
        if probs is not None:
            # --- Event Processing ---
            prediction = np.argmax(probs)
            confidence = np.max(probs)
            uncertainty = -np.sum(probs * np.log(probs + 1e-9))
            
            event = {
                "index": int(current_data_index), # Use the actual data index
                "true_label": int(label),
                "predicted_label": prediction,
                "confidence": float(confidence),
                "uncertainty_entropy": float(uncertainty),
                "signal": signal
            }
            st.session_state.all_events.append(event)
            # **CRITICAL:** Save this event so the *next* run can draw it.
            st.session_state.last_valid_event = event
            
            # --- NEW: Add to log if abnormal ---
            if event["predicted_label"] == 1:
                log_entry = f"Beat {idx + 1} (Data Index: {event['index']}) -> Predicted: 1 (Prob: {event['confidence']:.2f}) | True: {event['true_label']}"
                st.session_state.abnormal_log.append(log_entry)
            
            # --- Update UI Elements ---
            with content_placeholder.container():
                # --- Alert Area ---
                # Call handle_alert *without* params to use defaults
                alert_msg = handle_alert(event) 
                if alert_msg:
                    st.warning(alert_msg)
                else:
                    st.markdown("<div style='height: 69px;'></div>", unsafe_allow_html=True)
                
                # --- Live Metrics Table ---
                live_data = {
                    'Metric': ['Beat Index', 'Prediction Confidence', 'Model Uncertainty'],
                    # Show beat 'n' out of 'total_beats_to_run'
                    'Value': [f"{idx + 1} / {total_beats_to_run}", f"{event['confidence']:.2f}", f"{event['uncertainty_entropy']:.2f}"]
                }
                df = pd.DataFrame(live_data).set_index('Metric')
                st.dataframe(df, use_container_width=True)
                
                # --- NEW: Live Log Area ---
                st.subheader("Abnormal Beat Log")
                log_text = "\n".join(st.session_state.abnormal_log)
                st.text_area("Logged Alerts", log_text, height=150, key="log_area_live")

        else:
            # --- UI for Buffering State (e.g., CNN+LSTM) ---
            with content_placeholder.container():
                # --- Alert Area (empty) ---
                st.markdown("<div style='height: 69px;'></div>", unsafe_allow_html=True)
                
                # --- Buffering Metrics Table ---
                live_data = {
                    'Metric': ['Beat Index', 'Prediction Confidence', 'Model Uncertainty'],
                    'Value': [f"Buffering... ({len(st.session_state.sequence_buffer)}/{SEQUENCE_LENGTH})", "---", "---"]
                }
                df = pd.DataFrame(live_data).set_index('Metric')
                st.dataframe(df, use_container_width=True)
                
                # --- NEW: Live Log Area (while buffering) ---
                st.subheader("Abnormal Beat Log")
                log_text = "\n".join(st.session_state.abnormal_log)
                st.text_area("Logged Alerts", log_text, height=150, key="log_area_live_buffer")


        # --- Loop Control ---
        st.session_state.current_index += 1
        time.sleep(delay)
        st.rerun()
        
    else:
        # --- Simulation Finished ---
        st.session_state.simulation_running = False
        st.toast("Simulation complete!")
        st.rerun()