import os
import torch
import torch.nn as nn
import numpy as np
# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
# Paths - Point to where your training script saved the files
MODEL_PATH = os.path.join("kfold_results", "FINAL_BEST_MODEL.pth")
DATA_DIR = "gait reference phase 2"

# Constants (Must match training)
INPUT_SIZE = 3
OUTPUT_SIZE = 137
FREQ_DIM = 136
HIDDEN_SIZE = 512 # Default from your script, change if you tuned it differently

# ==========================================
# 2. MODEL & HELPER FUNCTIONS
# ==========================================

class SimpleFCNN(nn.Module):
    def __init__(self, input_size=3, output_size=137, hidden_size=512):
        super(SimpleFCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        return self.net(x)

def denormalize(pred_flat, mean, std):
    """Reverses the Standard Score normalization."""
    return (pred_flat * std) + mean

def recover_shape(flat_data):
    """
    Reconstructs (4, 2, 17) from flat (136,) vector.
    """
    # 1. Reshape to creation shape: (Freqs=17, Joints=4, Real/Imag=2)
    recovered = flat_data.reshape(17, 4, 2)
    # 2. Transpose to IFFT shape: (Joints=4, Real/Imag=2, Freqs=17)
    structured = recovered.transpose(1, 2, 0)
    return structured

def pred_ifft(predictions):
    """
    Performs Inverse FFT to get time-domain signals.
    """
    # Combine Real and Imaginary parts
    complex_pred = predictions[:, 0, :] + 1j * predictions[:, 1, :]
    
    # Inverse FFT (n=32 points)
    pred_time = np.fft.irfft(complex_pred, n=32, axis=1)
    
    # Transpose for plotting: (Time, Joints)
    return pred_time.T