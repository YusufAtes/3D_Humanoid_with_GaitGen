import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

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

# ==========================================
# 3. MAIN INTERACTIVE LOOP
# ==========================================

def main():
    # --- A. Load Resources ---
    print("Loading resources...")
    
    # 1. Load Statistics
    try:
        mean = np.load(os.path.join(DATA_DIR, "mean.npy"))
        std = np.load(os.path.join(DATA_DIR, "std.npy"))
        print("  > Statistics loaded.")
    except FileNotFoundError:
        print(f"Error: Could not find mean.npy or std.npy in {DATA_DIR}")
        return

    # 2. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleFCNN(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, hidden_size=HIDDEN_SIZE).to(device)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print(f"  > Model loaded from {MODEL_PATH}")
    else:
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    print("\n" + "="*40)
    print(" INTERACTIVE GAIT GENERATOR")
    print("="*40)

    while True:
        try:
            print("\nEnter Input Parameters (or 'q' to quit):")
            
            speed_val = 0.1
            r_leg_val = 0.75
            l_leg_val = 0.75
            
            input_vec = np.array([
                speed_val / 2.4, 
                r_leg_val / 1.0, 
                l_leg_val / 1.0
            ], dtype=np.float32)
            
            input_tensor = torch.tensor(input_vec).unsqueeze(0).to(device) # Shape [1, 3]

            # --- D. Run Inference ---
            with torch.no_grad():
                output = model(input_tensor)
                output = output.cpu().numpy().squeeze(0) # Shape [137]

            # --- E. Post-Processing ---
            # 1. Split Freqs and Period
            freqs_flat = output[:FREQ_DIM]
            predicted_period_norm = output[FREQ_DIM]
            
            # NOTE: If you normalized period separately, denormalize it here.
            # Assuming period output is raw or scaled similarly to frequency?
            # Based on your script, period was just concatenated. 
            # If it was part of the global normalization, use:
            # period_denorm = (predicted_period_norm * std) + mean (but mean is a vector...)
            # Since your normalization code handled the WHOLE vector (freq + period), 
            # we simply denormalize the *whole* output first if that was the training strategy.
            
            # CORRECT APPROACH based on your training script:
            # normalized_targets = np.hstack([norm_output, period_data])
            # Wait, your training script normalized ONLY the freq part in `normalize_and_flatten`?
            # Let's look at your previous code: 
            # "normalized_targets = np.hstack([norm_fft, total_period])"
            # This implies PERIOD IS NOT NORMALIZED. 
            
            freqs_denorm = denormalize(freqs_flat, mean, std)
            period_val = predicted_period_norm # Raw value

            # 2. Recover Shape
            structured_data = recover_shape(freqs_denorm) # (4, 2, 17)

            # 3. IFFT to Time Domain
            # shape: (32, 4) -> 32 time steps, 4 joints
            time_domain_signals = pred_ifft(structured_data) 

            # --- F. Visualization ---
            channels = ["Right Hip", "Right Knee", "Left Hip", "Left Knee"]
            
            fig, axs = plt.subplots(2, 2, figsize=(10, 8))
            fig.suptitle(f"Generated Gait\nSpeed: {speed_val}m/s | Period: {period_val:.2f}s")
            
            time_axis = np.linspace(0, 100, 32) # 0 to 100% of gait cycle
            
            for i, ax in enumerate(axs.flat):
                ax.plot(time_axis, time_domain_signals[:, i], color='tab:orange', linewidth=2)
                ax.set_title(channels[i])
                ax.set_xlabel("% Gait Cycle")
                ax.set_ylabel("Angle (rad)")
                ax.grid(True, alpha=0.3)
                
            plt.tight_layout()
            plt.show()
            
            print(f"  > Generated successfully. Period: {period_val:.4f}")

        except ValueError:
            print("  ! Invalid input. Please enter numbers.")
        except Exception as e:
            print(f"  ! An error occurred: {e}")

if __name__ == "__main__":
    main()