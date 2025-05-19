import scipy.io as sio
from nfindr import run_nfindr
from utils import spectral_angle_mapper, plot_endmembers
import numpy as np 

# Load Urban_6.mat data
data = sio.loadmat("Urban_6.mat")
print("Loaded keys in Urban_6.mat:", data.keys())

hyp_data = data['Y']  # shape (bands, pixels) => (162, 94249)
gt_endmembers = data['M']  # shape (bands, num_endmembers) => (162, 6)

print("Hyperspectral image shape:", hyp_data.shape)
print("Ground truth endmember shape:", gt_endmembers.shape)

pixels = hyp_data
bands, N = pixels.shape
num_endmembers = gt_endmembers.shape[1]

print(f"Running N-FINDR to find {num_endmembers} endmembers...")

# Run N-FINDR algorithm
endmembers = run_nfindr(pixels, num_endmembers, max_iter=5)

print("N-FINDR completed.")

# Compare with ground truth using SAM
sam_angles = spectral_angle_mapper(gt_endmembers, endmembers)
print("Spectral Angle Mapper (SAM) angles [radians]:", sam_angles)
print(f"Average SAM: {np.mean(sam_angles):.4f} radians")

# Plot ground truth vs estimated endmembers
plot_endmembers(gt_endmembers, endmembers)