import numpy as np
import matplotlib.pyplot as plt

def spectral_angle_mapper(M_true, M_est):
    """
    Calculate the spectral angle between true and estimated endmembers.
    
    Parameters:
    M_true: numpy array, shape (bands, num_endmembers) - Ground truth endmembers
    M_est: numpy array, shape (bands, num_endmembers) - Estimated endmembers
    
    Returns:
    numpy array of spectral angles in radians
    """
    # Create a mapping to match estimated endmembers to ground truth
    # This is a simple approach - for each ground truth endmember,
    # find the estimated endmember with the smallest angle
    M_true_norm = M_true / np.linalg.norm(M_true, axis=0, keepdims=True)
    M_est_norm = M_est / np.linalg.norm(M_est, axis=0, keepdims=True)

    sam = []
    for i in range(M_true.shape[1]):
        cos_sim = np.dot(M_true_norm[:, i], M_est_norm[:, i])
        angle = np.arccos(np.clip(cos_sim, -1.0, 1.0))
        sam.append(angle)
    return np.array(sam)

def plot_endmembers(M_true, M_est, wavelengths=None):
    """
    Plot ground truth vs estimated endmember spectra.
    
    Parameters:
    M_true: numpy array, shape (bands, num_endmembers) - Ground truth endmembers
    M_est: numpy array, shape (bands, num_endmembers) - Estimated endmembers
    wavelengths: numpy array, optional - x-axis values for plotting
    """
    bands, num = M_true.shape
    
    if wavelengths is None:
        wavelengths = np.arange(bands)  # Default to band indices
    
    plt.figure(figsize=(15, 10))
    
    # Calculate SAM for including in plot titles
    sam_angles = spectral_angle_mapper(M_true, M_est)
    
    for i in range(num):
        plt.subplot(2, (num+1)//2, i+1)
        plt.plot(wavelengths, M_true[:, i], 'b-', linewidth=2, label='Ground Truth')
        plt.plot(wavelengths, M_est[:, i], 'r--', linewidth=2, label='N-FINDR')
        
        plt.title(f'Endmember {i+1} (SAM: {sam_angles[i]:.4f} rad)')
        plt.xlabel('Wavelength/Band')
        plt.ylabel('Reflectance')
        plt.grid(True, alpha=0.3)
        
        if i == 0:  # Add legend only to the first subplot to avoid repetition
            plt.legend(loc='best')
    
    plt.tight_layout()
    plt.show()