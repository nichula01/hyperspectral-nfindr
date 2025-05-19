import numpy as np
from sklearn.decomposition import PCA
import math

def volume_simplex(A):
    """Calculate the volume of a simplex defined by vertices in A.
    A is a (p-1, p) matrix where each column is a vertex in (p-1)-dimensional space"""
    
    # Create the matrix of differences from the first vertex
    mat = A[:, 1:] - A[:, [0]]  # Shape: (p-1, p-1)
    
    # Calculate the volume using the determinant
    return np.abs(np.linalg.det(mat)) / math.factorial(A.shape[0])

def run_nfindr(X, p, max_iter=5):
    """
    Implementation of the N-FINDR algorithm to find endmembers in hyperspectral data.
    
    Parameters:
    X: numpy array, shape (bands, pixels) - each column is a pixel spectrum
    p: int, number of endmembers to find
    max_iter: int, maximum number of iterations
    
    Returns:
    endmembers: numpy array, shape (bands, p) - each column is an endmember spectrum
    """
    # Get data dimensions
    bands, pixels = X.shape
    
    # Reduce dimensionality to (p-1) using PCA
    pca = PCA(n_components=p-1)
    # Transpose X for PCA (sklearn expects samples in rows) then transpose back
    X_reduced = pca.fit_transform(X.T).T  # Shape: (p-1, pixels)
    
    # Initialize random endmember indices
    idx = np.random.choice(pixels, p, replace=False)
    
    # Extract initial endmember set
    E = X_reduced[:, idx]  # Shape: (p-1, p)
    
    # Calculate initial simplex volume
    V = volume_simplex(E)
    
    # Refine endmember selection through iterations
    for _ in range(max_iter):
        # For each endmember position
        for i in range(p):
            # Record current volume
            current_best_volume = V
            best_idx = idx[i]
            
            # Test each pixel as a potential replacement
            for j in range(pixels):
                # Skip if pixel is already an endmember
                if j in idx:
                    continue
                
                # Create temporary endmember set with potential replacement
                E_tmp = E.copy()
                E_tmp[:, i] = X_reduced[:, j]
                
                # Calculate new volume
                V_tmp = volume_simplex(E_tmp)
                
                # Update if volume increases
                if V_tmp > current_best_volume:
                    current_best_volume = V_tmp
                    best_idx = j
            
            # Apply the best replacement for this position
            if best_idx != idx[i]:
                idx[i] = best_idx
                E[:, i] = X_reduced[:, best_idx]
                V = current_best_volume

    # Transform selected endmembers back to original space
    endmembers = X[:, idx]
    
    return endmembers