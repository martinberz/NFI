# =============================================================================
# ACADEMIC NON-COMMERCIAL USE LICENSE & PATENT DISCLAIMER
# =============================================================================
# Copyright (c) 2026, Authors of "Linearizing Phase Retrieval: The Theory 
# of Near-Field Interference (NFI)" (Martin Berz, Christoph Dillitzer, et al.)
# All rights reserved.
#
# 1. Academic Use Only: Permission is hereby granted, free of charge, to any 
#    person obtaining a copy of this software and associated documentation files 
#    (the "Software"), to use, copy, and modify the Software strictly for 
#    academic, educational, and non-commercial research purposes.
# 
# 2. Commercial Use Prohibited: Any commercial use, reproduction, distribution, 
#    or modification of the Software is strictly prohibited without prior 
#    written consent from the copyright holders.
#
# 3. Patent Rights Reserved: Parts of the Near-Field Interference (NFI) 
#    methodology implemented in this Software are subject to pending or issued 
#    patents. NO LICENSE, EXPRESS OR IMPLIED, BY ESTOPPEL OR OTHERWISE, IS 
#    GRANTED TO ANY PATENT RIGHTS held by the authors or their affiliated 
#    institutions. The use of this Software does not convey any rights to 
#    practice the patented inventions for commercial purposes.
#
# 4. Disclaimer: THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
#    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# =============================================================================

"""
Near-Field Interference (NFI) Core Simulation Solver

This script contains the numerical simulation framework for the paper:
"Linearizing Phase Retrieval: The Theory of Near-Field Interference (NFI)"

It evaluates the deterministic longitudinal evolution of a coherent wavefield 
across a short free-space propagation distance, operating in either a continuous
coordinate-free Least Squares Analysis (LSA) mode or a discrete hexagonal 
basis projection mode.
"""

import numpy as np
import scipy.ndimage as ndimage
from scipy.fft import fft2, ifft2, fftfreq, fftshift, ifftshift
from scipy.linalg import eig, inv
from scipy.sparse.linalg import LinearOperator, lsmr

# Set consistent printing for numpy outputs
np.set_printoptions(legacy='1.13')

# =============================================================================
# 1. Simulation Configuration & Mathematical Grid Setup
# =============================================================================
nx = 512              # Observation grid size (pixels)
oversampling = 3      # Grid oversampling factor to prevent circular convolution wrap-around
q_N = 0.5             # Pupil cut-off frequency (defines the strict band limit)
z_0 = 30.0            # NFI longitudinal shear distance (in units of Rayleigh length z_R)
z_1 = 15.0            # Half shear distance (used for symmetric focal plane evaluation)
w_0 = 2.0 / np.pi     # Base Gaussian width parameter for paraxial propagation

N_s = 30              # Hexagonal grid array size (N_s x N_s)
M = N_s * N_s         # Total number of physical discrete scatterers (900)
d_hex = np.sqrt(8.0 / (np.pi * np.sqrt(3.0))) # Hexagonal lattice spacing constant

# Toggle Simulation Modes
LSA_CALC = False                # True = Continuous Krylov (LSA), False = Discrete Hexagonal Matrix
BIT_NUMBER = -1                # -1 = Noiseless. Set to e.g., 8 for 8-bit Poisson noise evaluation.
SPECTRAL_NOISE_FILTER = True   # Truncate unphysical high-frequency noise outside the physical bandwidth

# Set up spatial frequency grid and enforce strict band limitation (Paley-Wiener constraint)
qx = fftfreq(nx, d=1/oversampling)
qx_shifted = fftshift(qx)
Qx, Qy = np.meshgrid(qx, qx)
band_mask = (Qx**2 + Qy**2 <= q_N**2).astype(float)
valid_idx = np.where(band_mask.flatten() > 0)[0]
N_pupil = len(valid_idx)
Qx_valid, Qy_valid = Qx.flatten()[valid_idx], Qy.flatten()[valid_idx]

# Construct Hexagonal Scatterer Coordinates
X_coords, Y_coords = [], []
for j in range(N_s):
    for i in range(N_s):
        x = (i + 0.5 * (j % 2)) * d_hex
        y = j * (np.sqrt(3.0) / 2.0) * d_hex
        X_coords.append(x)
        Y_coords.append(y)
X_coords = np.array(X_coords) - np.mean(X_coords)
Y_coords = np.array(Y_coords) - np.mean(Y_coords)

# Phase interaction matrices: projects discrete spatial scatterers into the continuous band-limited Fourier space
phase_arg = -2j * np.pi * (np.outer(Qx_valid, X_coords) + np.outer(Qy_valid, Y_coords))
Phase_matrix = np.exp(phase_arg) / np.sqrt(N_pupil)
Phase_matrix_conj_T = Phase_matrix.conj().T

# Pre-calculate paraxial propagation kernels
def get_prop_kernel(z):
    """Generates the frequency-domain free-space paraxial transfer function."""
    return np.exp(-1/4 * w_0**2 * 1j * z * (2 * np.pi)**2 * (Qx**2 + Qy**2))

q_w0_prop_z0 = get_prop_kernel(z_0)
q_minus_valid = get_prop_kernel(-z_1).flatten()[valid_idx]
q_plus_valid = get_prop_kernel(z_1).flatten()[valid_idx]

eins_bg = np.ones((nx, nx), dtype=complex) / nx

# =============================================================================
# 2. Core Physics Functions (Forward Models and Projections)
# =============================================================================
def V_base(c_1d):
    """Projects discrete scatterer coefficients into Plane 1 (-z1)."""
    F_valid = np.dot(Phase_matrix, c_1d) * q_minus_valid
    F_full = np.zeros((nx, nx), dtype=complex)
    F_full.flat[valid_idx] = F_valid
    return fftshift(ifft2(F_full, norm='ortho'))

def V_focus_base(c_1d):
    """Projects discrete scatterer coefficients into the Middle Focal Plane (z=0)."""
    F_valid = np.dot(Phase_matrix, c_1d) 
    F_full = np.zeros((nx, nx), dtype=complex)
    F_full.flat[valid_idx] = F_valid
    return fftshift(ifft2(F_full, norm='ortho'))

def V_adj_base(E):
    """Adjoint operator: Projects continuous wavefield back to discrete scatterer coordinates."""
    F_full = fft2(ifftshift(E), norm='ortho')
    F_valid = F_full.flat[valid_idx] * q_plus_valid
    return np.dot(Phase_matrix_conj_T, F_valid)

def U(E):
    """Forward Paraxial Propagation Operator (Plane 1 to Plane 2)."""
    return ifft2(fft2(E, norm='ortho') * q_w0_prop_z0, norm='ortho')

def add_noise(IF_shearup):
    """Injects Poisson counting statistics (shot noise) based on intensity."""
    IF_noise_r = np.random.poisson(np.abs(IF_shearup)) - np.abs(IF_shearup)
    IF_noise_i = np.random.poisson(np.abs(IF_shearup)) - np.abs(IF_shearup)
    return IF_noise_r + 1j * IF_noise_i

def to_x(qm):
    """Transforms from compressed 1D valid pupil frequencies to 2D real space."""
    q_full = np.zeros((nx, nx), dtype=complex)
    q_full.flat[valid_idx] = qm
    return fftshift(ifft2(q_full, norm='ortho'))

def to_q(x_arr):
    """Transforms from 2D real space to compressed 1D valid pupil frequencies."""
    q_full = fft2(ifftshift(x_arr), norm='ortho')
    return q_full.flat[valid_idx]

def create_data_driven_mask(a2E1, a2E2, nu=0.1, close_radius=5):
    """
    Generates a closed geometric support mask strictly from raw intensity data.
    Used exclusively as the LSA computational anchor for unanchored Dark-Field regimes.
    """
    m1 = np.mean(a2E1)
    t1 = nu * np.median(a2E1[a2E1 > m1]) if np.any(a2E1 > m1) else 0
    m2 = np.mean(a2E2)
    t2 = nu * np.median(a2E2[a2E2 > m2]) if np.any(a2E2 > m2) else 0
    
    mask_raw = (a2E1 > t1) & (a2E2 > t2)
    
    # Morphological closing to fill diffractive nulls and pad the edges
    y, x = np.ogrid[-5:5+1, -5:5+1]
    struct5 = x**2 + y**2 <= 5**2
    y, x = np.ogrid[-close_radius:close_radius+1, -close_radius:close_radius+1]
    struct = x**2 + y**2 <= close_radius**2

    mask_closed = ndimage.binary_closing(mask_raw, structure=struct5)
    mask_dilated = ndimage.binary_dilation(mask_closed, structure=struct5)
    return mask_dilated.astype(float)

# =============================================================================
# 3. Main Simulation Execution
# =============================================================================
if __name__ == '__main__':
    print("Initializing Ground Truth Scatterer Field...")
    np.random.seed(42)
    # Generate 900 physical random scatterers with complex phase
    c_scatterers = np.random.randn(M) + 1j * np.random.randn(M)
    norm_pt_true = np.linalg.norm(c_scatterers)

    xE1_pt = V_base(c_scatterers)               
    xE_focus_pt = V_focus_base(c_scatterers)    
    xE2_pt = U(xE1_pt)                           
    max_E1_pt = np.max(np.abs(xE1_pt))

    # Define Simulation Matrix Cases based on Noise Regime
    if BIT_NUMBER > 0:
        cases = [
            {"id": 1, "bg_frac": 0.0, "use_wiener": False, "v_damp_factor": 1.6e-3, "tol": 1e-7},
            {"id": 2, "bg_frac": 0.0, "use_wiener": True,  "v_damp_factor": 1.6e-3, "tol": 1e-7}, 
            {"id": 3, "bg_frac": 1.0, "use_wiener": False, "v_damp_factor": 1.6e-1, "tol": 1e-2},
            {"id": 4, "bg_frac": 1.0, "use_wiener": True,  "v_damp_factor": 1.6e-1, "tol": 1e-2}, 
        ]
    else:
        cases = [
            {"id": 1, "bg_frac": 0.0, "use_wiener": False, "v_damp_factor": 0.2e-3, "tol": 1e-8},
            {"id": 2, "bg_frac": 0.0, "use_wiener": True,  "v_damp_factor": 0.2e-3, "tol": 1e-8}, 
            {"id": 3, "bg_frac": 1.0, "use_wiener": False, "v_damp_factor": 2e-2,   "tol": 1e-8},
            {"id": 4, "bg_frac": 1.0, "use_wiener": True,  "v_damp_factor": 2e-2,   "tol": 1e-8}, 
        ]

    # Initialize CSV Tracking
    csv_rows = []
    if LSA_CALC:
        csv_rows.append("bg_frac,use_wiener,num_passes,lsmr_iters_last,v_damp_factor,tol,cond_number,corr_pt")
    else:
        csv_rows.append("bg_frac,use_wiener,cond_number,corr_zero")

    for case in cases:
        case_id = case["id"]
        bg_frac = case["bg_frac"]
        use_wiener = case["use_wiener"]
        case_damp_factor = case["v_damp_factor"]
        case_tol = case["tol"]
        
        print(f"\n{'='*80}")
        print(f"RUNNING CASE {case_id}: bg={bg_frac:.2f}, wiener={use_wiener}, LSA_CALC={LSA_CALC}, Bits={BIT_NUMBER}")
        print(f"{'='*80}")
        
        # Construct Total Field (Scatterers + Background Reference Wave)
        bg_actual_amplitude = bg_frac * max_E1_pt * nx 
        xE1_bgr = bg_actual_amplitude * eins_bg 
        xE1_true = xE1_pt + xE1_bgr
        xE0_true = xE_focus_pt + xE1_bgr
        xE2_true = U(xE1_true)
        
        # Ground Truth Ideal Coherent Observables
        a2E1_true = np.abs(xE1_true)**2
        a2E2_true = np.abs(xE2_true)**2
        IF_true = np.conj(xE2_true) * xE1_true

        # Apply Poisson Noise Quantization (if applicable)
        if BIT_NUMBER < 0:
            a2E1_used, a2E2_used, IF_used = a2E1_true, a2E2_true, IF_true
        else:
            max_int = (2**BIT_NUMBER) - 1
            global_max = max(np.max(a2E1_true), np.max(a2E2_true), np.max(np.abs(np.real(IF_true))), np.max(np.abs(np.imag(IF_true))))
            scale_factor = max_int / global_max
            
            a2E1_scaled = a2E1_true * scale_factor
            a2E2_scaled = a2E2_true * scale_factor
            IF_scaled = IF_true * scale_factor
            
            a2E1_used = np.random.poisson(a2E1_scaled) / scale_factor
            a2E2_used = np.random.poisson(a2E2_scaled) / scale_factor
            IF_used = (IF_scaled + add_noise(IF_scaled)) / scale_factor

        # Spectral Noise Filtering (Truncate noise > 2*q_c created by pointwise observable multiplication)
        if SPECTRAL_NOISE_FILTER and BIT_NUMBER > 0:
            obs_band_mask = (Qx**2 + Qy**2 <= (2*q_N)**2).astype(float)
            
            a2E1_used = np.real(ifft2(fft2(a2E1_used, norm='ortho') * obs_band_mask, norm='ortho'))
            a2E1_used = np.maximum(a2E1_used, 0.0)
            
            a2E2_used = np.real(ifft2(fft2(a2E2_used, norm='ortho') * obs_band_mask, norm='ortho'))
            a2E2_used = np.maximum(a2E2_used, 0.0)
            
            IF_used = ifft2(fft2(IF_used, norm='ortho') * obs_band_mask, norm='ortho')

        # Wiener preconditioning dampens spurious near-zero modes in unanchored dark regions
        weight_mask = 1.0 / (a2E2_used + 0.05 * np.max(a2E2_used)) if use_wiener else 1.0
        support_mask_finite = create_data_driven_mask(a2E1_used, a2E2_used, nu=1.0, close_radius=50)

        # =========================================================================
        # PATH A: Continuous LSA Reconstruction (Krylov Subspace Inversion)
        # =========================================================================
        if LSA_CALC:
            def f_qmIFUA(cE0_test):
                """Continuous linear operator evaluating the NFI identity."""
                xE1, xE2 = to_x(cE0_test * q_minus_valid), to_x(cE0_test * q_plus_valid)
                return to_q((IF_used * xE2 - a2E2_used * xE1) * weight_mask) * q_plus_valid 

            def f_qmIFUA_adj(qm_y):
                """Adjoint continuous operator for Krylov backwards pass."""
                x_res_prime = to_x(qm_y * np.conj(q_plus_valid))
                xE1_prime = -a2E2_used * weight_mask * x_res_prime
                xE2_prime = np.conj(IF_used) * weight_mask * x_res_prime
                return to_q(xE1_prime) * np.conj(q_minus_valid) + to_q(xE2_prime) * np.conj(q_plus_valid)

            A_op = LinearOperator((N_pupil, N_pupil), matvec=f_qmIFUA, rmatvec=f_qmIFUA_adj, dtype=complex)
            
            # Dampening mitigates Krylov noise-overfitting
            v_damp = 0.0 if BIT_NUMBER < 0 else case_damp_factor * np.max(a2E2_used)
            tol = 1e-12 if BIT_NUMBER < 0 else case_tol
            
            # The multi-pass bootstrap is only required for Dark-Field due to lack of a global initial guess
            num_passes = 1 if bg_frac > 0 else 10
            
            # Initial Guess Mechanism
            if bg_frac > 0:
                qmE0_guess = to_q(bg_frac * max_E1_pt * nx * eins_bg)
            else:
                # Blind start with random phase to break Twin-Image symmetry
                xE2_guess_complex = np.sqrt(a2E2_used) * np.exp(1j * np.random.uniform(-0.1, 0.1, (nx, nx)))
                qmE0_guess = to_q(xE2_guess_complex) * np.conj(q_plus_valid) 

            total_iters = 0
            for p in range(num_passes):
                print(f"  --- LSMR Pass {p+1}/{num_passes} ---")
                b_guess = f_qmIFUA(qmE0_guess)
                max_iters = 10000 if (num_passes > 5 or num_passes == 1) else 1000
                res = lsmr(A_op, -b_guess, damp=v_damp, atol=tol, btol=tol, show=False, maxiter=max_iters)
                
                total_iters += res[2]
                qm_rec = res[0] + qmE0_guess
                
                # Apply Data-Driven Bootstrap between passes to mathematical anchor the phase
                if p < num_passes - 1:
                    xE0_temp = to_x(qm_rec)
                    skp = np.vdot(xE0_temp, xE0_true)
                    xE0_temp *= np.exp(1j * np.angle(skp))
                    # Mask and band-limit sequentially 
                    qmE0_guess = to_q(xE0_temp * support_mask_finite)

            conda_est = res[6]
            lsmr_iters_last = res[2]
            
            print(f"  Condition Number Est. : {conda_est:.2f}")        
            print(f"  LSMR Total Iterations : {total_iters} (Last pass: {lsmr_iters_last})")

            # Final Alignment against True Point Field
            xE0_rec_raw = to_x(qm_rec)
            skp_full = np.vdot(xE0_rec_raw, xE0_true)
            xE0_rec_aligned = xE0_rec_raw * np.exp(1j * np.angle(skp_full)) * (np.linalg.norm(xE0_true) / np.linalg.norm(xE0_rec_raw))
            
            xE0_pt_true = xE_focus_pt 
            xE0_pt_rec_aligned = xE0_rec_aligned - (to_x(to_q(bg_frac * max_E1_pt * nx * eins_bg)) if bg_frac > 0 else 0)
            
            skp_pt = np.vdot(xE0_pt_rec_aligned, xE0_pt_true)
            xE0_pt_rec_aligned *= np.exp(1j * np.angle(skp_pt)) * (np.linalg.norm(xE0_pt_true) / np.linalg.norm(xE0_pt_rec_aligned))
            
            # Calculate Geometric Correlation of True Recovered Phase
            corr_pt = np.abs(np.vdot(xE0_pt_true, xE0_pt_rec_aligned)) / (np.linalg.norm(xE0_pt_true) * np.linalg.norm(xE0_pt_rec_aligned))

            print(f"  Final Structural Correlation: {corr_pt:.4f}")
            csv_rows.append(f"{bg_frac:.2f},{use_wiener},{num_passes},{lsmr_iters_last},{case_damp_factor:.2e},{tol:.2e},{conda_est:.2f},{corr_pt:.3f}")

        # =========================================================================
        # PATH B: Discrete Hexagonal Projection (Dual-Basis Null-Space Evaluator)
        # =========================================================================
        else:
            if bg_frac == 0.0:
                basis_size = M
                def V(c): return V_base(c)
                def V_adj(E): return V_adj_base(E)
            else:
                basis_size = M + 1 # Include +1 to explicitly anchor the macroscopic Gabor background
                def V(c_ext): return V_base(c_ext[:M]) + c_ext[M] * eins_bg
                def V_adj(E): return np.append(V_adj_base(E), np.vdot(eins_bg, E))

            # Construct Dual Space Gram Matrix to handle non-orthogonal diffractive overlap
            print("  Building Gram Overlap Matrix (Dual Space)...")
            S = np.zeros((basis_size, basis_size), dtype=complex)
            for i in range(basis_size):
                e_i = np.zeros(basis_size, dtype=complex)
                e_i[i] = 1.0
                S[:, i] = V_adj(V(e_i))
            S_inv = inv(S)
            
            def IFUA_projected(c_test):
                """Evaluates the NFI residual and projects strictly back into the dual basis."""
                E_test = V(c_test)
                E_prop = U(E_test)
                raw_residual = IF_used * E_prop - a2E2_used * E_test
                balanced_residual = raw_residual * weight_mask
                return S_inv @ V_adj(balanced_residual) 

            # Construct Dense IFUA Operator Matrix
            print("  Constructing Dense Discrete IFUA Matrix...")
            IFUA_dense = np.zeros((basis_size, basis_size), dtype=complex)
            for i in range(basis_size):
                e_i = np.zeros(basis_size, dtype=complex)
                e_i[i] = 1.0
                IFUA_dense[:, i] = IFUA_projected(e_i)

            print("  Diagonalizing IFUA Matrix...")
            evals, evecs = eig(IFUA_dense)

            # Sort eigenspectrum logically to find the mathematical null space
            idx_zero = np.argmin(np.abs(evals))
            active_indices = [i for i in range(basis_size) if i != idx_zero]
            active_indices.sort(key=lambda i: np.real(evals[i]), reverse=True)
            sorted_indices = [idx_zero] + active_indices
            
            evals_ordered = evals[sorted_indices]
            evecs_ordered = evecs[:, sorted_indices]

            # Evaluate Operator Condition (Stability)
            min_nonzero_abs = np.min(np.abs(evals[active_indices]))
            cond_number = np.max(np.abs(evals)) / min_nonzero_abs
            
            # Evaluate Accuracy of Zero Mode against True Physical Object
            c_mode_pt = evecs_ordered[:M, 0] 
            mode_norm = np.linalg.norm(c_mode_pt)
            if mode_norm > 0:
                corr_zero = np.abs(np.vdot(c_scatterers, c_mode_pt)) / (norm_pt_true * mode_norm)
            else:
                corr_zero = 0.0

            print(f"  Condition Number (kappa)     : {cond_number:.2f}")
            print(f"  Final Structural Correlation : {corr_zero:.4f}")

            csv_rows.append(f"{bg_frac:.2f},{use_wiener},{cond_number:.2f},{corr_zero:.3f}")

    # =========================================================================
    # Final Output
    # =========================================================================
    csv_filename = f"summary_results_b_{BIT_NUMBER}_snf_{SPECTRAL_NOISE_FILTER}_lsa_{LSA_CALC}.csv"
    with open(csv_filename, "w") as f:
        f.write("\n".join(csv_rows) + "\n")
    print(f"\n[{'#'*78}]")
    print(f"Simulation Complete. Numeric Results successfully saved to: {csv_filename}")
    print(f"[{'#'*78}]")
