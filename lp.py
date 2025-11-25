"""
Image Deblurring using Linear Programming with L1 Regularization
"""

import numpy as np
from scipy.io import loadmat
from scipy.optimize import linprog, OptimizeWarning
import matplotlib.pyplot as plt
from time import time
from scipy import sparse
from scipy.sparse import csr_matrix, vstack, hstack, eye
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.stats import median_abs_deviation

def load_data(example_num):
    """Load example data"""
    data = loadmat(f'Example{example_num}.mat')
    print(f"Available keys in Example{example_num}.mat: {data.keys()}")
    x_tilde = data['xtilde'].flatten()  # Ensure 1D
    A = data['A']
    # Check for ground truth with both 'x' and 'xtrue' keys
    x_true = data.get('x', data.get('xtrue', None))
    if x_true is not None:
        x_true = x_true.flatten()  # Ensure 1D
    print(f"Shape of x_tilde: {x_tilde.shape}")
    print(f"Shape of A: {A.shape}")
    print(f"Ground truth available: {x_true is not None}")
    return x_tilde, A, x_true

def estimate_noise_level(x_tilde, A):
    """Estimate noise level in the blurred image"""
    
    # Compute residuals if we had a simple inverse solution
    try:
        x_est = np.linalg.lstsq(A, x_tilde, rcond=None)[0]
        residuals = x_tilde - A @ x_est
        noise_std = 1.4826 * median_abs_deviation(residuals)
        return noise_std / np.median(np.abs(x_tilde))  # Return relative noise level
    except:
        return 0.1  # Default if estimation fails

def select_optimal_lambda(x_tilde, A, lambda_range=None, num_lambdas=10):
    """Automatically select optimal lambda using a small sample of the problem"""
    n = len(x_tilde)
    sample_size = min(1000, n)  # Use a subset for faster computation
    idx = np.random.choice(n, sample_size, replace=False)
    
    A_sample = A[idx][:, idx] if sparse.issparse(A) else A[idx][:, idx]
    x_tilde_sample = x_tilde[idx]
    
    if lambda_range is None:
        # Estimate a reasonable range based on problem scale
        noise_level = estimate_noise_level(x_tilde, A)
        lambda_min = 1e-6 * noise_level
        lambda_max = 10.0 * noise_level
        lambda_range = np.logspace(np.log10(lambda_min), np.log10(lambda_max), num_lambdas)
    
    best_lambda = None
    best_cost = float('inf')
    
    for l in lambda_range:
        try:
            x_opt = solve_deblurring(x_tilde_sample, A_sample, l, verbose=False)
            # Cost = data fidelity + regularization
            data_fidelity = 0.5 * np.linalg.norm(A_sample @ x_opt - x_tilde_sample)**2
            reg_term = l * np.linalg.norm(x_opt, 1)
            total_cost = data_fidelity + reg_term
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_lambda = l
        except:
            continue
    
    return best_lambda if best_lambda is not None else lambda_range[len(lambda_range)//2]

# ---------- helper: normalization ----------
def normalize_problem(A, x_tilde, do_normalize=True):
    """
    Optionally scale A and x_tilde for numerical stability.
    Returns (A_scaled, x_tilde_scaled, scale_factor)
    We choose scale = max(abs(A)) to make matrix entries ~O(1).
    If do_normalize=False, returns originals and scale_factor=1.0
    """
    if not do_normalize:
        return A, x_tilde, 1.0

    if sparse.issparse(A):
        maxA = float(abs(A).max())
    else:
        maxA = float(np.max(np.abs(A)))

    maxb = float(np.max(np.abs(x_tilde))) if np.any(x_tilde != 0) else 1.0

    # avoid division by zero
    if maxA <= 0:
        scale = 1.0
    else:
        scale = maxA

    A_scaled = A / scale
    x_tilde_scaled = x_tilde / scale
    return A_scaled, x_tilde_scaled, scale

# ---------- LP builder ----------
def build_lp_standard_form(A, x_tilde, lambda_val, normalize=True):
    """
    Simplified, consistent LP:

    Variables: z = [x (n), t (n)]
    Objective: minimize  sum_i t_i  + lambda * sum_i x_i

    Constraints (A_ub z <= b_ub):
      A x - t <= x_tilde
     -A x - t <= -x_tilde

    Bounds:
      0 <= x <= 1
      t >= 0
    """

    # normalization for numerical stability
    A_s, b_s, scale = normalize_problem(A, x_tilde, do_normalize=normalize)
    n = len(b_s)

    # build block matrices (sparse-friendly)
    I_n = eye(n, format='csr')
    if sparse.issparse(A_s):
        A_mat = csr_matrix(A_s)
    else:
        A_mat = csr_matrix(A_s)

    # Two inequality blocks:
    # [  A   -I ] [ x ] <= [ x_tilde ]
    # [ -A   -I ] [ t ]    [ -x_tilde ]
    # Build A_ub as a sparse matrix with shape (2n, 2n)
    top = hstack([A_mat, -I_n], format='csr')   # shape (n, 2n)
    bot = hstack([-A_mat, -I_n], format='csr')  # shape (n, 2n)
    A_ub = vstack([top, bot], format='csr')     # shape (2n, 2n)

    b_ub = np.concatenate([b_s, -b_s])

    # Objective c: [lambda * ones(n) for x,  ones(n) for t]
    c = np.zeros(2 * n)
    c[0:n] = lambda_val     # penalize x (x >= 0), encourages small x when lambda large
    c[n:2*n] = 1.0          # penalize residual magnitudes t

    # Variable bounds
    bounds = []
    # x bounds: 0 <= x <= 1
    for _ in range(n):
        bounds.append((0.0, 1.0))
    # t bounds: 0 <= t < +inf
    for _ in range(n):
        bounds.append((0.0, None))

    var_info = {'x': (0, n), 't': (n, 2*n)}
    # return scaled A_ub and b_ub together with scale factor for unscaling if needed
    return c, A_ub, b_ub, bounds, var_info, scale

# ---------- solve_deblurring using highs-ds ----------
def solve_deblurring(x_tilde, A, lambda_val=None, verbose=True, auto_lambda=True,
                     normalize=True, lambda_grid=None):
    """
    Solve the deblurring problem using LP (HiGHS dual simplex).
    Uses the simplified LP formulation with variables [x, t].
    """

    n = len(x_tilde)

    # Auto-select lambda if not provided
    if lambda_val is None and auto_lambda:
        # if no user grid provided, choose wide grid
        if lambda_grid is None:
            lambda_grid = np.logspace(-6, 2, 24)
        # quick grid search using small subsample to find promising lambda
        best_l = None
        best_obj = np.inf
        # Subsample small set for speed
        idx = np.random.choice(n, min(200, n), replace=False)
        A_sub = A[idx][:, idx] if sparse.issparse(A) else A[np.ix_(idx, idx)]
        b_sub = x_tilde[idx]
        for l in lambda_grid:
            try:
                c_sub, A_ub_sub, b_ub_sub, bounds_sub, _, _ = build_lp_standard_form(A_sub, b_sub, l, normalize=False)
                res_sub = linprog(c_sub, A_ub=A_ub_sub, b_ub=b_ub_sub, bounds=bounds_sub,
                                  method='highs-ds', options={'presolve': True, 'disp': False})
                if res_sub.success:
                    # compute full-sample surrogate objective (approx)
                    obj = res_sub.fun
                    if obj < best_obj:
                        best_obj = obj
                        best_l = l
            except Exception:
                continue
        lambda_val = best_l if best_l is not None else (lambda_grid[len(lambda_grid)//2])
        if verbose:
            print(f"Auto-selected lambda ~ {lambda_val:.2e}")

    if lambda_val is None:
        lambda_val = 1e-3

    if verbose:
        print("\n" + "="*50)
        print(f"Solving deblurring with lambda = {lambda_val:.2e}")
        print(f"Problem size: n = {n}, A shape = {A.shape}")
        print("-"*50)

    # Build LP (normalize A and b inside builder)
    c, A_ub, b_ub, bounds, var_info, scale = build_lp_standard_form(A, x_tilde, lambda_val, normalize=normalize)

    start_time = time()

    # Call HiGHS dual simplex explicitly
    options = {'presolve': True, 'time_limit': 120, 'disp': verbose}

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=OptimizeWarning)
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                          method='highs-ds', options=options)

        solve_time = time() - start_time

        if not res.success:
            if verbose:
                print("Solver did not converge to optimal:", res.status, res.message)
            return None

        z = res.x  # length 2n
        x_opt = z[0:n]  # first block is x
        # Clip to [0,1] numerical slight violations
        x_opt = np.clip(x_opt, 0.0, 1.0)

        if verbose:
            print("Optimization successful.")
            print(f"Solve time: {solve_time:.3f}s, Objective: {res.fun:.6e}")

        return x_opt, solve_time

    except Exception as e:
        if verbose:
            print("Error in LP solve:", str(e))
        return None

def calculate_metrics(x_opt, x_tilde, A, x_true=None, solve_time=None):
    """
    Calculate performance metrics for deblurring
    
    Args:
        x_opt: Deblurred image (optimized solution)
        x_tilde: Blurred input image
        A: Blurring matrix
        x_true: Ground truth image (optional)
        solve_time: Time taken for optimization (if available)
        
    Returns:
        Dictionary containing metrics
    """
    metrics = {}
    
    # Input validation
    x_opt = np.asarray(x_opt).flatten()
    x_tilde = np.asarray(x_tilde).flatten()
    
    if x_opt.size != x_tilde.size:
        raise ValueError(f"x_opt size ({x_opt.size}) must match x_tilde size ({x_tilde.size})")
    
    # Store solve time if provided
    if solve_time is not None and solve_time > 0:
        metrics['solve_time'] = float(solve_time)
    
    try:
        # Calculate residuals
        residuals = A @ x_opt - x_tilde
        abs_residuals = np.abs(residuals)
        
        # Fraction of exactly fitted pixels (residuals very close to zero)
        tol = 1e-8
        metrics['exact_fit_fraction'] = float(np.mean(abs_residuals < tol))
        
        # Calculate sparsity as percentage of near-zero elements
        abs_x = np.abs(x_opt)
        max_val = np.max(abs_x)
        
        if max_val > 0:
            # Use a fixed threshold relative to the maximum value
            # This is a small fraction of the maximum value to identify near-zero elements
            threshold = 1e-4 * max_val
            
            # Count elements below threshold and calculate percentage
            n_below = np.sum(abs_x < threshold)
            sparsity = 100.0 * n_below / len(x_opt)
            
            # Ensure the value is within [0, 100]%
            metrics['sparsity'] = max(0.0, min(100.0, sparsity))
            
            # Store threshold for debugging
            metrics['sparsity_threshold'] = threshold
            metrics['n_below_threshold'] = int(n_below)
            metrics['total_elements'] = len(x_opt)
        else:
            # If all values are zero, sparsity is 100%
            metrics['sparsity'] = 100.0
            metrics['sparsity_threshold'] = 0.0
            metrics['n_below_threshold'] = len(x_opt)
            metrics['total_elements'] = len(x_opt)
        
        # Residual statistics
        metrics['max_residual'] = float(np.max(abs_residuals))
        metrics['mean_residual'] = float(np.mean(abs_residuals))
        metrics['residual_std'] = float(np.std(residuals))
        
        # Solution statistics
        metrics['min_val'] = float(np.min(x_opt))
        metrics['max_val'] = float(np.max(x_opt))
        metrics['mean_val'] = float(np.mean(x_opt))
        metrics['std_val'] = float(np.std(x_opt))
        
        # Calculate image metrics if ground truth is provided
        if x_true is not None:
            x_true = np.asarray(x_true).flatten()
            x_opt_flat = np.asarray(x_opt).flatten()
            
            # Find the minimum length to avoid index out of bounds
            min_len = min(len(x_true), len(x_opt_flat))
            
            # Use only the overlapping part of both arrays
            x_true = x_true[:min_len]
            x_opt_reshaped = x_opt_flat[:min_len]
            
            # RMSE (Root Mean Squared Error)
            mse = np.mean((x_opt_reshaped - x_true) ** 2)
            metrics['rmse'] = float(np.sqrt(mse))
            
            # Reshape for image metrics
            try:
                # Find a suitable shape for both images
                n = len(x_true)
                # Find factors of n that are close to square
                factors = [(i, n // i) for i in range(int(np.sqrt(n)), 0, -1) if n % i == 0]
                if factors:
                    h, w = factors[0]  # Get the most square shape
                    try:
                        x_opt_img = x_opt_reshaped.reshape(h, w)
                        x_true_img = x_true.reshape(h, w)
                        
                        # PSNR (Peak Signal-to-Noise Ratio)
                        metrics['psnr'] = float(psnr(x_true_img, x_opt_img, data_range=1.0))
                        
                        # SSIM (Structural Similarity Index)
                        metrics['ssim'] = float(ssim(x_true_img, x_opt_img, 
                                                  data_range=1.0, 
                                                  channel_axis=None))
                    except ValueError as ve:
                        # If reshaping fails, use 1D metrics only
                        metrics['psnr'] = float('nan')
                        metrics['ssim'] = float('nan')
                        print(f"Warning: Could not reshape to ({h}, {w}) for PSNR/SSIM: {str(ve)}")
                else:
                    # If no suitable shape found, use 1D metrics only
                    metrics['psnr'] = float('nan')
                    metrics['ssim'] = float('nan')
            except ValueError:
                # Fallback if reshaping fails
                metrics['psnr'] = float('nan')
                metrics['ssim'] = float('nan')
    
    except Exception as e:
        # Return NaNs if any error occurs during metric calculation
        error_metrics = ['exact_fit_fraction', 'sparsity', 'max_residual', 
                        'mean_residual', 'residual_std', 'min_val', 'max_val', 
                        'mean_val', 'std_val']
        for m in error_metrics:
            metrics[m] = float('nan')
        if 'rmse' in locals():
            metrics['rmse'] = float('nan')
        if 'psnr' in locals():
            metrics['psnr'] = float('nan')
        if 'ssim' in locals():
            metrics['ssim'] = float('nan')
            
        print(f"Warning: Error calculating metrics: {str(e)}")
    
    return metrics

def plot_results(x_tilde, x_opt, x_true=None, lambda_val=None, example_num=0):
    """Plot the blurred, deblurred, and ground truth images"""
    # Get image dimensions
    n = len(x_tilde)
    
    # Find the closest integers l1 and l2 such that l1 * l2 = n
    # This handles non-square images
    l1 = int(np.sqrt(n))
    while n % l1 != 0 and l1 > 0:
        l1 -= 1
    l2 = n // l1
    
    # If no valid factors found, use a square root approximation
    if l1 == 0 or l2 == 0:
        l1 = l2 = int(np.ceil(np.sqrt(n)))
        # Truncate or pad if needed
        if l1 * l2 > n:
            x_tilde = np.pad(x_tilde, (0, l1*l2 - n), 'constant')
            x_opt = np.pad(x_opt, (0, l1*l2 - n), 'constant')
            if x_true is not None:
                x_true = np.pad(x_true, (0, l1*l2 - n), 'constant')
    
    # Reshape the images
    img_shape = (l1, l2)
    img_blurred = x_tilde.reshape(img_shape)
    img_deblurred = x_opt.reshape(img_shape)
    
    # Create the figure
    plt.figure(figsize=(15, 9))
    
    # Plot blurred image
    plt.subplot(1, 2 + (x_true is not None), 1)
    plt.imshow(img_blurred, cmap='gray')
    plt.title('Blurred Image')
    plt.axis('off')
    
    # Plot deblurred image
    plt.subplot(1, 2 + (x_true is not None), 2)
    plt.imshow(img_deblurred, cmap='gray', vmin=0, vmax=1)
    title = 'Deblurred Image'
    if lambda_val is not None:
        title += f' (λ={lambda_val})'
    plt.title(title)
    plt.axis('off')
    
    # Plot ground truth if available
    if x_true is not None:
        img_true = x_true.reshape(img_shape)
        plt.subplot(1, 3, 3)
        plt.imshow(img_true, cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_regularization_path(x_opt_list, lambda_values):
    plt.figure(figsize=(10, 6))
    for i in range(min(50, len(x_opt_list[0]))):  # Plot first 50 coefficients
        coef_path = [x[i] for x in x_opt_list]
        plt.plot(lambda_values, coef_path, 'b-', alpha=0.3)
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('Coefficient value')
    plt.title('Regularization Path')
    plt.grid(True)
    plt.show()

def lambda_sensitivity_analysis(x_tilde, A, x_true=None, lambda_values=None):
    """Run sensitivity analysis for different lambda values"""
    if lambda_values is None:
        lambda_values = np.logspace(-4, 0, 5)  # Default lambda values
    
    metrics_list = []
    
    for lambda_val in lambda_values:
        print(f"\n--- Testing lambda = {lambda_val:.2e} ---")
        
        # Solve the deblurring problem
        x_opt, solve_time = solve_deblurring(x_tilde, A, lambda_val)
        
        # Calculate metrics
        metrics = {
            'lambda': lambda_val,
            'solve_time': solve_time,
            'objective': np.sum(np.abs(A @ x_opt - x_tilde)) + lambda_val * np.sum(np.abs(x_opt))
        }
        
        # Calculate additional metrics
        metrics.update(calculate_metrics(x_opt, x_true, x_tilde, A))
        
        # Add sparsity metric
        if x_true is not None:
            stats = analyze_solution(x_opt, x_true, x_tilde, A)
            metrics['sparsity'] = stats['sparsity']
        
        metrics_list.append(metrics)
    
    return metrics_list

def add_noise(image, noise_level=0.05):
    """Add Gaussian noise to the image"""
    noise = np.random.normal(0, noise_level * np.max(image), image.shape)
    return image + noise

def analyze_solution(x_opt, x_true, x_tilde, A):
    """Analyze the solution in detail"""
    # Calculate residuals
    residuals = A @ x_opt - x_tilde
    abs_residuals = np.abs(residuals)
    
    # Solution statistics
    stats = {
        'min_val': np.min(x_opt),
        'max_val': np.max(x_opt),
        'mean_val': np.mean(x_opt),
        'sparsity': np.mean(np.abs(x_opt) < 1e-6) * 100,  # Fraction of near-zero values
        'max_residual': np.max(abs_residuals),
        'mean_residual': np.mean(abs_residuals),
        'residual_std': np.std(residuals)
    }
    return stats

def plot_comparison(x_tilde, x_opt, x_true, title_suffix=""):
    """Plot comparison of original, blurred, and deblurred images with additional visualizations"""
    n = len(x_tilde)
    # Find best square dimensions for display
    l = int(np.ceil(np.sqrt(n)))
    while n % l != 0 and l > 0:
        l -= 1
    if l == 0:
        l = int(np.ceil(np.sqrt(n)))
    m = n // l
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 9))
    
    # Reshape images
    def safe_reshape(x, shape):
        x_flat = x.flatten()
        if len(x_flat) < l * m:
            x_flat = np.pad(x_flat, (0, l * m - len(x_flat)))
        return x_flat.reshape(shape)
    
    shape = (l, m)
    img_blurred = safe_reshape(x_tilde, shape)
    img_deblurred = safe_reshape(x_opt, shape)
    
    # Plot 1: Blurred Image
    plt.subplot(2, 3, 1)
    plt.imshow(img_blurred, cmap='gray')
    plt.title(f'Blurred Image\n{title_suffix}')
    plt.axis('off')
    
    # Plot 2: Deblurred Image
    plt.subplot(2, 3, 2)
    plt.imshow(img_deblurred, cmap='gray', vmin=0, vmax=1)
    plt.title(f'Deblurred Image\n{title_suffix}')
    plt.axis('off')
    
    # Plot 3: Ground Truth (if available)
    if x_true is not None:
        img_true = safe_reshape(x_true, shape)
        plt.subplot(2, 3, 3)
        plt.imshow(img_true, cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')
    
    # Plot 4: Difference between deblurred and ground truth
    if x_true is not None:
        plt.subplot(2, 3, 4)
        diff = np.abs(img_deblurred - img_true)
        plt.imshow(diff, cmap='hot', vmin=0, vmax=0.5)
        plt.colorbar()
        plt.title('Absolute Error (Deblurred - True)')
        plt.axis('off')
    
    # Plot 5: Histogram of pixel intensities
    plt.subplot(2, 3, 5)
    plt.hist(x_opt, bins=50, alpha=0.7, color='blue', label='Deblurred')
    if x_true is not None:
        plt.hist(x_true, bins=50, alpha=0.5, color='green', label='Ground Truth')
    plt.title('Pixel Intensity Distribution')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Plot 6: Spatial distribution of non-zero pixels
    plt.subplot(2, 3, 6)
    non_zero_mask = (np.abs(x_opt) > 1e-6).reshape(shape)
    plt.imshow(non_zero_mask, cmap='binary')
    plt.title(f'Non-zero Pixels (Sparsity: {np.mean(np.abs(x_opt) < 1e-6) * 100:.1f}%)')
    plt.axis('off')
    
    plt.tight_layout()
    # Create a clean filename by replacing special characters
    clean_title = title_suffix.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
    plt.show()

def plot_metrics_vs_lambda(metrics_list, noise_level=0.0):
    """
    Plot metrics vs lambda values with improved visualization
    
    Args:
        metrics_list: List of dictionaries containing metrics for each lambda
        noise_level: Noise level for plot title
    """
    if not metrics_list or len(metrics_list) < 2:
        print("Not enough data points to plot metrics vs lambda")
        return
    
    # Sort metrics by lambda
    metrics_list = sorted(metrics_list, key=lambda x: x['lambda'])
    lambdas = np.array([m['lambda'] for m in metrics_list])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f'Metrics vs Regularization Strength (Noise: {noise_level*100:.0f}%)', 
                fontsize=16, y=1.02)
    
    # Helper function to plot a single metric
    def plot_metric(ax, metric_key, title, ylabel, color, logy=False, invert=False):
        if metric_key not in metrics_list[0]:
            ax.set_visible(False)
            return
            
        values = np.array([m.get(metric_key, np.nan) for m in metrics_list])
        valid_idx = ~np.isnan(values)
        
        if not np.any(valid_idx):
            ax.set_visible(False)
            return
            
        if logy:
            ax.semilogy(lambdas[valid_idx], values[valid_idx], 'o-', 
                       linewidth=2, markersize=8, color=color, markerfacecolor='white')
        else:
            ax.semilogx(lambdas[valid_idx], values[valid_idx], 'o-', 
                       linewidth=2, markersize=8, color=color, markerfacecolor='white')
        
        # Find and mark optimal point
        if invert:
            opt_idx = np.nanargmin(values) if np.any(~np.isnan(values)) else 0
        else:
            opt_idx = np.nanargmax(values) if np.any(~np.isnan(values)) else 0
            
        ax.plot(lambdas[opt_idx], values[opt_idx], 'ro', markersize=10, 
               markeredgewidth=2, markerfacecolor='none')
        
        # Add text annotation
        ax.annotate(f'λ={lambdas[opt_idx]:.1e}\n{values[opt_idx]:.3f}',
                   xy=(lambdas[opt_idx], values[opt_idx]),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
                   arrowprops=dict(arrowstyle='->', color='black'))
        
        ax.set_xlabel('Regularization Strength (λ)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, pad=15)
        ax.grid(True, which="both", ls="--", alpha=0.7)
        ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Plot each metric in its subplot
    plot_metric(axes[0,0], 'rmse', 'RMSE vs λ', 'RMSE', 'dodgerblue')
    plot_metric(axes[0,1], 'psnr', 'PSNR vs λ', 'PSNR (dB)', 'forestgreen')
    plot_metric(axes[1,0], 'ssim', 'SSIM vs λ', 'SSIM', 'darkviolet')
    
    # Special handling for sparsity (convert to percentage)
    if 'sparsity' in metrics_list[0]:
        sparsities = [m.get('sparsity', 0) * 100 for m in metrics_list]  # Convert to percentage
        axes[1,1].semilogx(lambdas, sparsities, 'o-', linewidth=2, markersize=8, 
                          color='darkorange', markerfacecolor='white')
        axes[1,1].set_xlabel('Regularization Strength (λ)', fontsize=12)
        axes[1,1].set_ylabel('Sparsity (%)', fontsize=12)
        axes[1,1].set_title('Solution Sparsity vs λ', fontsize=14, pad=15)
        axes[1,1].set_ylim(0, 100)  # 0-100% range
        axes[1,1].grid(True, which="both", ls="--", alpha=0.7)
        axes[1,1].tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    plt.close()

def run_analysis(example_num=0, noise_level=0.0, lambda_values=None, auto_lambda=True):
    """Run complete analysis with visualization and metrics
    
    Args:
        example_num: Index of the example to run (0, 1, or 2)
        noise_level: Amount of Gaussian noise to add (0.0 to 1.0)
        lambda_values: List of lambda values for regularization strength
                      If None, will be automatically determined
        auto_lambda: Whether to use automatic lambda selection
    """
    # Load data
    x_tilde, A, x_true = load_data(example_num)
    
    # Add noise if specified
    if noise_level > 0:
        print(f"Adding {noise_level*100:.1f}% Gaussian noise...")
        x_tilde = add_noise(x_tilde, noise_level)
    
    # If lambda values not provided, use a range that encourages sparsity
    if lambda_values is None:
        # Use a wider range with more emphasis on larger lambdas to encourage sparsity
        lambda_values = np.logspace(-3, 1, num=20)  # 20 values from 0.01 to 100
        # Also add some specific values that work well for sparsity
        lambda_values = np.sort(np.unique(np.concatenate([
            lambda_values,
            np.logspace(0, 2, num=5)  # Add more points in the 1-100 range
        ])))
    
    # Store metrics for each lambda
    metrics_list = []
    
    # Define lambda values to test
    if lambda_values is None:
        lambda_values = np.logspace(-6, 2, 10)  # 10 points from 1e-6 to 100
    
    print(f"\nTesting {len(lambda_values)} lambda values:")
    for i, lambda_val in enumerate(lambda_values):
        print(f"\n[{i+1}/{len(lambda_values)}] Testing lambda = {lambda_val:.2e}")
        print("-" * 50)
        
        # Solve with current lambda
        result = solve_deblurring(x_tilde, A, lambda_val, verbose=True)
        
        if result is not None:
            x_opt, solve_time = result
            
            # Calculate metrics with solve_time
            metrics = calculate_metrics(x_opt, x_true, x_tilde, A, solve_time=solve_time)
            metrics['lambda'] = lambda_val
            metrics_list.append(metrics)
            
            # Print metrics
            print("\nMetrics:")
            print(f"  RMSE: {metrics.get('rmse', float('nan')):.4f}")
            print(f"  PSNR: {metrics.get('psnr', float('nan')):.2f} dB")
            print(f"  SSIM: {metrics.get('ssim', float('nan')):.4f}")
            sparsity = metrics.get('sparsity')
            if sparsity is not None and not np.isnan(sparsity):
                print(f"  Sparsity: {sparsity*100:.1f}%")
            print(f"  Solve time: {solve_time:.2f} seconds")
            
            # Plot results for this lambda
            if i == 0 or i == len(lambda_values) - 1 or (i+1) % 5 == 0:
                plot_comparison(x_tilde, x_opt, x_true, 
                              title_suffix=f" (λ={lambda_val:.2e}, Noise: {noise_level*100:.0f}%)")
                plt.close()
    
    # Sort metrics by lambda for plotting
    metrics_list.sort(key=lambda x: x['lambda'])
    
    # Generate metrics vs lambda plots
    if len(metrics_list) > 1:
        plot_metrics_vs_lambda(metrics_list, noise_level)
    
    # Visualize best result (based on RMSE if ground truth available, otherwise first lambda)
    best_idx = 0
    if x_true is not None and 'rmse' in metrics_list[0]:
        best_idx = np.argmin([m.get('rmse', np.inf) for m in metrics_list])
    
    best_lambda = metrics_list[best_idx]['lambda']
    result = solve_deblurring(x_tilde, A, best_lambda, verbose=False)
    if result is not None:
        x_opt, _ = result  # We only need the solution, not the solve_time here
        # Plot results with detailed visualizations
        plot_comparison(x_tilde, x_opt, x_true, 
                       f"λ={best_lambda:.1e}, Noise={noise_level*100:.1f}%")
    
    # Plot metrics vs lambda
    if x_true is not None and len(metrics_list) > 1:
        plt.figure(figsize=(15, 10))
        
        # Extract metrics
        lambdas = [m['lambda'] for m in metrics_list]
        rmses = [m.get('rmse', np.nan) for m in metrics_list]
        psnrs = [m.get('psnr', np.nan) for m in metrics_list]
        ssims = [m.get('ssim', np.nan) for m in metrics_list]
        sparsities = [m.get('sparsity', np.nan) for m in metrics_list]
        
        # RMSE plot
        plt.subplot(2, 2, 1)
        plt.semilogx(lambdas, rmses, 'o-', linewidth=2)
        plt.axvline(x=best_lambda, color='r', linestyle='--', 
                   label=f'Best λ={best_lambda:.1e}')
        plt.title('RMSE vs Regularization Strength (λ)')
        plt.xlabel('λ (log scale)')
        plt.ylabel('RMSE')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        
        # PSNR plot
        plt.subplot(2, 2, 2)
        plt.semilogx(lambdas, psnrs, 'o-', linewidth=2, color='green')
        plt.axvline(x=best_lambda, color='r', linestyle='--', 
                   label=f'Best λ={best_lambda:.1e}')
        plt.title('PSNR vs Regularization Strength (λ)')
        plt.xlabel('λ (log scale)')
        plt.ylabel('PSNR (dB)')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        
        # SSIM plot
        plt.subplot(2, 2, 3)
        plt.semilogx(lambdas, ssims, 'o-', linewidth=2, color='purple')
        plt.axvline(x=best_lambda, color='r', linestyle='--', 
                   label=f'Best λ={best_lambda:.1e}')
        plt.title('SSIM vs Regularization Strength (λ)')
        plt.xlabel('λ (log scale)')
        plt.ylabel('SSIM')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        
        # Sparsity plot
        plt.subplot(2, 2, 4)
        plt.loglog(lambdas, sparsities, 'o-', linewidth=2, color='orange')
        plt.axvline(x=best_lambda, color='r', linestyle='--', 
                   label=f'Best λ={best_lambda:.1e}')
        plt.title('Sparsity vs Regularization Strength (λ)')
        plt.xlabel('λ (log scale)')
        plt.ylabel('Sparsity (fraction of near-zero pixels)')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'metrics_vs_lambda_noise_{noise_level*100:.0f}%.png', 
                   bbox_inches='tight', dpi=300)
        plt.show()
    
    # Detailed analysis
    if x_true is not None:
        stats = analyze_solution(x_opt, x_true, x_tilde, A)
        print("\n=== Solution Analysis ===")
        print(f"{'Metric':<20} {'Value':<15} Description")
        print("-" * 50)
        print(f"{'min_val':<20} {stats['min_val']:<15.4f} Minimum pixel value in deblurred image")
        print(f"{'max_val':<20} {stats['max_val']:<15.4f} Maximum pixel value in deblurred image")
        print(f"{'mean_val':<20} {stats['mean_val']:<15.4f} Mean pixel value in deblurred image")
        print(f"{'sparsity':<20} {stats['sparsity']*100:<15.1f}% Fraction of near-zero pixels (<1e-6)")
        print(f"{'max_residual':<20} {stats['max_residual']:<15.4f} Maximum residual (A@x_opt - x_tilde)")
        print(f"{'mean_residual':<20} {stats['mean_residual']:<15.4f} Mean absolute residual")
        print(f"{'residual_std':<20} {stats['residual_std']:<15.4f} Standard deviation of residuals")
    
    return metrics_list
    

def main():
    # Example to run (0, 1, or 2)
    example_num = 1  # Start with example 0 which has ground truth
    
    # Load data
    print(f"Loading Example {example_num}...")
    x_tilde, A, x_true = load_data(example_num)

    # Run analysis with different noise levels
    noise_levels = [0.0, 0.05, 0.1]  # 0%, 5%, 10% noise
    
    # Dictionary to store results for comparison
    all_metrics = {}
    
    for noise_level in noise_levels:
        print(f"\n{'='*80}")
        print(f"RUNNING ANALYSIS WITH {noise_level*100:.0f}% NOISE")
        print('='*80)
        
        # Run analysis with current noise level
        metrics_list = run_analysis(example_num=example_num, 
                                  noise_level=noise_level)
        
        # Store metrics for comparison
        all_metrics[f"{noise_level*100:.0f}%"] = metrics_list
    
    # Compare results across different noise levels if we have ground truth
    if x_true is not None and len(noise_levels) > 1:
        print("\n" + "="*80)
        print("COMPARISON ACROSS NOISE LEVELS")
        print("="*80)
        
        # Create a figure to compare metrics across noise levels
        plt.figure(figsize=(15, 10))
        
        # Plot RMSE comparison
        plt.subplot(2, 2, 1)
        for noise_str, metrics in all_metrics.items():
            lambdas = [m['lambda'] for m in metrics]
            rmses = [m.get('rmse', np.nan) for m in metrics]
            plt.loglog(lambdas, rmses, 'o-', label=f'{noise_str} noise')
        plt.title('RMSE Comparison Across Noise Levels')
        plt.xlabel('λ (log scale)')
        plt.ylabel('RMSE')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        
        # Plot PSNR comparison
        plt.subplot(2, 2, 2)
        for noise_str, metrics in all_metrics.items():
            lambdas = [m['lambda'] for m in metrics]
            psnrs = [m.get('psnr', np.nan) for m in metrics]
            plt.semilogx(lambdas, psnrs, 'o-', label=f'{noise_str} noise')
        plt.title('PSNR Comparison Across Noise Levels')
        plt.xlabel('λ (log scale)')
        plt.ylabel('PSNR (dB)')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        
        # Plot SSIM comparison
        plt.subplot(2, 2, 3)
        for noise_str, metrics in all_metrics.items():
            lambdas = [m['lambda'] for m in metrics]
            ssims = [m.get('ssim', np.nan) for m in metrics]
            plt.semilogx(lambdas, ssims, 'o-', label=f'{noise_str} noise')
        plt.title('SSIM Comparison Across Noise Levels')
        plt.xlabel('λ (log scale)')
        plt.ylabel('SSIM')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        
        # Plot Sparsity comparison
        plt.subplot(2, 2, 4)
        for noise_str, metrics in all_metrics.items():
            lambdas = [m['lambda'] for m in metrics]
            sparsities = [m.get('sparsity', np.nan) for m in metrics]
            plt.loglog(lambdas, sparsities, 'o-', label=f'{noise_str} noise')
        plt.title('Sparsity Comparison Across Noise Levels')
        plt.xlabel('λ (log scale)')
        plt.ylabel('Sparsity (fraction)')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print summary of best results for each noise level
        print("\n=== Best Results Summary ===")
        print(f"{'Noise':<10} {'Best λ':<15} {'RMSE':<10} {'PSNR (dB)':<10} "
              f"{'SSIM':<10} {'Sparsity':<10}")
        print("-" * 70)
        
        for noise_str, metrics in all_metrics.items():
            if not metrics:
                continue
                
            # Find best lambda based on RMSE
            best_idx = np.argmin([m.get('rmse', np.inf) for m in metrics])
            best = metrics[best_idx]
            
            print(f"{noise_str:<10} {best['lambda']:<15.2e} "
                  f"{best.get('rmse', np.nan):<10.4f} "
                  f"{best.get('psnr', np.nan):<10.2f} "
                  f"{best.get('ssim', np.nan):<10.4f} "
                  f"{best.get('sparsity', np.nan):<9.2f}%")

if __name__ == "__main__":
    main()
