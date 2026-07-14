import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# --- 1. Data Loading ---
# Use the absolute path you provided
file_path = r'D:/deephash_original/data/imagenet/wanet.xlsx'

try:
    # Read Excel file
    df = pd.read_excel(file_path)

    # Extract the Hamming Distance column
    distances_raw = df['汉明距离'].dropna().values

    # --- Remove zero values (exclude from analysis) ---
    distances = distances_raw[distances_raw != 0]
    zeros_removed = len(distances_raw) - len(distances)

    print(f"✅ Data loaded successfully!")
    print(f"   Original total rows: {len(distances_raw)}")
    print(f"   Removed zero values: {zeros_removed}")
    print(f"   Remaining data points: {len(distances)}")
    print(f"   Data range (excluding zeros): {distances.min()} - {distances.max()}")

    # --- 2. Traditional Method (1.5σ principle) ---
    mu = np.mean(distances)
    sigma = np.std(distances)
    threshold_traditional = mu + 1.5 * sigma

    print(f"\n--- 📏 Traditional Method ---")
    print(f"   Mean (μ): {mu:.4f}")
    print(f"   Standard Deviation (σ): {sigma:.4f}")
    print(f"   🔴 Threshold (μ + 1.5σ): {threshold_traditional:.4f}")

    # --- 3. GMM Method (Gaussian Mixture Model) ---
    # Reshape data to 2D array (n_samples, n_features)
    X = distances.reshape(-1, 1)

    # Initialize GMM with two components (clean samples + poisoned samples)
    gmm = GaussianMixture(n_components=2, random_state=42, max_iter=300)
    gmm.fit(X)

    # Extract fitted parameters
    weights = gmm.weights_
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_).flatten()

    # Identify which is the clean distribution and which is the abnormal distribution
    # Larger Hamming distance typically indicates poisoned samples
    idx_abnormal = np.argmax(means)  # Index with larger mean
    idx_normal = 1 - idx_abnormal  # The other index

    mean_clean = means[idx_normal]
    mean_poison = means[idx_abnormal]

    print(f"\n--- 🧠 GMM Method (Bimodal Fitting) ---")
    print(f"   Clean samples center: {mean_clean:.4f}")
    print(f"   Poisoned samples center: {mean_poison:.4f}")
    print(f"   Component weights: Clean={weights[idx_normal]:.3f}, Poison={weights[idx_abnormal]:.3f}")

    # --- 4. Calculate GMM Threshold (Intersection Point) ---
    # Find the point where the two Gaussian distributions intersect
    # Generate a range of points for scanning
    x_range = np.linspace(min(distances), max(distances), 1000)

    # Calculate probability contributions from each component
    # p_k = w_k * N(x | mu_k, sigma_k)
    p1 = weights[0] * np.exp(-0.5 * ((x_range - means[0]) / stds[0]) ** 2) / stds[0]
    p2 = weights[1] * np.exp(-0.5 * ((x_range - means[1]) / stds[1]) ** 2) / stds[1]

    # Find the point where p1 and p2 are closest (intersection)
    diff = np.abs(p1 - p2)
    intersection_idx = np.argmin(diff)
    threshold_gmm_exact = x_range[intersection_idx]

    print(f"   🔵 GMM Decision Boundary (Intersection): {threshold_gmm_exact:.4f}")

    # --- 5. Final Comparison ---
    print("\n" + "=" * 40)
    print("📊 Final Threshold Comparison Report")
    print("=" * 40)
    print(f"Traditional Method (1.5σ):  {threshold_traditional:.4f}")
    print(f"GMM Method:                  {threshold_gmm_exact:.4f}")
    print("-" * 40)

    # Calculate percentage of data above each threshold
    above_traditional = np.sum(distances > threshold_traditional) / len(distances) * 100
    above_gmm = np.sum(distances > threshold_gmm_exact) / len(distances) * 100

    print(f"Data points above traditional threshold: {above_traditional:.2f}%")
    print(f"Data points above GMM threshold: {above_gmm:.2f}%")
    print("-" * 40)

    if threshold_gmm_exact < threshold_traditional:
        print("💡 Suggestion: GMM threshold is lower, can detect more potential attacks.")
    else:
        print("💡 Suggestion: GMM threshold is higher, may have lower false positive rate.")

    # --- 6. Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left subplot: Histogram with thresholds
    ax1.hist(distances, bins=100, density=True, alpha=0.6, color='gray', label='Data Distribution')
    ax1.axvline(threshold_traditional, color='red', linestyle='--', linewidth=2,
                label=f'Traditional Threshold {threshold_traditional:.2f}')
    ax1.axvline(threshold_gmm_exact, color='blue', linestyle='-', linewidth=2,
                label=f'GMM Threshold {threshold_gmm_exact:.2f}')
    ax1.axvline(mean_clean, color='green', linestyle=':', linewidth=1.5, alpha=0.7,
                label=f'Clean Center {mean_clean:.2f}')
    ax1.axvline(mean_poison, color='orange', linestyle=':', linewidth=1.5, alpha=0.7,
                label=f'Poison Center {mean_poison:.2f}')

    ax1.set_title(f'Hamming Distance Distribution (Zeros Removed, n={len(distances)})', fontsize=12)
    ax1.set_xlabel('Hamming Distance', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(axis='y', alpha=0.7)

    # Right subplot: Zoomed view for better clarity
    # Focus on the region where most data lies (5th to 95th percentile)
    lower_bound = np.percentile(distances, 5)
    upper_bound = np.percentile(distances, 95)

    ax2.hist(distances, bins=100, density=True, alpha=0.6, color='gray', label='Data Distribution')
    ax2.axvline(threshold_traditional, color='red', linestyle='--', linewidth=2,
                label=f'Traditional Threshold {threshold_traditional:.2f}')
    ax2.axvline(threshold_gmm_exact, color='blue', linestyle='-', linewidth=2,
                label=f'GMM Threshold {threshold_gmm_exact:.2f}')
    ax2.axvline(mean_clean, color='green', linestyle=':', linewidth=1.5, alpha=0.7,
                label=f'Clean Center {mean_clean:.2f}')
    ax2.axvline(mean_poison, color='orange', linestyle=':', linewidth=1.5, alpha=0.7,
                label=f'Poison Center {mean_poison:.2f}')

    ax2.set_xlim(lower_bound, upper_bound)
    ax2.set_title(f'Zoomed View ({lower_bound:.0f} - {upper_bound:.0f} Percentile Range)', fontsize=12)
    ax2.set_xlabel('Hamming Distance', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(axis='y', alpha=0.7)

    plt.tight_layout()
    plt.show()

    # --- 7. Additional Statistics ---
    print("\n" + "=" * 40)
    print("📈 Additional Statistics (Zeros Excluded)")
    print("=" * 40)
    print(f"Median: {np.median(distances):.4f}")
    print(f"Mode: {float(pd.Series(distances).mode().iloc[0]):.4f}")
    print(f"Variance: {np.var(distances, ddof=1):.4f}")
    print(f"Interquartile Range (IQR): {np.percentile(distances, 75) - np.percentile(distances, 25):.4f}")
    print(f"Skewness: {float(pd.Series(distances).skew()):.4f}")
    print(f"Kurtosis: {float(pd.Series(distances).kurtosis()):.4f}")

except FileNotFoundError:
    print(f"❌ Error: File not found at {file_path}")
    print("Please check if the file path is correct and the file exists.")
except KeyError:
    print(f"❌ Error: Column '汉明距离' not found in the Excel file.")
    print("Please check the column name in your Excel file.")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")
    print("Please check if the file path is correct or if the Excel file is being used by another program.")