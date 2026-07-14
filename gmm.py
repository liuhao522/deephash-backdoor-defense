import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# --- 1. Data Loading ---
# Use the absolute path you provided
file_path = r'D:/deephash_original/data/imagenet/sig.xlsx'

try:
    # Read Excel file
    df = pd.read_excel(file_path)

    # Extract the Hamming Distance column (assumed column name is '汉明距离')
    # dropna() removes missing values, .values converts to array
    distances = df['汉明距离'].dropna().values

    print(f"✅ Data loaded successfully! Total rows: {len(distances)}")
    print(f"   Data range: {distances.min()} - {distances.max()}")

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
    if threshold_gmm_exact < threshold_traditional:
        print("💡 Suggestion: GMM threshold is lower, can detect more potential attacks.")
    else:
        print("💡 Suggestion: GMM threshold is higher, may have lower false positive rate.")

    # --- 6. Plotting ---
    plt.figure(figsize=(12, 6))
    plt.hist(distances, bins=100, density=True, alpha=0.6, color='gray', label='Data Distribution')
    plt.axvline(threshold_traditional, color='red', linestyle='--', linewidth=2,
                label=f'Traditional Threshold {threshold_traditional:.2f}')
    plt.axvline(threshold_gmm_exact, color='blue', linestyle='-', linewidth=2,
                label=f'GMM Threshold {threshold_gmm_exact:.2f}')

    # Add vertical lines for the GMM component means
    plt.axvline(mean_clean, color='green', linestyle=':', linewidth=1.5, alpha=0.7,
                label=f'Clean Center {mean_clean:.2f}')
    plt.axvline(mean_poison, color='orange', linestyle=':', linewidth=1.5, alpha=0.7,
                label=f'Poison Center {mean_poison:.2f}')

    plt.title(f'Hamming Distance Distribution with Threshold Detection (Total: {len(distances)})', fontsize=14)
    plt.xlabel('Hamming Distance', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(axis='y', alpha=0.7)
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"❌ Error: File not found at {file_path}")
    print("Please check if the file path is correct and the file exists.")
except KeyError:
    print(f"❌ Error: Column '汉明距离' not found in the Excel file.")
    print("Please check the column name in your Excel file.")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")
    print("Please check if the file path is correct or if the Excel file is being used by another program.")