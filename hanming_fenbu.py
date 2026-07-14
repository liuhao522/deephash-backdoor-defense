import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set English font and chart style
plt.rcParams['font.family'] = 'DejaVu Sans'  # Use English supported font
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('default')


def main():
    # Load Excel file
    file_path = 'D:/deephash_original/dataset/excel/cifar10_youxia.xlsx'

    try:
        # Read Excel file with proper encoding and all data
        df = pd.read_excel(file_path, engine='openpyxl')

        print("Data loaded successfully!")
        print(f"Data shape: {df.shape}")
        print("\nColumn names:")
        print(df.columns.tolist())
        print("\nFirst 5 rows:")
        print(df.head())

        # Check if Hamming distance column exists
        hamming_column = '汉明距离'

        if hamming_column not in df.columns:
            print(f"Error: Column '{hamming_column}' does not exist")
            print("Available columns:", df.columns.tolist())
            return

        print(f"Using Hamming distance column: '{hamming_column}'")

        # Extract Hamming distances
        hamming_distances = df[hamming_column].dropna().values

        print(f"\nTotal rows in Excel: {len(df)}")
        print(f"Valid Hamming distances found: {len(hamming_distances)}")
        print(f"Rows with NaN values: {len(df) - len(hamming_distances)}")

        if len(hamming_distances) == 0:
            print("Error: No valid Hamming distances found")
            return

        # Print basic statistics
        print(f"\nHamming distance statistics:")
        print(f"Min: {np.min(hamming_distances):.2f}")
        print(f"Max: {np.max(hamming_distances):.2f}")
        print(f"Mean: {np.mean(hamming_distances):.2f}")
        print(f"Median: {np.median(hamming_distances):.2f}")
        print(f"Std: {np.std(hamming_distances):.2f}")

        # Create distribution plot
        plt.figure(figsize=(14, 8))

        # Calculate appropriate bins
        max_distance = int(np.max(hamming_distances))
        min_distance = int(np.min(hamming_distances))

        # Create bins for histogram
        if max_distance - min_distance > 100:
            # For large range, use more bins
            bins_num = 50
        else:
            bins_num = max_distance - min_distance + 1

        # Plot histogram
        n, bins, patches = plt.hist(hamming_distances, bins=bins_num,
                                    alpha=0.75, edgecolor='black',
                                    density=True, rwidth=0.8,
                                    color='skyblue', linewidth=1.2)

        plt.xlabel('Hamming Distance', fontsize=14, fontweight='bold')
        plt.ylabel('Frequency', fontsize=14, fontweight='bold')
        plt.title('Hamming Distance Distribution of MNIST Hash Codes',
                  fontsize=16, fontweight='bold', pad=20)

        plt.grid(True, alpha=0.3, linestyle='--')

        # Add statistical information
        mean_dist = np.mean(hamming_distances)
        median_dist = np.median(hamming_distances)
        std_dist = np.std(hamming_distances)

        plt.axvline(mean_dist, color='red', linestyle='--', linewidth=2.5,
                    label=f'Mean: {mean_dist:.2f}')
        plt.axvline(median_dist, color='green', linestyle='--', linewidth=2.5,
                    label=f'Median: {median_dist:.2f}')

        # Set x-axis ticks appropriately
        x_ticks = np.arange(min_distance, max_distance + 1, step=max(1, (max_distance - min_distance) // 10))
        plt.xticks(x_ticks, fontsize=10)
        plt.yticks(fontsize=10)

        # Add legend
        plt.legend(fontsize=12, loc='upper right', framealpha=0.9)

        # Add text box with statistics
        stats_text = (f'Statistics:\nSamples: {len(hamming_distances)}\n'
                      f'Mean: {mean_dist:.2f}\n'
                      f'Median: {median_dist:.2f}\n'
                      f'Std: {std_dist:.2f}\n'
                      f'Min: {np.min(hamming_distances):.1f}\n'
                      f'Max: {np.max(hamming_distances):.1f}')

        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                 fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()

        # Save image with high quality
        output_path = 'D:/deephash_original/hamming_distance_distribution_from_excel.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\nImage saved to: {output_path}")

        plt.show()

        # Print detailed statistics
        print("\n=== Detailed Hamming Distance Statistics ===")
        print(f"Total samples: {len(hamming_distances):,}")
        print(f"Hamming distance range: [{np.min(hamming_distances):.2f}, {np.max(hamming_distances):.2f}]")
        print(f"Mean Hamming distance: {mean_dist:.2f}")
        print(f"Median Hamming distance: {median_dist:.2f}")
        print(f"Standard deviation: {std_dist:.2f}")
        print(f"Variance: {np.var(hamming_distances):.2f}")

        # Show distance distribution percentages
        print("\nDistance distribution percentages:")

        # Create distance ranges for percentage calculation
        distance_ranges = [
            (0, 10, "0-10"),
            (10, 20, "10-20"),
            (20, 30, "20-30"),
            (30, 40, "30-40"),
            (40, 50, "40-50"),
            (50, 60, "50-60"),
            (60, 70, "60-70"),
            (70, 80, "70-80"),
            (80, 90, "80-90"),
            (90, 100, "90-100")
        ]

        for low, high, label in distance_ranges:
            if high > max_distance:
                continue
            count = np.sum((hamming_distances >= low) & (hamming_distances < high))
            percentage = (count / len(hamming_distances)) * 100
            if percentage > 0.01:  # Only show significant percentages
                print(f"{label}: {percentage:.2f}% ({count:,} samples)")

        # Show specific distance values that have significant counts
        print(f"\nSpecific distance values with significant counts:")
        unique_distances, counts = np.unique(hamming_distances, return_counts=True)
        for dist, count in zip(unique_distances, counts):
            percentage = (count / len(hamming_distances)) * 100
            if percentage > 0.1:  # Only show distances with > 0.1% occurrence
                print(f"Distance {int(dist)}: {percentage:.2f}% ({count:,} samples)")

        # Additional analysis
        print(f"\nAdditional analysis:")
        print(f"Distance < 32 (half of 64 bits): {np.mean(hamming_distances < 32) * 100:.2f}%")
        print(f"Distance 32-64: {np.mean((hamming_distances >= 32) & (hamming_distances < 64)) * 100:.2f}%")
        print(f"Distance >= 64: {np.mean(hamming_distances >= 64) * 100:.2f}%")

        # Check if distances are precomputed pairwise or per sample
        if len(hamming_distances) == len(df):
            print(f"\nNote: Hamming distances appear to be per sample (not pairwise)")
            print(f"Each value represents the distance for one sample")
        else:
            print(f"\nNote: Hamming distances appear to be pairwise comparisons")
            print(f"Total pairwise comparisons: {len(hamming_distances):,}")

    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        print("Please check the file path")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()