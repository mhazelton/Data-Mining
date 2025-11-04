# Max Hazelton
# CISC-4631-R01, Lab 1
# Min-Max Normalization and Z-Score Normalization
# Input: .txt file with two columns: <label> <numeric_value> (one entry per line)
#
# Notes:
# - Z-score uses the population standard deviation (divide by N)
# - The data file is assumed to be in the SAME FOLDER as this Python file.

import math
import os

NAME = "Max Hazelton"  # <-- Put your name here to appear in the output

def read_txt_dataset(path):
    """
    Reads a simple two-column .txt dataset.
    Expected format per line: <label> <numeric_value>
    Lines that cannot be parsed are ignored.
    Returns: (labels, values) where
      labels: list[str], values: list[float]
    """
    labels = []
    values = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            label = parts[0]
            try:
                val = float(parts[-1].replace(',', ''))
            except ValueError:
                # Not a data line (e.g., header). Skip it
                continue
            labels.append(label)
            values.append(val)
    return labels, values

def min_max_normalize(values):
    """
    Min-Max normalization to [0,1].
    x' = (x - min) / (max - min)
    If all values are equal, returns all zeros 
    """
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if math.isclose(vmin, vmax):
        return [0.0 for _ in values]
    return [(x - vmin) / (vmax - vmin) for x in values]

def zscore_normalize(values):
    """
    Z-score normalization (population std):
    z = (x - mean) / std, where std = sqrt( sum((x-mean)^2)/N )
    If std is zero, returns all zeros to avoid division by zero.
    """
    if not values:
        return []
    n = len(values)
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / n  # population variance
    std = math.sqrt(var)
    if math.isclose(std, 0.0):
        return [0.0 for _ in values]
    return [(x - mean) / std for x in values]

def pick(values_by_label, label_order, targets):
    """
    Helper to pick values for specific labels (e.g., employees C, K, Q).
    Returns dict[label] = value for labels that exist in the dataset.
    """
    idx_map = {lab: i for i, lab in enumerate(label_order)}
    out = {}
    for t in targets:
        if t in idx_map:
            out[t] = values_by_label[idx_map[t]]
    return out

def main():
    print(f"This is {NAME}'s Min-Max and Z-Score Normalization Program")
    input("Press Enter to run the program using 'salaries.txt' from this folder...")

    # Always use salaries.txt next to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = "salaries.txt"
    path = os.path.join(script_dir, filename)

    if not os.path.exists(path):
        print(f"File not found: {filename}")
        print("Make sure the .txt file is in the same folder as this Python file.")
        return

    labels, values = read_txt_dataset(path)
    if not values:
        print("No valid data found. Please check the file format.")
        return

    # Min-Max
    mm = min_max_normalize(values)
    # Z-Score
    zs = zscore_normalize(values)

    # Show small preview
    print("\n--- Summary ---")
    print(f"Count: {len(values)}  Min: {min(values)}  Max: {max(values)}  Mean: {sum(values)/len(values):.4f}")

    # Prompt user which labels to inspect (comma-separated), or press Enter to use examples
    picks = input("Enter labels to display (comma-separated), or press Enter for examples: ").strip()
    if picks:
        target_labels = [p.strip() for p in picks.split(",") if p.strip()]
        mm_pick = pick(mm, labels, target_labels)
        zs_pick = pick(zs, labels, target_labels)
    else:
        # Examples from the assignment text:
        # Min-Max: C, K, Q
        # Z-Score: D, G, Q
        mm_pick = pick(mm, labels, ["C", "K", "Q"])
        zs_pick = pick(zs, labels, ["D", "G", "Q"])

    print("\n--- Min-Max Normalization [0,1] ---")
    for lab, val in mm_pick.items():
        print(f"{lab}: {val:.6f}")

    print("\n--- Z-Score Normalization ---")
    for lab, val in zs_pick.items():
        print(f"{lab}: {val:.6f}")

if __name__ == "__main__":
    main()

