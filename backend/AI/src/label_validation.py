import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
labeled_dir = os.path.join(script_dir, '..', 'data', 'labeled')
processed_dir = os.path.join(script_dir, '..', 'data', 'processed')

# Ensure processed directory exists
os.makedirs(processed_dir, exist_ok=True)

# Load the provided labeled dataset
labeled_path = os.path.join(labeled_dir, 'pubmedqa_labeled.csv')
print(f"Loading labeled data from {labeled_path}...")
df = pd.read_csv(labeled_path)

# Step 1: Randomly take 20% of the dataset
_, sample_20_percent = train_test_split(df, test_size=0.20, random_state=42)

# Step 2: Divide that 20% into two equal smaller files (50% of the 20% each)
val_set_1, val_set_2 = train_test_split(sample_20_percent, test_size=0.50, random_state=42)

print(f"Original labeled dataset size: {len(df)} rows")
print(f"20% sampled dataset size: {len(sample_20_percent)} rows")
print(f"Validation Set 1 size: {len(val_set_1)} rows")
print(f"Validation Set 2 size: {len(val_set_2)} rows")

# Save the validation sets
val_1_path = os.path.join(processed_dir, 'validation_set_1.csv')
val_2_path = os.path.join(processed_dir, 'validation_set_2.csv')

val_set_1.to_csv(val_1_path, index=False)
val_set_2.to_csv(val_2_path, index=False)

print(f"Saved {val_1_path}")
print(f"Saved {val_2_path}")
