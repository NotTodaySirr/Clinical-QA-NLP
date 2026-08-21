import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Directories Verification
script_dir = os.path.dirname(os.path.abspath(__file__))
raw_dir = os.path.join(script_dir, '..', 'data', 'raw')
processed_dir = os.path.join(script_dir, '..', 'data', 'processed')

if not os.path.exists(raw_dir):
    os.makedirs(raw_dir)
else:
    pass

if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)
else:
    pass

# Dataset Download
print("Download Dataset from HuggingFace")
dataset = load_dataset('qiaojin/PubMedQA', 'pqa_unlabeled')

df = pd.DataFrame(dataset['train'])
df.head()

raw_path = os.path.join(raw_dir, 'pubmedqa_unlabeled_raw.csv')
df.to_csv(raw_path, index=False)

print(f"Raw data save to {raw_path}")