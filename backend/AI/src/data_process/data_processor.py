import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from .data_ingestion import MedicalQADataset

model_name = 'dmis-lab/biobert-v1.1'
class DataProcessor: 
    def __init__(self, raw_dataframe, model_name=model_name, max_len=512):
        self.raw_df = raw_dataframe
        self.processed_df = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len

        # Placeholders
        self.train_df = None
        self.val_df = None
        self.test_df = None
        
    def clean(self):
        df = self.raw_df[['question', 'context', 'label_decision']].copy()
        
        # Drop rows with missing values in the target columns
        df.dropna(subset=['question', 'context', 'label_decision'], inplace=True)

        # Drop duplicate rows
        df.drop_duplicates(subset=['question', 'context'], inplace=True)

        # Ensure question and context are strings
        df['question'] = df['question'].astype(str)
        df['context'] = df['context'].astype(str)

        self.processed_df = df
        
        print(f"Data cleaning complete. Remaining length: {len(self.processed_df)}")
        
    def preprocess(self):
        if self.processed_df is None:
            raise ValueError("Missing cleaned data")
        
        label_mapping = {'No': 0, 'Yes': 1, 'Maybe': 2}
        
        # Map label to int
        self.processed_df['label_decision'] = self.processed_df['label_decision'].map(label_mapping)
        
        # Drop rows where label is missing (e.g. '-')
        self.processed_df = self.processed_df.dropna(subset=['label_decision'])
        
        # Ensure label_decision is integer
        self.processed_df['label_decision'] = self.processed_df['label_decision'].astype(int)
        
        print(f"Data preprocessing complete. Remaining length: {len(self.processed_df)}")
        
    def stratified_split(self, test_size=0.2, random_state=42):
        if self.processed_df is None: 
            raise ValueError("Missing preprocessed data")
        
        # Split the data into training and temp (test + validation)
        self.train_df, temp_df = train_test_split(
            self.processed_df,
            test_size = test_size,
            random_state = random_state,
            stratify = self.processed_df['label_decision']
        )

        # Split temp into validation and test sets
        self.val_df, self.test_df = train_test_split(
            temp_df,
            test_size = 0.5,
            random_state = random_state,
            stratify = temp_df['label_decision']
        )
        
        print(f"Split data into training, validation, and test sets")
        print(f"Training set: {len(self.train_df)} samples")
        print(f"Validation set: {len(self.val_df)} samples")
        print(f"Test set: {len(self.test_df)} samples")
        
    def get_dataloader(self, batch_size = 8):
        if self.train_df is None:
            raise ValueError("Missing stratified training data")

        # Instantiate the PyTorch Datasets
        train_dataset = MedicalQADataset(self.train_df, self.tokenizer, self.max_len)
        val_dataset = MedicalQADataset(self.val_df, self.tokenizer, self.max_len)
        test_dataset = MedicalQADataset(self.test_df, self.tokenizer, self.max_len)

        # Create DataLoaders for batching
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print("DataLoaders generated and ready for model fine-tuning.")
        return train_loader, val_loader, test_loader
        
        
        
        
        
        
    

