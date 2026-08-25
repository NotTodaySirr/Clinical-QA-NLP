import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from .data_ingestion import MedicalQADataset

# Default tokeniser
model_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'

class DataProcessor:
    """
    Description: Loads a pre-cleaned CSV and prepares it for model training.
    Process:
        1. Load cleaned CSV from disk
        2. Preprocess labels
        3. Stratified split into train/val/test
        4. Return PyTorch DataLoaders
    Input:
        cleaned_csv_path: Path to the cleaned CSV produced by DataCleaner
        model_name: BioBERT tokeniser name
        max_len: Maximum sequence length
    Output:
        PyTorch DataLoaders
    """
    def __init__(self, cleaned_csv_path: str, model_name=model_name, max_len=512):
        """
        Initialisation — loads the cleaned CSV from disk.

        Args:
            cleaned_csv_path: Path to the cleaned CSV produced by DataCleaner
            model_name: BioBERT tokeniser name
            max_len: Maximum sequence length
        """
        print(f"Loading cleaned data from {cleaned_csv_path}...")
        self.processed_df = pd.read_csv(cleaned_csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len

        # Placeholders
        self.train_df = None
        self.val_df = None
        self.test_df = None

    def preprocess(self):
        """
        Description: Preprocess the dataset
        Process: 
            1. Map label_decision to integers
            2. Drop rows with missing values in the label_decision column
            3. Ensure label_decision is integer
        Input:
            Cleaned DataFrame
        Output:
            Processed DataFrame
        """ 
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
        """
        Description: Get the dataloader for the dataset
        Process: 
            1. Instantiate the PyTorch Datasets
            2. Create DataLoaders for batching
        Input:
            batch_size: Batch size for the dataloader
        Output:
            DataLoaders for training, validation, and testing
        """ 
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
        
        
        
        
        
        
    

