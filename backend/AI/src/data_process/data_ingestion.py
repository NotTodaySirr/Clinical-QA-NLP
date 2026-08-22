import torch
from torch.utils.data import Dataset

class MedicalQADataset(Dataset):
    """
    Description: This class is used to load the dataset and prepare it for training.
    Process: 
        1. Load the dataset
        2. Tokenize the dataset
        3. Return the dataset
    Input:
        dataframe: Pandas DataFrame with columns ['question', 'context', 'label_decision']
        tokenizer: Hugging Face tokenizer
        max_length: Maximum sequence length
    Output:
        Pytorch dataset
    """
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        question = str(self.dataframe.loc[idx, 'question'])
        context = str(self.dataframe.loc[idx, 'context'])
        label = self.dataframe.loc[idx, 'label_decision']

        # Sequence pairing, padding, and truncation
        encoding = self.tokenizer(
            question,
            context,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }