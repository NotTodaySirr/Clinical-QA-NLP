import ast
import re
from typing import Any
import torch
from torch.utils.data import Dataset


def clean_context_text(raw_context: Any) -> str:
    """
    Cleans raw context string (which might be a stringified python dictionary or JSON)
    into a coherent, structured medical text: 'SECTION: Sentence. SECTION: Sentence.'
    Prevents token budget waste on JSON/dict keys and ensures crucial findings fit within
    the token budget.
    """
    if raw_context is None:
        return ""
    if not isinstance(raw_context, str):
        if isinstance(raw_context, dict):
            contexts = raw_context.get("contexts", [])
            labels = raw_context.get("labels", [])
            if contexts and labels and len(contexts) == len(labels):
                sections = [
                    f"{str(lbl).strip().upper()}: {str(txt).strip()}"
                    for lbl, txt in zip(labels, contexts)
                    if str(txt).strip()
                ]
                return " ".join(sections)
            elif contexts:
                return " ".join([str(c).strip() for c in contexts if str(c).strip()])
        return str(raw_context).strip()

    raw_context = raw_context.strip()
    if raw_context.startswith("{") and ("contexts" in raw_context or "labels" in raw_context):
        try:
            parsed = ast.literal_eval(raw_context)
            if isinstance(parsed, dict):
                contexts = parsed.get("contexts", [])
                labels = parsed.get("labels", [])
                if contexts and labels and len(contexts) == len(labels):
                    sections = [
                        f"{str(lbl).strip().upper()}: {str(txt).strip()}"
                        for lbl, txt in zip(labels, contexts)
                        if str(txt).strip()
                    ]
                    return " ".join(sections)
                elif contexts:
                    return " ".join([str(c).strip() for c in contexts if str(c).strip()])
        except Exception:
            pass

    return re.sub(r"\s+", " ", raw_context).strip()


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
        """
        Initialisation
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        """
        Returns the length of the dataset
        """
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        """
            Description: Get the item at the given index
            Process:
                1. Get the question, context, and label
                2. Clean the context string into structured medical text
                3. Tokenise the question and context
                4. Return the tokenised data
            Input:
                idx: Index of the item
            Output:
                Tokenised data
        """
        question = str(self.dataframe.loc[idx, 'question'])
        raw_context = self.dataframe.loc[idx, 'context']
        context = clean_context_text(raw_context)
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