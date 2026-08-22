import os
import pandas as pd

from .data_processor import DataProcessor
from .data_retriever import MedicalKnowledgeRetriever

script_dir = os.path.dirname(os.path.abspath(__file__))

# Path to labeled data (for model training)
# TODO: Replace with the actual path to your labeled dataset
labeled_csv_path = os.path.join(script_dir, '..', '..', 'data', 'labeled', 'pubmedqa_labeled.csv')

# Path to unlabeled cleaned data (for retrieval knowledge base)
unlabeled_cleaned_path = os.path.join(script_dir, '..', '..', 'data', 'cleaned', 'pubmedqa_unlabeled_cleaned.csv')

# Process labeled data for training
processor = DataProcessor(cleaned_csv_path=labeled_csv_path)
processor.preprocess()
processor.stratified_split()
train_loader, val_loader, test_loader = processor.get_dataloader(batch_size=8)

# Build Retrieval Knowledge Base from unlabeled cleaned data
unlabeled_df = pd.read_csv(unlabeled_cleaned_path)

retriever = MedicalKnowledgeRetriever()
retriever.build_knowledge_base(unlabeled_df)
retriever.save_database(
    index_path=os.path.join(script_dir, '..', '..', 'saved_model', 'faiss_medical.index')
)

