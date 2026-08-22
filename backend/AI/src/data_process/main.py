import os
import pandas as pd
from .data_processor import DataProcessor
from .data_retriever import MedicalKnowledgeRetriever

# Load dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
labeled_path = os.path.join(script_dir, '..', '..', 'data', 'labeled', 'pubmedqa_labeled.csv')

labeled_data_df = pd.read_csv(labeled_path)

# Initialise and execute the pipeline
processor = DataProcessor(raw_dataframe=labeled_data_df)
processor.clean()
processor.preprocess()
processor.stratified_split()
train_loader, val_loader, test_loader = processor.get_dataloader(batch_size=8)

# Initialise and execute retrieval pipeline
retriever = MedicalKnowledgeRetriever()
retriever.build_knowledge_base(labeled_data_df)
retriever.save_database(index_path=os.path.join(script_dir, '..', '..', 'saved_model', 'faiss_medical.index'))