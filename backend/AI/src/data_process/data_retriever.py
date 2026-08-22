import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Use embedding model 
model_name = 'S-PubMedBert-MS-MARCO'

class MedicalKnowledgeRetriever:
    """
    Description: This class is used to retrieve the most relevant context to the question
    Process: 
        1. Build the knowledge base
        2. Save the knowledge base
        3. Load the knowledge base
        4. Retrieve the context
    Input:
        raw_dataframe: Pandas DataFrame with columns ['question', 'context', 'label_decision']
        model_name: Embedding model name
        top_k: Number of relevant contexts to retrieve
    Output:
        Most relevant context to the question
    """
    def __init__(self, model=model_name):
        """
        Initialisation
        """ 
        self.embedder = SentenceTransformer(model)
        self.index = None
        self.contexts = []
        
    def build_knowledge_base(self, dataframe):
        """
        Description: Build the knowledge base
        Process: 
            1. Copy the raw dataset
            2. Drop rows with missing values in the target columns
            3. Drop duplicate rows
            4. Ensure question and context are strings
        Input:
            raw_dataframe: Pandas DataFrame with columns ['question', 'context', 'label_decision']
        Output:
            Cleaned DataFrame
        """
        # Extract unique medical abstracts to avoid duplicating data
        self.contexts = dataframe['context'].dropna().unique().tolist()
        print(f"Generating embeddings for {len(self.contexts)} unique contexts...")
        
        # Convert text to dense numerical vectors
        embeddings = self.embedder.encode(
            self.contexts, 
            show_progress_bar=True, 
            convert_to_numpy=True
        )
        
        # Initialise FAISS using L2 (Euclidean) distance
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        
        # Populate the database
        self.index.add(embeddings)
        print("FAISS Knowledge Base built successfully.")
        
    def save_database(self, index_path="faiss_medical.index", text_path="contexts.npy"):
        """
        Description: Save the knowledge base
        Process: 
            1. Copy the raw dataset
            2. Drop rows with missing values in the target columns
            3. Drop duplicate rows
            4. Ensure question and context are strings
        Input:
            raw_dataframe: Pandas DataFrame with columns ['question', 'context', 'label_decision']
        Output:
            Cleaned DataFrame
        """
        if self.index is None:
            raise ValueError("Knowledge base is empty.")
            
        faiss.write_index(self.index, index_path)
        np.save(text_path, np.array(self.contexts))
        print("Database saved to disk.")

    def load_database(self, index_path="faiss_medical.index", text_path="contexts.npy"):
        """
        Description: Load the knowledge base
        Process: 
            1. Copy the raw dataset
            2. Drop rows with missing values in the target columns
            3. Drop duplicate rows
            4. Ensure question and context are strings
        Input:
            raw_dataframe: Pandas DataFrame with columns ['question', 'context', 'label_decision']
        Output:
            Cleaned DataFrame
        """ 
        self.index = faiss.read_index(index_path)
        self.contexts = np.load(text_path, allow_pickle=True).tolist()
        print("Database loaded successfully.")
        
    def retrieve_context(self, question, top_k=1):
        """
        Description: Retrieve the context
        Process: 
            1. Copy the raw dataset
            2. Drop rows with missing values in the target columns
            3. Drop duplicate rows
            4. Ensure question and context are strings
        Input:
            raw_dataframe: Pandas DataFrame with columns ['question', 'context', 'label_decision']
        Output:
            Cleaned DataFrame
        """ 
        if self.index is None:
            raise ValueError("Database not loaded.")
            
        # Embed the user's question
        question_vector = self.embedder.encode([question], convert_to_numpy=True)
        
        # Perform similarity search (returns distances and array indices)
        distances, indices = self.index.search(question_vector, top_k)
        
        # Extract the best matching text using the returned index
        best_match_idx = indices[0][0]
        best_context = self.contexts[best_match_idx]
        
        return best_context
    
    
    