from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS

# Example documents
docs = [
    "This is the first document.",
    "This is the second document.",
    # Add more documents as needed
]

# Load a pre-trained SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for the documents
embeddings = model.encode(docs)

# Debugging: Print the embeddings
print("Embeddings:", embeddings)

# Ensure embeddings are in the correct shape
if embeddings.shape[0] > 0:
    # Create a FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)

    # Add embeddings to the FAISS index
    index.add(np.array(embeddings))

    # Example query to find similar documents
    query = "This is a query document."
    query_embedding = model.encode([query])
    
    # Search the FAISS index
    D, I = index.search(np.array(query_embedding), k=5)
    
    # Print the results
    print("Distances:", D)
    print("Indices:", I)
else:
    print("No embeddings were generated. Please check your documents and embedding logic.")
