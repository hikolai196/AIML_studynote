from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.ollama import OllamaEmbedding
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex

# Load documents
documents = SimpleDirectoryReader(
    input_files=["paul_graham_essay.txt"]
).load_data()

# Initialize embedding model
ollama_embedding = OllamaEmbedding(
    model_name="mxbai-embed-large",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)

# Create Chroma client and collection
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("paul_collection")

# Create vector store
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Save embedding to disk
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create index and store in "paul_collection"
index = VectorStoreIndex.from_documents(
    documents, 
    storage_context=storage_context, 
    embed_model=ollama_embedding
)

print("Vector database built and stored successfully.")