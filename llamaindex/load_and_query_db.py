import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Initialize embedding model
ollama_embedding = OllamaEmbedding(
    model_name="mxbai-embed-large",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)

# Create Chroma client and collection
db2 = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db2.get_or_create_collection("paul_collection")

# Create vector store
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Load from vector store
index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=ollama_embedding,
)

# Initialize language model
llm = Ollama(model="gemma2:2b", request_timeout=420.0, temperature=0.1)

# Query Data from the persisted index
query_engine = index.as_query_engine(llm=llm)

response = query_engine.query("Who is the author?")
print(response)