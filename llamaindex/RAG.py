from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.ollama import OllamaEmbedding

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

from llama_index.llms.ollama import Ollama

documents = SimpleDirectoryReader(
    input_files= ["paul_graham_essay.txt"]
).load_data()

ollama_embedding = OllamaEmbedding(
    model_name="mxbai-embed-large",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)

# create client
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("paul_collection")

# save embedding to disk
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# create index
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=ollama_embedding
)

# load from disk
db2 = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db2.get_or_create_collection("paul_collection")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=ollama_embedding,
)

llm = Ollama(model="llama3.1", request_timeout=420.0)
# request_timeout=120.0: This argument sets a timeout for requests 
# to the language model. If the model doesn't respond within 120 seconds, 
# the request will time out.

# Query Data from the persisted index
query_engine = index.as_query_engine(llm=llm)

response = query_engine.query("What did the author do growing up?")
print(response)

