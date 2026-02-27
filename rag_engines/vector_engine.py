import os
import time
from typing import List, Optional, Union
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, ServiceContext
from utils.performance_tracker import QueryResult


class VectorRAGEngine:
    """Traditional Vector-based RAG implementation using ChromaDB"""
    
    def __init__(self, model_name: str = "llama3.1"):
        self.model_name = model_name
        self.index = None
        self.query_engine = None
        self.build_time = 0.0
        # Initialize global settings immediately to prevent OpenAI defaults
        self.initialize_llm_and_embeddings()
        
    def initialize_llm_and_embeddings(self):
        """Initialize Ollama LLM and HuggingFace embeddings"""
        try:
            # Initialize Ollama LLM
            llm = Ollama(model=self.model_name)
            
            # Initialize HuggingFace embeddings
            embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en-v1.5"
            )
            
            # Set global settings to avoid OpenAI defaults
            Settings.llm = llm
            Settings.embed_model = embed_model
            # Set default chunk size
            Settings.chunk_size = 512
            
            return True
        except Exception as e:
            raise Exception(f"Failed to initialize LLM/embeddings: {str(e)}")
    
    def build_index(self, documents: List[str]) -> float:
        """
        Build vector index from documents
        
        Args:
            documents: List of document texts
            
        Returns:
            float: Time taken to build index in seconds
        """
        start_time = time.time()
        
        try:
            # Global settings are already initialized in __init__
            
            # Create temporary directory for documents
            temp_dir = "temp_docs"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Save documents to temporary files
            doc_files = []
            for i, doc_text in enumerate(documents):
                doc_path = os.path.join(temp_dir, f"doc_{i}.txt")
                with open(doc_path, 'w', encoding='utf-8') as f:
                    f.write(doc_text)
                doc_files.append(doc_path)
            
            # Load documents
            reader = SimpleDirectoryReader(input_files=doc_files)
            docs = reader.load_data()
            
            # Clean up temporary files
            for doc_path in doc_files:
                os.remove(doc_path)
            os.rmdir(temp_dir)
            
            # Create node parser with 512-token chunks
            node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
            
            # Build vector index with explicit models
            self.index = VectorStoreIndex.from_documents(
                documents=docs,
                transformations=[node_parser],
                llm=Settings.llm,
                embed_model=Settings.embed_model
            )
            
            # Query engine will be created on-demand to avoid issues
            
            build_time = time.time() - start_time
            self.build_time = build_time
            return build_time
            
        except Exception as e:
            raise Exception(f"Failed to build vector index: {str(e)}")
    
    def query(self, query_text: str) -> QueryResult:
        """
        Query the vector index
        
        Args:
            query_text: Query string
            
        Returns:
            QueryResult: Query result with response and metadata
        """
        if self.index is None:
            raise Exception("Index not built. Call build_index() first.")
        
        start_time = time.time()
        
        try:
            # Use index query directly with simple similarity search
            query_engine = self.index.as_query_engine(similarity_top_k=3)
            response = query_engine.query(query_text)
            
            # Extract source nodes
            source_nodes = getattr(response, 'source_nodes', [])
            
            execution_time = time.time() - start_time
            
            return QueryResult(
                response=str(response),
                source_nodes=source_nodes,
                execution_time=execution_time
            )
            
        except Exception as e:
            raise Exception(f"Query failed: {str(e)}")
    
    def get_build_time(self) -> float:
        """Get the time taken to build the index"""
        return self.build_time
    
    def is_initialized(self) -> bool:
        """Check if the engine is ready for queries"""
        return self.index is not None