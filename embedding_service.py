import os
import openai
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class EmbeddingService:    
    def __init__(self, use_openai: bool = True):
        self.use_openai = use_openai
        
        if use_openai:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required")
            openai.api_key = self.openai_api_key
            self.model_name = "text-embedding-3-small"
            self.embedding_dim = 1536
        else:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dim = 384
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        if self.use_openai:
            return self._create_openai_embeddings(texts)
        else:
            return self._create_local_embeddings(texts)
    
    def _create_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            logger.info(f"Creating embeddings for {len(texts)} text chunks")
            
            response = openai.embeddings.create(
                input=texts,
                model=self.model_name
            )
            
            embeddings = [item.embedding for item in response.data]
            logger.info(f"Successfully created {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error creating OpenAI embeddings: {str(e)}")
            raise ValueError(f"Failed to create embeddings: {str(e)}")
    
    def _create_local_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            logger.info(f"Creating local embeddings for {len(texts)} text chunks")
            embeddings = self.model.encode(texts)
            logger.info(f"Successfully created {len(embeddings)} local embeddings")
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error creating local embeddings: {str(e)}")
            raise ValueError(f"Failed to create embeddings: {str(e)}")
    
    def create_single_embedding(self, text: str) -> List[float]:
        embeddings = self.create_embeddings([text])
        return embeddings[0]
    
    def get_embedding_dimension(self) -> int:
        return self.embedding_dim 