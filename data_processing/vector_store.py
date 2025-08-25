import logging
import pickle
import faiss
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
from data_processing.vectorizer import ProductEmbedding

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, embedding_dim: int = 384):  # all-MiniLM-L6-v2 has 384 dimensions
        self.embedding_dim = embedding_dim
        self.index = None
        self.product_embeddings: List[ProductEmbedding] = []
        self._is_trained = False
        
    def add_products(self, product_embeddings: List[ProductEmbedding]):
        logger.info(f"Adding {len(product_embeddings)} products to vector store...")
        
        if not product_embeddings:
            logger.warning("No product embeddings provided")
            return
        
        self.product_embeddings = product_embeddings
        
        embeddings_array = np.array([pe.embedding for pe in product_embeddings])
        embeddings_array = embeddings_array.astype('float32')  # FAISS requires float32
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        faiss.normalize_L2(embeddings_array)
        
        self.index.add(embeddings_array)
        self._is_trained = True
        
        logger.info(f"Vector store built with {self.index.ntotal} products")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[ProductEmbedding, float]]:
        if not self._is_trained:
            logger.error("Vector store not trained. Add products first.")
            return []
        
        query_array = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_array)  # Normalize for cosine similarity
        
        similarities, indices = self.index.search(query_array, top_k)
        
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(self.product_embeddings):  # Safety check
                product_embedding = self.product_embeddings[idx]
                results.append((product_embedding, float(similarity)))
        
        logger.info(f"Found {len(results)} similar products")
        return results
    
    def get_product_by_id(self, product_id: str) -> Optional[ProductEmbedding]:
        for pe in self.product_embeddings:
            if pe.product_id == product_id:
                return pe
        return None
    
    def save(self, filepath: str):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if self.index:
            faiss.write_index(self.index, str(filepath.with_suffix('.faiss')))
        
        products_data = []
        for pe in self.product_embeddings:
            product_data = {
                'product_id': pe.product_id,
                'title': pe.title,
                'description': pe.description,
                'product_type': pe.product_type,
                'vendor': pe.vendor,
                'tags': pe.tags,
                'variants': pe.variants,
                'price_range': pe.price_range,
                'combined_text': pe.combined_text
            }
            products_data.append(product_data)
        
        with open(filepath.with_suffix('.pkl'), 'wb') as f:
            pickle.dump({
                'products_data': products_data,
                'embedding_dim': self.embedding_dim,
                'is_trained': self._is_trained
            }, f)
        
        logger.info(f"Vector store saved to {filepath}")
    
    def load(self, filepath: str, vectorizer=None):
        filepath = Path(filepath)
        
        faiss_path = filepath.with_suffix('.faiss')
        if faiss_path.exists():
            self.index = faiss.read_index(str(faiss_path))
        
        pkl_path = filepath.with_suffix('.pkl')
        if pkl_path.exists():
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
                
            self.embedding_dim = data['embedding_dim']
            self._is_trained = data['is_trained']
            
            embeddings = self.index.reconstruct_n(0, self.index.ntotal) if self.index else []
            
            self.product_embeddings = []
            for prod_data, embedding in zip(data['products_data'], embeddings):
                pe = ProductEmbedding(
                    product_id=prod_data['product_id'],
                    title=prod_data['title'],
                    description=prod_data['description'],
                    product_type=prod_data['product_type'],
                    vendor=prod_data['vendor'],
                    tags=prod_data['tags'],
                    variants=prod_data['variants'],
                    price_range=prod_data['price_range'],
                    combined_text=prod_data['combined_text'],
                    embedding=embedding
                )
                self.product_embeddings.append(pe)
        
        logger.info(f"Vector store loaded from {filepath}")
    
    def get_stats(self) -> dict:
        return {
            'total_products': len(self.product_embeddings),
            'embedding_dimension': self.embedding_dim,
            'is_trained': self._is_trained,
            'index_size': self.index.ntotal if self.index else 0
        }