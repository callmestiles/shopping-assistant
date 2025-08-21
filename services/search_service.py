import logging
from typing import List, Dict, Any, Optional
from data_processing.vectorizer import ProductVectorizer, ProductEmbedding
from data_processing.vector_store import VectorStore

logger = logging.getLogger(__name__)

class SearchResult:
    def __init__(self, product: ProductEmbedding, score: float):
        self.product = product
        self.score = score
        self.confidence = self._calculate_confidence(score)
    
    def _calculate_confidence(self, score: float) -> str:
        if score > 0.8:
            return "high"
        elif score > 0.6:
            return "medium"
        elif score > 0.4:
            return "low"
        else:
            return "very_low"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "product_id": self.product.product_id,
            "title": self.product.title,
            "description": self.product.description[:200] + "..." if len(self.product.description) > 200 else self.product.description,
            "product_type": self.product.product_type,
            "vendor": self.product.vendor,
            "tags": self.product.tags,
            "price_range": self.product.price_range,
            "variants_count": len(self.product.variants),
            "available_variants": [v for v in self.product.variants if v.get("available", False)],
            "similarity_score": round(self.score, 3),
            "confidence": self.confidence,
            "matched_text": self.product.combined_text[:300] + "..." if len(self.product.combined_text) > 300 else self.product.combined_text
        }

class ProductSearchService:
    """
    Main service for product search functionality.
    
    This is the high-level interface that LangGraph agent uses.
    It handles the complete pipeline: query -> vector search -> results.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.vectorizer = ProductVectorizer(model_name)
        self.vector_store = VectorStore()
        self._is_initialized = False
    
    async def initialize_from_products(self, products: List[Dict[str, Any]]):
        """
        Initialize the search service with product data.
        
        This is what you'll call with the products from your Shopify client.
        It handles the entire vectorization and indexing process.
        """
        logger.info("Initializing search service with product data...")
        
        if not products:
            logger.error("No products provided for initialization")
            return False
        
        try:
            #  Vectorize products
            product_embeddings = self.vectorizer.vectorize_products(products)
            
            if not product_embeddings:
                logger.error("No valid product embeddings created")
                return False
            
            # Build vector store
            self.vector_store.add_products(product_embeddings)
            
            self._is_initialized = True
            logger.info(f"Search service initialized with {len(product_embeddings)} products")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize search service: {e}")
            return False
    
    def search(self, query: str, max_results: int = 10, min_score: float = 0.3) -> List[SearchResult]:
        """
        Search for products matching the query.
        
        Args:
            query: User's search query (e.g., "red dress size large")
            max_results: Maximum number of results to return
            min_score: Minimum similarity score threshold
            
        Returns:
            List of SearchResult objects sorted by relevance
        """
        if not self._is_initialized:
            logger.error("Search service not initialized")
            return []
        
        if not query.strip():
            logger.warning("Empty search query")
            return []
        
        try:
            query_embedding = self.vectorizer.vectorize_query(query)
            
            raw_results = self.vector_store.search(query_embedding, max_results)
            
            search_results = []
            for product_embedding, score in raw_results:
                if score >= min_score:
                    search_result = SearchResult(product_embedding, score)
                    search_results.append(search_result)
            
            logger.info(f"Found {len(search_results)} products for query: '{query}'")
            return search_results
            
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return []
    
    def get_product_by_id(self, product_id: str) -> Optional[SearchResult]:
        if not self._is_initialized:
            return None
            
        product_embedding = self.vector_store.get_product_by_id(product_id)
        if product_embedding:
            return SearchResult(product_embedding, 1.0)  # Perfect match score
        return None
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze a query to extract potential product attributes.
        
        This is useful for understanding what the user is looking for
        and identifying missing information. We'll expand this later
        when we add LangGraph.
        """
        query_lower = query.lower()
        
        # Simple keyword extraction - you can make this more sophisticated
        size_keywords = ['small', 'medium', 'large', 'xl', 'xxl', 'xs', 's', 'm', 'l']
        color_keywords = ['red', 'blue', 'green', 'black', 'white', 'yellow', 'pink', 'purple']
        
        detected_attributes = {
            'sizes': [size for size in size_keywords if size in query_lower],
            'colors': [color for color in color_keywords if color in query_lower],
            'has_size_info': any(size in query_lower for size in size_keywords),
            'has_color_info': any(color in query_lower for color in color_keywords),
            'query_length': len(query.split()),
            'is_specific': len(query.split()) > 2  # Simple heuristic
        }
        
        return detected_attributes
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the search service"""
        stats = {
            'is_initialized': self._is_initialized,
            'vectorizer_model': self.vectorizer.model.get_sentence_embedding_dimension(),
        }
        
        if self._is_initialized:
            stats.update(self.vector_store.get_stats())
        
        return stats
    
    def save_index(self, filepath: str):
        """Save the search index to disk"""
        if self._is_initialized:
            self.vector_store.save(filepath)
    
    def load_index(self, filepath: str):
        """Load a pre-built search index from disk"""
        self.vector_store.load(filepath)
        self._is_initialized = True