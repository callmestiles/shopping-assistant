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
        if score > 0.7:
            return "high"
        elif score > 0.5:
            return "medium"
        elif score > 0.3:
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
    
    def search(self, query: str, max_results: int = 10, min_score: float = 0.5, 
           requirements: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        if not self._is_initialized:
            logger.error("Search service not initialized")
            return []
        
        if not query.strip():
            logger.warning("Empty search query")
            return []
        
        try:
            # Get base similarity results
            query_embedding = self.vectorizer.vectorize_query(query)
            raw_results = self.vector_store.search(query_embedding, max_results * 2)  # Get more for filtering
            
            # Log the similarity scores for debugging
            logger.info(f"Raw similarity scores for query '{query}': {[score for _, score in raw_results[:5]]}")
            
            search_results = []
            for product_embedding, score in raw_results:
                if score >= min_score:
                    search_result = SearchResult(product_embedding, score)
                    search_results.append(search_result)
                else:
                    logger.debug(f"Filtered out result with score {score:.3f} (below threshold {min_score})")
            
            # Log how many results passed the threshold
            logger.info(f"Results above threshold {min_score}: {len(search_results)} out of {len(raw_results)}")
            
            # Apply requirement-based filtering and scoring
            if requirements:
                search_results = self._filter_and_rerank_by_requirements(search_results, requirements)
            
            # Limit to requested number
            search_results = search_results[:max_results]
            
            # Final logging
            if search_results:
                logger.info(f"Found {len(search_results)} products for query: '{query}' with scores: {[r.score for r in search_results]}")
            else:
                logger.info(f"No products found for query: '{query}' above similarity threshold {min_score}")
            
            return search_results
            
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return []

    def _filter_and_rerank_by_requirements(self, results: List[SearchResult], 
                                        requirements: Dict[str, Any]) -> List[SearchResult]:
        filtered_results = []
        
        for result in results:
            product = result.product
            match_score = result.score
            penalty = 0.0
            bonus = 0.0
            
            # Product type matching
            if requirements.get('product_type'):
                required_type = requirements['product_type'].lower()
                product_type = (product.product_type or '').lower()
                
                # Direct match
                if required_type == product_type:
                    bonus += 0.1
                # Partial match in title or description
                elif required_type in product.title.lower() or required_type in product.description.lower():
                    bonus += 0.05
                # Type mismatch penalty
                elif product_type and required_type not in product_type:
                    penalty += 0.15
            
            # Color matching
            if requirements.get('color'):
                required_color = requirements['color'].lower()
                product_text = f"{product.title} {product.description} {' '.join(product.tags)}".lower()
                
                # Check variants for color
                has_color = any(
                    any(opt.get('value', '').lower() == required_color 
                        for opt in variant.get('options', []) 
                        if opt.get('name', '').lower() in ['color', 'colour'])
                    for variant in product.variants
                )
                
                if has_color or required_color in product_text:
                    bonus += 0.08
                else:
                    penalty += 0.1
            
            # Brand matching
            if requirements.get('brand'):
                required_brand = requirements['brand'].lower()
                product_brand = (product.vendor or '').lower()
                
                if required_brand == product_brand:
                    bonus += 0.1
                elif required_brand in product_brand or product_brand in required_brand:
                    bonus += 0.05
                else:
                    penalty += 0.15
            
            # Size matching
            if requirements.get('size'):
                required_size = requirements['size'].lower()
                
                # Check variants for size
                has_size = any(
                    any(opt.get('value', '').lower() == required_size 
                        for opt in variant.get('options', []) 
                        if opt.get('name', '').lower() == 'size')
                    for variant in product.variants
                )
                
                if has_size:
                    bonus += 0.05
                # Don't penalize if no size info available
            
            # Price range matching
            if requirements.get('price_max'):
                max_price = requirements['price_max']
                product_min_price = self._extract_min_price(product)
                
                if product_min_price and product_min_price <= max_price:
                    bonus += 0.03
                elif product_min_price and product_min_price > max_price:
                    penalty += 0.08
            
            # Apply scoring adjustments (but keep them smaller to not override similarity)
            adjusted_score = max(0.0, match_score + bonus - penalty)
            
            # Only include if still above a reasonable threshold after adjustments
            if adjusted_score >= 0.25:
                result.score = adjusted_score
                result.confidence = result._calculate_confidence(adjusted_score)
                filtered_results.append(result)
            else:
                logger.debug(f"Filtered out result after requirements: {product.title} (score: {adjusted_score:.3f})")
        
        # Sort by adjusted score
        filtered_results.sort(key=lambda x: x.score, reverse=True)
        return filtered_results

    def _extract_min_price(self, product: ProductEmbedding) -> Optional[float]:
        try:
            price_range = product.price_range
            min_price_info = price_range.get('minVariantPrice', {})
            amount = min_price_info.get('amount')
            return float(amount) if amount else None
        except (ValueError, TypeError):
            return None
    
    def get_product_by_id(self, product_id: str) -> Optional[SearchResult]:
        if not self._is_initialized:
            return None
            
        product_embedding = self.vector_store.get_product_by_id(product_id)
        if product_embedding:
            return SearchResult(product_embedding, 1.0)  # Perfect match score
        return None
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
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