import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ProductEmbedding:
    """Container for product data and its embedding"""
    product_id: str
    title: str
    description: str
    product_type: str
    vendor: str
    tags: List[str]
    variants: List[Dict[str, Any]]
    price_range: Dict[str, Any]
    combined_text: str  # The text that was vectorized
    embedding: np.ndarray

class ProductVectorizer:
    """
    Handles converting product data into vector embeddings for similarity search.
    
    This class takes raw product data from Shopify and creates meaningful
    text representations that can be converted into vectors for semantic search.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        logger.info(f"Initialized SentenceTransformer model: {model_name}")
    
    def _create_product_text(self, product: Dict[str, Any]) -> str:
        # Extract basic info
        title = product.get("title", "").strip()
        description = product.get("description", "").strip()
        product_type = product.get("productType", "").strip()
        vendor = product.get("vendor", "").strip()
        tags = product.get("tags", [])
        
        # Clean description: remove HTML and limit length
        if description:
            import re
            # Remove HTML tags
            description = re.sub(r'<[^>]+>', ' ', description)
            # Remove extra whitespace
            description = ' '.join(description.split())
            # Limit length
            description = description[:400]
        
        # Extract and process variants
        variants_info = []
        color_options = set()
        size_options = set()
        material_options = set()
        
        variants = product.get("variants", {}).get("edges", [])
        for variant_edge in variants:
            variant = variant_edge["node"]
            variant_title = variant.get("title", "")
            
            # Process selectedOptions for attributes
            options = variant.get("selectedOptions", [])
            for option in options:
                option_name = option.get("name", "").lower()
                option_value = option.get("value", "").lower()
                
                if option_name in ['color', 'colour']:
                    color_options.add(option_value)
                elif option_name in ['size']:
                    size_options.add(option_value)
                elif option_name in ['material', 'fabric']:
                    material_options.add(option_value)
            
            if variant_title and variant_title != "Default Title":
                variants_info.append(variant_title.lower())
        
        # Build searchable text with strategic keyword placement
        text_parts = []
        
        # Primary product identification (highest weight)
        if title:
            text_parts.append(f"{title}")
        
        if product_type:
            text_parts.append(f"{product_type}")
            # Add synonyms for common product types
            type_synonyms = {
                'shirt': ['top', 'blouse', 'tee'],
                'dress': ['gown', 'frock'],
                'pants': ['trousers', 'jeans'],
                'shoes': ['footwear', 'sneakers'],
            }
            if product_type.lower() in type_synonyms:
                text_parts.extend(type_synonyms[product_type.lower()])
        
        # Attributes (medium-high weight)
        if color_options:
            colors = list(color_options)
            text_parts.append(f"colors: {' '.join(colors)}")
            text_parts.extend(colors)  # Add individual colors
        
        if size_options:
            sizes = list(size_options)
            text_parts.append(f"sizes: {' '.join(sizes)}")
            text_parts.extend(sizes)  # Add individual sizes
        
        if material_options:
            materials = list(material_options)
            text_parts.append(f"materials: {' '.join(materials)}")
        
        # Brand (medium weight)
        if vendor:
            text_parts.append(f"brand: {vendor}")
        
        # Tags (medium weight)
        if tags:
            clean_tags = [tag.lower().replace('-', ' ') for tag in tags]
            text_parts.append(f"tags: {' '.join(clean_tags)}")
        
        # Description (lower weight, for context)
        if description:
            text_parts.append(f"description: {description}")
        
        # Variants (for completeness)
        if variants_info:
            unique_variants = list(set(variants_info))
            text_parts.append(f"available: {' '.join(unique_variants)}")
        
        # Join with separators that help the model understand structure
        combined_text = " | ".join(text_parts)
        
        # Add search-friendly keywords based on product type
        if product_type:
            search_keywords = self._get_search_keywords(product_type.lower())
            if search_keywords:
                combined_text += f" | keywords: {' '.join(search_keywords)}"
        
        return combined_text

    def _get_search_keywords(self, product_type: str) -> List[str]:
        """Get additional search keywords for product types"""
        keyword_map = {
            'shirt': ['clothing', 'apparel', 'wear', 'casual', 'formal'],
            'dress': ['clothing', 'women', 'formal', 'party', 'elegant'],
            'pants': ['clothing', 'bottoms', 'casual', 'formal'],
            'shoes': ['footwear', 'fashion', 'accessories'],
            'bag': ['accessory', 'fashion', 'carry'],
            'jacket': ['outerwear', 'clothing', 'warm'],
        }
        return keyword_map.get(product_type, ['clothing', 'fashion'])
    
    def vectorize_products(self, products: List[Dict[str, Any]]) -> List[ProductEmbedding]:
        logger.info(f"Vectorizing {len(products)} products...")
        logger.info("Starting vectorization for {} products.".format(len(products)))
        
        product_embeddings = []
        
        # Prepare all texts for batch processing (much faster than one by one)
        texts = []
        product_data = []
        
        for product in products:
            # Skip products without essential information
            if not product.get("title"):
                logger.warning(f"Skipping product without title: {product.get('id', 'unknown')}")
                continue
                
            combined_text = self._create_product_text(product)
            texts.append(combined_text)
            product_data.append(product)
        
        # Generate embeddings in batch (efficient)
        logger.info("Generating embeddings...")
        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=True)
        
        # Create ProductEmbedding objects
        for i, (product, embedding) in enumerate(zip(product_data, embeddings)):
            # Extract variant information for easy access
            variants = []
            variant_edges = product.get("variants", {}).get("edges", [])
            for edge in variant_edges:
                variant = edge["node"]
                variants.append({
                    "id": variant.get("id"),
                    "title": variant.get("title"),
                    "price": variant.get("price"),
                    "available": variant.get("availableForSale", False),
                    "options": variant.get("selectedOptions", [])
                })
            
            product_embedding = ProductEmbedding(
                product_id=product["id"],
                title=product.get("title", ""),
                description=product.get("description", ""),
                product_type=product.get("productType", ""),
                vendor=product.get("vendor", ""),
                tags=product.get("tags", []),
                variants=variants,
                price_range=product.get("priceRangeV2", {}),
                combined_text=texts[i],
                embedding=embedding
            )
            
            product_embeddings.append(product_embedding)
        
        logger.info(f"Successfully vectorized {len(product_embeddings)} products")
        return product_embeddings
    
    def vectorize_query(self, query: str) -> np.ndarray:
        """
        Convert a user search query into a vector for similarity matching.
        
        The user's query gets the same vectorization treatment as products
        so we can compare them in vector space.
        """
        return self.model.encode([query])[0]