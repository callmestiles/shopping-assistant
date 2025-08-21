import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ProductEmbedding:
    product_id: str
    title: str
    description: str
    product_type: str
    vendor: str
    tags: List[str]
    variants: List[Dict[str, Any]]
    price_range: Dict[str, Any]
    combined_text: str
    embedding: np.ndarray

class ProductVectorizer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        logger.info(f"Initialized SentenceTransformer model: {model_name}")
    
    def _create_product_text(self, product: Dict[str, Any]) -> str:
        title = product.get("title", "")
        description = product.get("description", "")
        product_type = product.get("product_type", "")
        vendor = product.get("vendor", "")
        tags = product.get("tags", [])
        
        variants_info = []
        variants = product.get("variants", {}).get("edges", [])
        for variant_edge in variants:
            variant = variant_edge["node"]
            variant_title = variant.get("title", "")
            
            options = variant.get("selectedOptions", [])
            option_values = [opt["value"] for opt in options if opt["value"]]
            
            if variant_title and variant_title != "Default Title":
                variants_info.append(variant_title)
            variants_info.extend(option_values)
        
        text_parts = []
        
        if title:
            text_parts.append(f"Product: {title}")
        
        if description:
            clean_desc = description.replace('<', ' <').replace('>', ' >')
            clean_desc = ' '.join(clean_desc.split())[:500]
            text_parts.append(f"Description: {clean_desc}")
            
        if product_type:
            text_parts.append(f"Type: {product_type}")
            
        if vendor:
            text_parts.append(f"Brand: {vendor}")
            
        if variants_info:
            unique_variants = list(set(variants_info))
            text_parts.append(f"Available in: {', '.join(unique_variants)}")
        
        if tags:
            text_parts.append(f"Tags: {', '.join(tags)}")
            
        combined_text = ' | '.join(text_parts)
        return combined_text
    
    def vectorize_products(self, products: List[Dict[str, Any]]) -> List[ProductEmbedding]:
        logger.info(f"Starting vectorization for {len(products)} products.")
        
        product_embeddings = []
        
        texts = []
        product_data = []
        
        for product in products:
            if not product.get("title"):
                logger.warning(f"Skipping product with missing title: {product.get('id', 'unknown')}")
                continue
            
            combined_text = self._create_product_text(product)
            texts.append(combined_text)
            product_data.append(product)
        
        logger.info("Generating embeddings...")
        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=True)
        
        for i, (product, embedding) in enumerate(zip(product_data, embeddings)):
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
        logger.info(f"Vectorizing query: {query}")
        query_embedding = self.model.encode([query])[0]
        logger.info("Query vectorization complete.")
        return query_embedding
        
