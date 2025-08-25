from typing import Dict, Any, List, Optional, Type, ClassVar
from langchain.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr
import logging

from services.search_service import ProductSearchService

logger = logging.getLogger(__name__)

class ShopifyProductSearchInput(BaseModel):
    """Input schema for Shopify product search tool"""
    query: str = Field(description="Natural language search query for products")
    max_results: int = Field(default=5, description="Maximum number of results to return")
    min_score: float = Field(default=0.4, description="Minimum similarity score threshold (0.0-1.0)")
    product_type: Optional[str] = Field(default=None, description="Specific product type filter")
    color: Optional[str] = Field(default=None, description="Color filter")
    size: Optional[str] = Field(default=None, description="Size filter")
    price_max: Optional[float] = Field(default=None, description="Maximum price filter")
    brand: Optional[str] = Field(default=None, description="Brand filter")

class ShopifyProductSearchTool(BaseTool):
    """
    LangChain tool for searching Shopify products using semantic vector search.
    
    This tool integrates with LangGraph workflows to provide product search
    capabilities to AI assistants.
    """
    
    name: ClassVar[str] = "shopify_product_search"
    description: ClassVar[str] = """
    Search for products in a Shopify store using natural language queries.
    
    This tool performs semantic search across product titles, descriptions, types,
    and attributes to find matching products. It uses similarity scoring to ensure
    only relevant products are returned.
    
    The tool will return products that are semantically similar to the query.
    If no products meet the similarity threshold, it will return a "no results" message.
    
    Use this tool when users ask about specific products, want to browse categories,
    or need product recommendations. 
    
    Examples of when to use:
    - "Find blue jeans" 
    - "Show me winter jackets under $200"
    - "I need running shoes size 10"
    - "What dresses do you have?"
    
    Important: Only call this tool ONCE per user query. Do not make multiple searches 
    for the same request as this can lead to redundant results.
    """
    
    args_schema: Type[BaseModel] = ShopifyProductSearchInput
    _search_service: ProductSearchService = PrivateAttr()

    def __init__(self, search_service: ProductSearchService):
        super().__init__()
        self._search_service = search_service

    def _run(self, 
             query: str,
             max_results: int = 5,
             min_score: float = 0.5,
             product_type: Optional[str] = None,
             color: Optional[str] = None,
             size: Optional[str] = None,
             price_max: Optional[float] = None,
             brand: Optional[str] = None) -> str:
        """
        Execute the product search and return formatted results.
        
        Returns a formatted string that can be used by the LLM to provide
        natural responses to users.
        """

        if not self._search_service._is_initialized:
            return "Error: Product search service is not initialized. Please contact support."
        
        try:
            # Build requirements dict from filters
            requirements = {}
            if product_type:
                requirements['product_type'] = product_type
            if color:
                requirements['color'] = color
            if size:
                requirements['size'] = size
            if price_max:
                requirements['price_max'] = price_max
            if brand:
                requirements['brand'] = brand
            
            # Log the search parameters for debugging
            logger.info(f"Searching with query: '{query}', min_score: {min_score}, requirements: {requirements}")
            
            # Perform search
            results = self._search_service.search(
                query=query,
                max_results=max_results,
                min_score=min_score,
                requirements=requirements if requirements else None
            )
            
            # Handle no results case
            if not results:
                return self._format_no_results(query, requirements, min_score)
            
            # Format results for LLM consumption
            formatted_response = self._format_results(results, query)
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error in product search: {e}")
            return f"Error occurred during product search: {str(e)}"
    
    def _format_no_results(self, query: str, requirements: Dict[str, Any], min_score: float) -> str:
        """Format a helpful response when no results are found"""
        
        response_parts = [f"I couldn't find any products matching '{query}' in our store."]
        
        # Provide specific feedback based on the query
        suggestions = []
        
        if requirements:
            if requirements.get('brand'):
                suggestions.append(f"Try searching without the specific brand '{requirements['brand']}'")
            if requirements.get('color'):
                suggestions.append(f"Consider other colors similar to '{requirements['color']}'")
            if requirements.get('product_type'):
                suggestions.append(f"Look for broader categories instead of '{requirements['product_type']}'")
        
        # General suggestions based on query analysis
        query_lower = query.lower()
        if any(word in query_lower for word in ['shoes', 'sneakers', 'boots', 'sandals']):
            suggestions.append("Try searching for 'footwear' or browse our general product categories")
        elif any(word in query_lower for word in ['dress', 'shirt', 'pants', 'jacket']):
            suggestions.append("Try searching for 'clothing' or 'apparel'")
        elif any(word in query_lower for word in ['nike', 'adidas', 'puma']):
            suggestions.append("Try searching without specifying a brand")
        
        if not suggestions:
            suggestions = [
                "Try using more general search terms",
                "Browse our available product categories",
                "Check if the product name is spelled correctly"
            ]
        
        if suggestions:
            response_parts.append("\nHere are some suggestions:")
            for i, suggestion in enumerate(suggestions[:3], 1):
                response_parts.append(f"{i}. {suggestion}")
        
        response_parts.append(f"\nNote: I'm looking for products with at least {int(min_score*100)}% similarity to your search.")
        
        return "\n".join(response_parts)
    
    def _format_results(self, results: List, query: str) -> str:
        """Format search results for LLM consumption"""
        
        response_parts = [f"Found {len(results)} products matching '{query}':\n"]
        
        for i, result in enumerate(results, 1):
            result_dict = result.to_dict()
            
            # Basic product info
            title = result_dict['title']
            vendor = result_dict['vendor']
            product_type = result_dict['product_type']
            confidence = result_dict['confidence']
            similarity_score = result_dict['similarity_score']
            
            # Price information
            price_info = ""
            price_range = result_dict.get('price_range', {})
            min_price = price_range.get('minVariantPrice', {})
            if min_price.get('amount'):
                currency = min_price.get('currencyCode', 'USD')
                price_info = f" - ${min_price['amount']} {currency}"
            
            # Availability info
            variants_count = result_dict['variants_count']
            available_variants = len(result_dict.get('available_variants', []))
            
            # Build result string
            result_text = f"{i}. **{title}**{price_info}\n"
            result_text += f"   Brand: {vendor}\n"
            result_text += f"   Type: {product_type}\n"
            result_text += f"   Match Score: {similarity_score} ({confidence})\n"
            result_text += f"   Availability: {available_variants}/{variants_count} variants available\n"
            
            # Add brief description
            description = result_dict['description']
            if description and description.strip():
                # Truncate description for readability
                desc_preview = description[:100] + "..." if len(description) > 100 else description
                result_text += f"   Description: {desc_preview}\n"
            
            # Add variant options if available
            variants = result_dict.get('available_variants', [])
            if variants:
                options = self._extract_variant_options(variants)
                if options:
                    result_text += f"   Available Options: {options}\n"
            
            result_text += "\n"
            response_parts.append(result_text)
        
        return "".join(response_parts)
    
    def _extract_variant_options(self, variants: List[Dict]) -> str:
        """Extract and format variant options"""
        colors = set()
        sizes = set()
        other_options = set()
        
        for variant in variants[:5]:  # Limit to first 5 variants
            options = variant.get('options', [])
            for option in options:
                name = option.get('name', '').lower()
                value = option.get('value', '')
                
                if name in ['color', 'colour']:
                    colors.add(value)
                elif name == 'size':
                    sizes.add(value)
                elif value and value != "Default Title":  # Skip empty or default values
                    other_options.add(f"{name}: {value}")
        
        option_parts = []
        if colors:
            option_parts.append(f"Colors: {', '.join(sorted(colors))}")
        if sizes:
            option_parts.append(f"Sizes: {', '.join(sorted(sizes))}")
        if other_options:
            option_parts.append(f"Other: {', '.join(sorted(other_options))}")
        
        return "; ".join(option_parts)

# Factory function to create the tool
async def create_shopify_search_tool(products_data: List[Dict[str, Any]]) -> ShopifyProductSearchTool:
    """
    Factory function to create and initialize the Shopify search tool.
    
    Args:
        products_data: List of product dictionaries from Shopify API
        
    Returns:
        Initialized ShopifyProductSearchTool ready for use
    """
    
    # Initialize search service
    search_service = ProductSearchService()
    success = await search_service.initialize_from_products(products_data)
    
    if not success:
        raise RuntimeError("Failed to initialize search service with product data")
    
    logger.info(f"Shopify search tool initialized with {len(products_data)} products")
    
    # Create and return the tool
    return ShopifyProductSearchTool(search_service)
