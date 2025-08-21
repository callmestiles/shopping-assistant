import asyncio
import json
import logging
from data_extraction.shopify_client import ShopifyClient
from services.search_service import ProductSearchService

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_search_system():
    """
    Test the complete search system pipeline:
    1. Fetch products from Shopify
    2. Initialize search service
    3. Perform test searches
    4. Display results
    """
    
    print("🚀 Testing AI Shopping Assistant Search System")
    print("=" * 50)
    
    # Step 1: Fetch products from Shopify
    print("Step 1: Fetching products from Shopify...")
    async with ShopifyClient() as client:
        products = await client.fetch_all_products()
    
    if not products:
        print("❌ No products fetched. Check your Shopify configuration.")
        return
    
    print(f"✅ Fetched {len(products)} products")
    
    # Step 2: Initialize search service
    print("\nStep 2: Initializing search service...")
    search_service = ProductSearchService()
    success = await search_service.initialize_from_products(products)
    
    if not success:
        print("❌ Failed to initialize search service")
        return
    
    print("✅ Search service initialized")
    
    # Print some stats
    stats = search_service.get_stats()
    print(f"📊 Stats: {stats}")
    
    # Step 3: Test searches
    test_queries = [
        "minimal snowboard",
        "shoes size 10", 
        "blue shirt",
        "winter jacket",
        "accessories",
    ]
    
    print(f"\n🔍 Step 3: Testing searches...")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\n🔎 Searching for: '{query}'")
        print("-" * 30)
        
        results = search_service.search(query, max_results=3, min_score=0.2)
        
        if not results:
            print("❌ No results found")
            continue
            
        for i, result in enumerate(results, 1):
            result_dict = result.to_dict()
            print(f"{i}. {result_dict['title']}")
            print(f"   Brand: {result_dict['vendor']}")
            print(f"   Type: {result_dict['product_type']}")
            print(f"   Score: {result_dict['similarity_score']} ({result_dict['confidence']} confidence)")
            print(f"   Variants: {result_dict['variants_count']} available")
            
            # Show price if available
            price_range = result_dict.get('price_range', {})
            min_price = price_range.get('minVariantPrice', {})
            if min_price.get('amount'):
                print(f"   Price: ${min_price['amount']} {min_price.get('currencyCode', '')}")
            
            print(f"   Description: {result_dict['description'][:100]}...")
            print()
    
    # Step 4: Test query analysis
    print(f"\n📊 Step 4: Query Analysis Examples")
    print("=" * 50)
    
    analysis_queries = [
        "red dress size large",
        "shoes",
        "blue medium shirt cotton",
        "gift for mom"
    ]
    
    for query in analysis_queries:
        analysis = search_service.analyze_query(query)
        print(f"Query: '{query}'")
        print(f"  - Detected sizes: {analysis['sizes']}")
        print(f"  - Detected colors: {analysis['colors']}")
        print(f"  - Has size info: {analysis['has_size_info']}")
        print(f"  - Is specific: {analysis['is_specific']}")
        print()
    
    print("✅ Testing complete!")

if __name__ == "__main__":
    asyncio.run(test_search_system())