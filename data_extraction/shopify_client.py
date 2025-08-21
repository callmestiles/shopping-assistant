import asyncio
from typing import List, Dict, Any, Optional
import aiohttp
from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport
from config.settings import settings, get_shopify_graphql_url, get_shopify_headers
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ShopifyClient:
    def __init__(self):
        self.url = get_shopify_graphql_url()
        self.headers = get_shopify_headers()
        self.client: Optional[Client] = None
        
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.close()

    async def connect(self):
        try:
            transport = AIOHTTPTransport(url=self.url, headers=self.headers)
            if not self.client:
                self.client = Client(transport=transport)
            logger.info("Connected to Shopify GraphQL API🚀")
        except Exception as e:
            logger.error(f"Error connecting to Shopify GraphQL API: {e}")
            raise
        
        
    async def close(self):
        self.client = None
        
    async def fetch_products(self, cursor: Optional[str] = None, limit:int = 50) -> Dict[str, Any]:
        query = gql("""
            query GetProducts($first: Int!, $after: String) {
                products(first: $first, after: $after) {
                    pageInfo {
                        hasNextPage
                        hasPreviousPage
                        startCursor
                        endCursor
                    }
                    edges {
                        cursor
                        node {
                            id
                            title
                            handle
                            description
                            descriptionHtml
                            productType
                            vendor
                            tags
                            status
                            createdAt
                            updatedAt
                            priceRangeV2 {
                                minVariantPrice {
                                    amount
                                    currencyCode
                                }
                                maxVariantPrice {
                                    amount
                                    currencyCode
                                }
                            }
                            variants(first: 100) {
                                edges {
                                    node {
                                        id
                                        title
                                        price
                                        compareAtPrice
                                        sku
                                        inventoryQuantity
                                        availableForSale
                                        selectedOptions {
                                            name
                                            value
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        """)
       
        variables = {
            "first": limit,
        }
        if cursor:
            variables["after"] = cursor
            
        try:
            if not self.client:
                await self.connect()
            result = await self.client.execute_async(query, variable_values=variables)                
            logger.info(f"Fetched {len(result['products']['edges'])} products")
            return result
        except Exception as e:
            logger.error(f"Error fetching products: {e}")
            raise
    
    async def fetch_all_products(self) -> List[Dict[str, Any]]:
        all_products = []
        cursor = None
        page = 1
        
        logger.info("Fetching all products from Shopify...")
        
        try:
            while True:
                logger.info(f"Fetching page {page}...")
                result = await self.fetch_products(cursor=cursor, limit=50)
                
                products_data = result["products"]
                products = [edge["node"] for edge in products_data["edges"]]
                all_products.extend(products)
                
                page_info = products_data["pageInfo"]
                if not page_info["hasNextPage"]:
                    break
                
                cursor = page_info["endCursor"]
                page += 1
                
                await asyncio.sleep(0.5)
        except Exception as e:
            logger.error(f"Error fetching products: {e}")
            raise

        logger.info(f"Fetched {len(all_products)} products from Shopify.")
        return all_products
    
    async def get_product_by_id(self, product_id: str) -> Optional[Dict[str, Any]]:
        query = gql("""
            query GetProductById($id: ID!) {
                product(id: $id) {
                    id
                    title
                    handle
                    description
                    descriptionHtml
                    productType
                    vendor
                    tags
                    status
                    createdAt
                    updatedAt
                    priceRangeV2 {
                        minVariantPrice {
                            amount
                            currencyCode
                        }
                        maxVariantPrice {
                            amount
                            currencyCode
                        }
                    }
                    variants(first: 100) {
                        edges {
                            node {
                                id
                                title
                                price
                                compareAtPrice
                                sku
                                inventoryQuantity
                                availableForSale
                                selectedOptions {
                                    name
                                    value
                                }
                            }
                        }
                    }
                }
            }
        """)
        
        variables = {"id": product_id}
        try:
            if not self.client:
                await self.connect()
            result = await self.client.execute_async(query, variable_values=variables)
            product = result.get("product")
            if product:
                logger.info(f"Fetched product with ID {product_id}")
            else:
                logger.warning(f"No product found with ID {product_id}")
            return product
        except Exception as e:
            logger.error(f"Error fetching product by ID {product_id}: {e}")
            return None
    
    async def search_products(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        search_query = gql("""
            query SearchProducts($query: String!, $first: Int!) {
                products(query: $query, first: $first) {
                    edges {
                        node {
                            id
                            title
                            handle
                            description
                            descriptionHtml
                            productType
                            vendor
                            tags
                            status
                            createdAt
                            updatedAt
                            priceRangeV2 {
                                minVariantPrice {
                                    amount
                                    currencyCode
                                }
                                maxVariantPrice {
                                    amount
                                    currencyCode
                                }
                            }
                        }
                    }
                }
            }
        """)
        
        variables = {"query": query, "first": limit}
        try:
            if not self.client:
                await self.connect()
            result = await self.client.execute_async(search_query, variable_values=variables)
            products = [edge["node"] for edge in result["products"]["edges"]]
            logger.info(f"Found {len(products)} products matching query '{query}'")
            return products
        except Exception as e:
            logger.error(f"Error searching products: {e}")
            return []

async def main():
        async with ShopifyClient() as client:
            products = await client.fetch_all_products()
        print(f"Total products fetched: {len(products)}")
        if products:
            print("First product sample:")
            print(products[0])  # print first product for verification

if __name__ == "__main__":
    asyncio.run(main())