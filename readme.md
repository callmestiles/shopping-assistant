# Shopify Shopping Assistant

## Overview

Shopify Shopping Assistant is an AI-powered tool that helps users search and discover products from a Shopify store using natural language queries. It leverages semantic search, vector embeddings, and integrates with LangChain and LangGraph to provide intelligent product recommendations and search capabilities.

## Features

- **Natural Language Product Search:** Search for products using conversational queries.
- **Semantic Vector Search:** Finds products based on similarity in descriptions, attributes, and more.
- **Shopify Integration:** Connects directly to Shopify's GraphQL API for real-time product data.
- **Customizable Filters:** Search by product type, color, size, price, brand, and more.
- **AI Agent Workflow:** Uses LangGraph and LangChain for advanced agent-based workflows.

## Project Structure

- `main.py`: Entry point for the application and agent setup.
- `config/`: Configuration and environment settings for Shopify and OpenAI.
- `data_extraction/`: Shopify API client for fetching product data.
- `data_processing/`: Vectorizer and vector store for semantic search.
- `services/`: Product search service and result formatting.
- `tools/`: LangChain tool for product search integration.

## Setup Instructions

1. **Clone the repository:**

   ```powershell
   git clone <repo-url>
   cd shopping-assistant
   ```

2. **Create and activate a Python virtual environment:**

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**

   ```powershell
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   - Create a `.env` file in the root directory with your Shopify and OpenAI credentials:
     ```env
     SHOPIFY_SHOP_URL=yourshop.myshopify.com
     SHOPIFY_ACCESS_TOKEN=your_access_token
     OPENAI_KEY=your_openai_api_key
     ```

## Usage

Run the main application:

```powershell
python main.py
```

You can interact with the assistant to search for products, get recommendations, and apply filters using natural language.

## Technologies Used

- Python 3.12+
- LangChain, LangGraph
- Shopify GraphQL API
- FAISS (vector search)
- Pydantic, dotenv

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements or bug fixes.
