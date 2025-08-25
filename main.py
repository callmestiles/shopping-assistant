import asyncio
import logging
import os
from typing import List, Dict, Any, Annotated
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

from data_extraction.shopify_client import ShopifyClient
from tools.shopify_search_tool import create_shopify_search_tool

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define the state for our agent
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

class ShopifyShoppingAssistant:
    """
    Main application class that integrates the Shopify search tool
    with a LangGraph agent to create a shopping assistant.
    """
    
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.graph = None
        self.tools = []
    
    async def initialize(self):
        """Initialize the shopping assistant with product data and tools"""
        logger.info("Initializing Shopify Shopping Assistant...")
        
        # Step 1: Fetch products from Shopify
        logger.info("Fetching products from Shopify...")
        try:
            async with ShopifyClient() as client:
                products = await client.fetch_all_products()
            
            if not products:
                raise RuntimeError("No products fetched from Shopify")
            
            logger.info(f"Successfully fetched {len(products)} products")
        
        except Exception as e:
            logger.error(f"Failed to fetch products: {e}")
            raise
        
        # Step 2: Create search tool
        logger.info("Creating search tool...")
        try:
            search_tool = await create_shopify_search_tool(products)
            self.tools = [search_tool]
            logger.info("Search tool created successfully")
        
        except Exception as e:
            logger.error(f"Failed to create search tool: {e}")
            raise
        
        # Step 3: Initialize LLM and LangGraph
        logger.info("Initializing LangGraph agent...")
        try:
            llm = ChatOpenAI(
                api_key=self.openai_api_key,
                model="gpt-3.5-turbo",
                temperature=0.1  # Low temperature for consistent responses
            )
            
            # Bind tools to the LLM
            llm_with_tools = llm.bind_tools(self.tools)
            
            # Create the graph
            self.graph = self._create_graph(llm_with_tools)
            
            logger.info("Shopping assistant initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise
    
    def _create_graph(self, llm_with_tools):
        """Create the LangGraph workflow"""
        
        def should_continue(state: AgentState) -> str:
            """Determine whether to continue or end the conversation"""
            messages = state['messages']
            last_message = messages[-1]
            
            # If the LLM makes a tool call, we route to the "tools" node
            if last_message.tool_calls:
                return "tools"
            # Otherwise, we stop (reply to the user)
            return END
        
        def call_model(state: AgentState) -> Dict[str, List[BaseMessage]]:
            """Call the LLM with the current state"""
            messages = state['messages']
            
            # Add system message if this is the first call
            if not any(isinstance(msg, AIMessage) for msg in messages):
                system_message = self._get_system_message()
                messages = [HumanMessage(content=system_message)] + messages
            
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}
        
        # Create the state graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(self.tools))
        
        # Set the entrypoint
        workflow.add_edge(START, "agent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            should_continue,
        )
        
        # Add edge from tools back to agent
        workflow.add_edge("tools", "agent")
        
        # Compile the graph
        return workflow.compile()
    
    def _get_system_message(self) -> str:
        """Get the system message for the shopping assistant"""
        return """You are a helpful AI shopping assistant for a Shopify store. Your role is to help customers find products they're looking for.

        Key behaviors:
        - Always be friendly, helpful, and conversational
        - Use the shopify_product_search tool when customers ask about products
        - Help customers refine their searches if needed
        - Provide detailed product information when found
        - Suggest alternatives if exact matches aren't available
        - Ask clarifying questions if the customer's request is too vague

        Guidelines for using the search tool:
        - Use specific filters (product_type, color, size, price_max, brand) when the customer mentions them
        - Start with broader searches and narrow down if too many results
        - If no results, suggest similar or alternative products
        - Always format product information clearly

        Example interactions:
        Customer: "I'm looking for a red dress"
        You: I'll help you find a red dress! Let me search our collection for you.

        Customer: "Show me shoes under $100"
        You: I'll search for shoes under $100 for you.

        Remember: You're here to help customers find products they'll love!"""
    
    async def chat(self, message: str) -> str:
        """
        Process a user message and return the assistant's response
        
        Args:
            message: User's input message
            
        Returns:
            Assistant's response
        """
        if not self.graph:
            return "Error: Assistant not initialized. Please contact support."
        
        try:
            # Create initial state with the user message
            initial_state = {
                "messages": [HumanMessage(content=message)]
            }
            
            # Process the message through the graph
            result = await self.graph.ainvoke(initial_state)
            
            # Get the last AI message
            last_message = result["messages"][-1]
            if isinstance(last_message, AIMessage):
                return last_message.content
            else:
                return "I apologize, but I didn't generate a proper response. Could you try again?"
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return "I apologize, but I encountered an error processing your request. Could you please try again?"
    
    async def chat_with_history(self, message: str, chat_history: List[BaseMessage] = None) -> tuple[str, List[BaseMessage]]:
        """
        Process a user message with conversation history and return the response plus updated history
        
        Args:
            message: User's input message
            chat_history: Previous conversation messages
            
        Returns:
            Tuple of (assistant's response, updated chat history)
        """
        if not self.graph:
            return "Error: Assistant not initialized. Please contact support.", chat_history or []
        
        try:
            # Prepare messages with history
            messages = chat_history or []
            messages.append(HumanMessage(content=message))
            
            # Create initial state
            initial_state = {"messages": messages}
            
            # Process through the graph
            result = await self.graph.ainvoke(initial_state)
            
            # Get the response and updated history
            updated_messages = result["messages"]
            last_message = updated_messages[-1]
            
            if isinstance(last_message, AIMessage):
                return last_message.content, updated_messages
            else:
                return "I apologize, but I didn't generate a proper response. Could you try again?", updated_messages
            
        except Exception as e:
            logger.error(f"Error processing message with history: {e}")
            return "I apologize, but I encountered an error processing your request. Could you please try again?", chat_history or []

# Interactive testing function
async def interactive_chat():
    """Run an interactive chat session for testing"""
    
    # Check for API key
    openai_key = os.getenv("OPENAI_KEY")
    if not openai_key:
        print("Error: OPENAI_KEY environment variable not set")
        return
    
    print("Initializing Shopping Assistant...")
    print("=" * 50)
    
    # Initialize assistant
    assistant = ShopifyShoppingAssistant(openai_key)
    
    try:
        await assistant.initialize()
    except Exception as e:
        print(f"Failed to initialize assistant: {e}")
        return
    
    print("\nShopping Assistant Ready!")
    print("=" * 50)
    print("Ask me about products! Type 'quit' to exit, 'reset' to clear conversation.")
    print()
    
    # Maintain conversation history
    chat_history = []
    
    # Interactive loop
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye! Happy shopping!")
                break
            
            if user_input.lower() == 'reset':
                chat_history = []
                print("Conversation reset!")
                continue
            
            if not user_input:
                continue
            
            # Get response with history
            print("Assistant: ", end="", flush=True)
            response, chat_history = await assistant.chat_with_history(user_input, chat_history)
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

# Automated testing function
async def run_test_scenarios():
    """Run automated test scenarios"""
    
    openai_key = os.getenv("OPENAI_KEY")
    if not openai_key:
        print("Error: OPENAI_KEY environment variable not set")
        return
    
    print("Running Test Scenarios...")
    print("=" * 50)
    
    # Initialize assistant
    assistant = ShopifyShoppingAssistant(openai_key)
    await assistant.initialize()
    
    # Test scenarios
    test_cases = [
        {
            "name": "Basic Product Search",
            "messages": [
                "Hi there!",
                "I'm looking for shoes",
                "Do you have any Nike shoes?",
                "What about running shoes under $150?"
            ]
        },
        {
            "name": "Specific Product Request",
            "messages": [
                "I need a red dress for a party",
                "Size medium please", 
                "Show me the details of the first one"
            ]
        },
        {
            "name": "No Results Handling",
            "messages": [
                "Do you have any purple elephant costumes?",
                "Okay, how about regular costumes?",
                "What about party supplies?"
            ]
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTest Case: {test_case['name']}")
        print("-" * 40)
        
        # Use conversation history for each test case
        chat_history = []
        
        for message in test_case['messages']:
            print(f"User: {message}")
            response, chat_history = await assistant.chat_with_history(message, chat_history)
            print(f"Assistant: {response[:200]}...")  # Truncate for readability
            print()
            
            # Small delay between messages
            await asyncio.sleep(1)
        
        print(f"✅ Test case '{test_case['name']}' completed")

async def main():
    """Main function with menu options"""
    
    print("Shopify Shopping Assistant (LangGraph Version)")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Interactive chat")
        print("2. Run test scenarios")
        print("3. Exit")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == "1":
            await interactive_chat()
        elif choice == "2":
            await run_test_scenarios()
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    asyncio.run(main())