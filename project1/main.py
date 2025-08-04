# This project is a demo for Rasheed AI, showcasing how to use the OpenAI API.
# It includes a simple script to interact with the API and fetch responses.

from langchain_core.messages import HumanMessage # High-level framework for building AI applications
from langchain_openai import ChatOpenAI # Allow us to use OpenAI's chat models
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent # Complex framwork for building AI agents
from dotenv import load_dotenv # Load environment variables from .env file

# We are building a simple AI agent. The difference between an AI agent and a chatbot
# is that an AI agent have access to tools.

load_dotenv()  # Load environment variables from .env file

@tool
def greet(name: str) -> str:
    """Greet a person with their name."""
    print("Tool: greet")  # Log the tool usage
    return f"Hello, {name}! How can I assist you today?"

def main():
    model = ChatOpenAI(temperature=0.0)  # Initialize the OpenAI chat model with a randomness factor of 0.0

    tools = [greet] # List of tools the agent can use
    agent_executor = create_react_agent(model, tools) # Create a React agent with the model and tools

    print("Welcome to Rasheed AI demo! Type 'exit' to quit.")
    print("You can ask me anything, and I will try to help you.")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == 'exit':
            print("Exiting the demo. Goodbye!")
            break

        print("Rasheed AI: ", end="")
        
        # Stream the response from the agent executor
        # Note: chunka are parts of the response that are sent as they are generated
        for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content = user_input)]}
        ):
            if "agent" in chunk and "messages" in chunk["agent"]:
                for message in chunk["agent"]["messages"]:
                    print(message.content, end="")
        
        print()  # New line after the response

if __name__ == "__main__":
    main()  # Run the main function when the script is executed