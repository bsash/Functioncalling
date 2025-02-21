import os
from typing import Annotated 
from dotenv import load_dotenv
from haystack.tools import tool, Tool
from haystack.components.tools import ToolInvoker
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret
from haystack.dataclasses import ChatMessage
from haystack.components.generators.utils import print_streaming_chunk
from compound_interest import calculate
from tiktoken import encoding_for_model

# Load environment variables
load_dotenv()

# Replace the @tool decorator and function with a Tool class instance
calculate_investment = Tool(
    name="calculate_investment",
    description="Calculate compound interest for an investment",
    parameters={
        "type": "object",
        "properties": {
            "principal": {
                "type": "number",
                "description": "initial investment amount"
            },
            "rate": {
                "type": "number",
                "description": "Annual interest rate as percentage"
            },
            "time": {
                "type": "number",
                "description": "Time period in years"
            },
            "monthly_expense": {
                "type": "number",
                "description": "Monthly withdrawal/expense amount"
            }
        },
        "required": ["principal", "rate", "time", "monthly_expense"],
        "additionalProperties": False
    },
    function=lambda **kwargs: calculate(
        principal=kwargs.get('principal', 1000),
        rate=kwargs.get('rate', 5),
        time=kwargs.get('time', 1),
        monthly_expense=kwargs.get('monthly_expense', 0)
    )
)

# Initialize the chat generator with OpenAI
chat_generator = OpenAIChatGenerator(
    api_key=Secret.from_env_var("OPENAI_API_KEY"),
    model="gpt-4o-mini",
    tools=[calculate_investment],
    tools_strict=True
)
tool_invoker = ToolInvoker(tools=[calculate_investment])

def count_tokens(messages):
    """Count tokens in messages"""
    encoding = encoding_for_model("gpt-4o-mini")
    num_tokens = 0
    for message in messages:
        # Count tokens in the content
        num_tokens += len(encoding.encode(message.text or ""))
        # Add overhead for each message (4 tokens for metadata)
        num_tokens += 4
    return num_tokens

def chat_loop():
    """Main chat loop for interacting with the user"""
    request_count = 0
    messages = [
        ChatMessage.from_system(
            "You are a helpful financial advisor that helps users calculate compound interest. "
            "Ask users for their investment details one by one and then use the calculate_investment "
            "function to show them the results. Be friendly and explain the results in simple terms."
        )
    ]
    
    print("Financial Advisor: Hello! I'm here to help you calculate compound interest on your investments.")
    print("You can type 'quit' at any time to exit.")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            print("\nFinancial Advisor: Goodbye! Have a great day!")
            break
        
        messages.append(ChatMessage.from_user(user_input))
        response = chat_generator.run(messages=messages)
        replies = response["replies"]
        
        # Handle the first reply
        assistant_reply = replies[0]
        messages.append(assistant_reply)
        print(f"\nFinancial Advisor: {assistant_reply.text}")

        if assistant_reply.tool_calls:
            print("Calling tool...")
            tool_response = tool_invoker.run(messages=[assistant_reply])
            tool_messages = tool_response["tool_messages"]
            messages.extend(tool_messages)
            
            # Get final response after tool execution
            final_response = chat_generator.run(messages=messages)
            final_reply = final_response["replies"][0]
            messages.append(final_reply)
            print(f"\nFinancial Advisor: {final_reply.text}")

        # # Debug information
        # request_count += 1
        # token_count = count_tokens(messages)
        # print(f"\nDEBUG - Request #{request_count}")
        # print(f"Current token count: {token_count}")
        # print(f"Estimated cost: ${(token_count / 1000) * 0.002:.4f} USD")

if __name__ == "__main__":
    chat_loop() 