import os
from dotenv import load_dotenv
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret
from haystack.dataclasses import ChatMessage
from haystack.components.generators.utils import print_streaming_chunk
from compound_interest import calculate
from tiktoken import encoding_for_model

# Load environment variables
load_dotenv()

# Initialize the chat generator with OpenAI
chat_generator = OpenAIChatGenerator(
    api_key=Secret.from_env_var("OPENAI_API_KEY"),
    model="gpt-3.5-turbo",
    streaming_callback=print_streaming_chunk,
    generation_kwargs={
        "functions": [
            {
                "name": "calculate_investment",
                "description": "Calculate compound interest with monthly expenses",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "principal": {
                            "type": "number",
                            "description": "Initial investment amount"
                        },
                        "rate": {
                            "type": "number",
                            "description": "Annual interest rate as a percentage"
                        },
                        "time": {
                            "type": "integer",
                            "description": "Time period in years"
                        },
                        "monthly_expense": {
                            "type": "number",
                            "description": "Monthly withdrawal/expense amount"
                        }
                    },
                    "required": ["principal", "rate", "time", "monthly_expense"]
                }
            }
        ]
    }
)

def calculate_investment(principal: float, rate: float, time: int, monthly_expense: float):
    """Wrapper function to call the calculate function and format the output"""
    results, schedule = calculate(principal, rate, time, monthly_expense)
    return f"{results}\n\n{schedule}"

def count_tokens(messages):
    """Count tokens in messages"""
    encoding = encoding_for_model("gpt-3.5-turbo")
    num_tokens = 0
    for message in messages:
        # Count tokens in the content
        num_tokens += len(encoding.encode(message.content or ""))
        # Add overhead for each message (4 tokens for metadata)
        num_tokens += 4
    return num_tokens

def chat_loop():
    """Main chat loop for interacting with the user"""
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
        # response = chat_generator.run(messages)
        
        # if response.meta.get("function_call"):
        #     function_args = response.meta["function_call"]["arguments"]
        #     result = calculate_investment(**function_args)
            
        #     # Add function response to messages
        #     messages.append(ChatMessage.from_assistant(response.data))
        #     messages.append(ChatMessage.from_function(
        #         result,
        #         name="calculate_investment"
        #     ))
            
        #     # Get AI to explain the results
        #     response = chat_generator.run(messages)
        
        # Debug information instead of actual API call
        request_count += 1
        token_count = count_tokens(messages)
        print(f"\nDEBUG - Request #{request_count}")
        print(f"Current token count: {token_count}")
        print(f"Estimated cost: ${(token_count / 1000) * 0.002:.4f} USD")
        
        # For debugging, simulate a simple response
        print("\nFinancial Advisor: [Simulated response for debugging]")
        
        # Add simulated response to messages for token counting
        messages.append(ChatMessage.from_assistant("[Simulated response for debugging]"))

        # print(f"\nFinancial Advisor: {response.data}")
        # messages.append(ChatMessage.from_assistant(response.data))

if __name__ == "__main__":
    chat_loop() 