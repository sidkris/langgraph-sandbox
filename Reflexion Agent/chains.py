import datetime
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

load_dotenv()

# Initialize the language model
llm = ChatOpenAI(model="gpt-4-turbo")

# Define the system's reflection prompt template
reflection_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an expert researcher and can self-reflect on your answers to improve their quality.

            Current Time: {time}

            1. Answer the following question in ~250 words.
            2. Reflect on your answer and critique it. Be harsh to identify improvements.
            3. Suggest search queries to refine and improve your answer.
            """
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
).partial(time=lambda: datetime.datetime.now().isoformat())

# Revise instructions for the second phase
revise_instructions = """ 
Revise your previous answer using the new information.

-- You should use the previous critique to add important information to your answer.
-- You must include numerical citations in your revised answer to ensure it can be verified.
-- Add a reference section to the bottom of your answer (which does not count towards the word limit) in the form of:
   - [1] https://www.example.com
   - [2] https://www.example.com
-- You should remove superfluous information to ensure it does not exceed the 250-word limit.
"""

# Step 1: Get the initial response and reflection from the agent
human_message = HumanMessage(content="What is computer vision in self-driving cars? Which are the key startups in this space?")
chain = reflection_prompt_template | llm
initial_response = chain.invoke(input={"messages": [human_message]})

# Step 2: Now, add revise instructions and request a revised response
revise_prompt = f"""
Here is the critique you provided earlier: 
{initial_response.content}

{revise_instructions}
"""

revise_message = HumanMessage(content = revise_prompt)
revised_response = llm(revise_message.content)

# Print both the initial and revised responses
print("Initial Response with Reflection and Critique:")
print(initial_response.content)

print("\nRevised Response:")
print(revised_response.content)
