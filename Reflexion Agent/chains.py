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

# Human input
human_message = HumanMessage(content = "What is computer vision in self-driving cars? Which are the key startups in this space?")

# Run the chain with self-reflection
chain = reflection_prompt_template | llm
response = chain.invoke(input={"messages": [human_message]})

# Print the result
print(response.content)
