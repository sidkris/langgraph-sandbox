import datetime
from dotenv import load_dotenv
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from schemas import AnswerQuestion

# Load environment variables (API keys)
load_dotenv()

# Initialize the LLM with the desired model
llm = ChatOpenAI(model="gpt-4-turbo-preview")

# Define the output parser using the PydanticToolsParser for structured output
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

# Define the main prompt template with system instructions
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an expert researcher.

            Current Time: {time}

            1. Provide a detailed ~250 word answer.
            2. Reflect and critique your answer. Be severe to maximize improvement.
            3. Recommend search queries to research information and improve your answer.
            """
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
).partial(time=lambda: datetime.datetime.now().isoformat())

# Define the human message for input
human_message = HumanMessage(content="Write a summary about computer vision in self-driving cars. List startups in this space that have raised significant capital.")

# Step 1: Test the LLM directly with the prompt template
def test_llm_directly():
    prompt = actor_prompt_template.format(messages=[human_message])
    print(f"Formatted Prompt Sent to LLM:\n{prompt}")

    try:
        llm_response = llm(prompt)  # Test direct LLM call
        print(f"LLM Response:\n{llm_response}")
        return llm_response
    except Exception as e:
        print(f"Error in LLM invocation: {e}")
        return None

# Step 2: Try invoking with the parser to test if parsing works
def test_with_parser():
    chain = actor_prompt_template | llm | parser_pydantic

    try:
        response = chain.invoke(input={"messages": [human_message]})
        print(f"Chain Response:\n{response}")
        return response
    except Exception as e:
        print(f"Error in Chain Invocation: {e}")
        return None

if __name__ == "__main__":
    # Step 1: Test the LLM call directly
    llm_response = test_llm_directly()

    if llm_response:
        # Step 2: If LLM response works, test the full chain with the parser
        response = test_with_parser()

        # Print final response
        if response:
            print(f"Final Response: {response}")
        else:
            print("No response received from the chain.")
    else:
        print("LLM did not return a response.")


#=======================================================================================


# import datetime 
# from dotenv import load_dotenv
# from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser, PydanticToolsParser
# from langchain_core.messages import HumanMessage
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_openai import ChatOpenAI

# from schemas import AnswerQuestion

# load_dotenv()

# llm = ChatOpenAI(model = "gpt-4-turbo-preview")
# parser = JsonOutputToolsParser(return_id = True)
# parser_pydantic = PydanticToolsParser(tools = [AnswerQuestion])

# actor_prompt_template = ChatPromptTemplate.from_messages(
#     [
#         (
#         "system", 
#         """
#         You are an expert researcher. 

#         Current Time : {time}

#         1. Provide a detailed ~250 word answer.
#         2. Reflect and critique your answer. Be severe to maximize improvement.
#         3. Recommend search queries to research information and improve your answer.
#         """
#         ),
#         MessagesPlaceholder(variable_name = "messages"),
#     ]
# ).partial(time = lambda : datetime.datetime.now().isoformat())


# first_response_prompt_template = actor_prompt_template.partial(
#     first_instruction = "Provide a detailed ~250 word answer."
# )

# first_response = first_response_prompt_template | llm.bind_tools(tools = [AnswerQuestion], tool_choice = "AnswerQuestion")


# if __name__ == "__main__" :

#     human_message = HumanMessage(content = "Write a summary about computer vision in self driving cars. List startups in this space that have raised significant capital.")

#     chain = (first_response_prompt_template | llm.bind_tools(tools = [AnswerQuestion], tool_choice = "AnswerQuestion") | parser_pydantic)

#     response = chain.invoke(input = {"messages" : [human_message]})

#     print(response)