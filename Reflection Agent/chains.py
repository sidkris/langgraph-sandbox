from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 
from langchain_openai import ChatOpenAI 


reflection_prompt = ChatPromptTemplate.from_messages(
    [
       ("system", "You are an expert at writing professional summaries for profiles on Linkedin. Generate critique and recommendation for user profile summaries. Provide detailed recommendations suggestions on length of the summary, style, etc."),
       MessagesPlaceholder(variable_name = "messages"),
    ]
)


generation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
        "You are an expert at writing professional summaries for profiles on Linkedin."
        "Generate the best Linkedin summary possible based on the user's request"
        "If the user provides critique, respond with a revised version of your previous attempt(s)."),
        MessagesPlaceholder(variable_name = "messages"),
    ]
)


llm = ChatOpenAI()

generate_chain = generation_prompt | llm 
reflect_chain = reflection_prompt | llm 
