from typing import List, Sequence
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph 
from chains import generate_chain, reflect_chain

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

REFLECT = "reflect"
GENERATE = "generate"

def generation_node(state : Sequence[BaseMessage]):
    return generate_chain.invoke({"messages" : state})


def reflection_node(messages : Sequence[BaseMessage]) -> List[BaseMessage]:
    res = reflect_chain.invoke({"messages" : messages})
    return [HumanMessage(content = res.content)]


builder = MessageGraph()
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)

def should_continue(state : List[BaseMessage]):
    if len(state) > 6 :
        return END
    return REFLECT


builder.add_conditional_edges(GENERATE, should_continue)
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())
graph.get_graph().print_ascii()


if __name__ == "__main__":

    inputs = HumanMessage(content = """

                Make this summary better :
                         
                        Vice President, IPV Projects and Data Science at Nomura, with international experience, including 
                        time on Wall Street. Creator of the ‘megaprofiler’ Python library, which helps data scientists 
                        and engineers thoroughly understand datasets before analysis or modelling. Specializes in machine 
                        learning, generative AI, quantitative finance, and data-driven solutions, with a proven track record
                        of leading global projects and driving innovation in financial product valuations.

            """)
    
    response = graph.invoke(inputs)