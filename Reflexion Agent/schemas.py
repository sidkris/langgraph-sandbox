from typing import List

from langchain_core.pydantic_v1 import BaseModel, Field 

class Reflection(BaseModel):

    missing : str = Field(description = "critique of what is missing")
    superflous : str = Field(description = "critique of what is superflous")


class AnswerQuestion(BaseModel):

    answer : str = Field(description = "approx. 250 word detailed answer to the question.")
    reflection : Reflection = Field(description = "your reflection on the initial answer.")

    search_queries : List[str] = Field(description = "1-3 search queries for researching improvements to address the critique of your current answer.")


class ReviseAnswer(AnswerQuestion):

    references :  List[str] = Field(description = "Citations motivating your updated answer.")