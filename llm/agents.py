from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
) 
from typing import List, Optional
import logging
from dotenv import load_dotenv
load_dotenv()
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

logging.basicConfig(level=logging.INFO, format='%(asctime)s -%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)   

class MasterAnswer(BaseModel):
    answer: str = Field(..., description="Answer to the query")

class PayAnswer(BaseModel):
    amount: float = Field(..., description="Amount to pay")
    name: str = Field(..., description="Name of the person to pay")

def Master_Agent(query: str):
    logger.info("Processing Query: %s", query)
    
    examples = [
        {"input": "Tell me about Proof of work", "output": '{"answer": "Search-Agent"}'},
        {"input": "Transfer 2 Test money to mintu", "output": '{"answer": "Pay-Agent"}'},
    ]

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", '''You are a Web3 and technology agent which has knowledge of Web3 blockchain and its protocols and other technologies. You have to identify the intent of a user's query from its input. If the user's query is related to blockchain and other tech topics, respond with Search-Agent. If the user's query is related to making a payment to someone, respond with Pay-Agent. Your response must be in valid JSON format with an "answer" field.'''),
            few_shot_prompt,
            ("human", "{input}"),
        ]
    )

    # Create the model and chain
    model = ChatOpenAI(model='gpt-4o-mini', temperature=0.0)  # Corrected model name
    parser = JsonOutputParser(pydantic_object=MasterAnswer)
    
    chain = final_prompt | model | parser
    
    try:
        result = chain.invoke({"input": query})
        logger.info("Query Result: %s", result)
        return result
    except Exception as e:
        logger.error("Error processing query: %s", e)
        raise

# Example usage
if __name__ == "__main__":
    result = Master_Agent("Tell me about blockchain technology")
    print(result)


def Pay_Agent(query:str):

    logger.info("Processing Query: %s", query)
    examples = [
        {"input": "Send 5 eth to mohan", "output": "{\"amount\": 5.0, \"name\": \"mohan\"}"},
        {"input": "Tranfer 2 Test money to mintu", "output": "{\"amount\": 2.0, \"name\": \"mintu\"}"},
    ]

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", '''You are an AI assistant helping users with transferring money or cryptocurrency. You will receive a query from the user, where the intent is to send a specified amount of money or cryptocurrency to a particular person. Your task is to identify and extract the amount and recipient's name from the query.

Respond only in JSON format with the following keys:

"amount": the value of the money or cryptocurrency to be sent, including the currency symbol if specified.
"name": the name of the person to whom the money or cryptocurrency will be sent.

'''),
            few_shot_prompt,
            ("human", "{input}"),
        ]
    )
    from langchain_openai import ChatOpenAI
    model = ChatOpenAI(model='gpt-4o-mini',temperature=0.0)
    parser = JsonOutputParser(pydantic_object=PayAnswer)
    chain = final_prompt | model | parser
    try:
        result = chain.invoke({"input": query})
        logger.info("Query Result: %s", result)
        # print(result.content)
        return result
    except Exception as e:
        logger.error("Error processing query: %s", e)
        raise
Master_Agent("Tell me about zk sync.")
Pay_Agent("Send 8 eth to shweta")