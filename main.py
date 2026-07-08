from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch

class Source(BaseModel):
    """Schema for the source used by the agent"""
    url: str = Field(description="The url of the source")

class AgentResponse(BaseModel):
    """Schema for the agent response with answers and sourcest"""
    sources: List[Source] = Field(default_factory=list,description="The list of sources used to generate the answer")
    answer: str = Field(description="The agent's answer to the user's question")

llm = ChatOpenAI()
tools = [TavilySearch()]
agent = create_agent(
        model=llm,
        tools=tools,
        response_format=AgentResponse
    )

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": [HumanMessage(content="search for 3 job postings in AI using langchain in DC area.")]})
    print(result)

if __name__ == "__main__":
    main()
