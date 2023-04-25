from langchain.memory import ConversationSummaryMemory
from langchain.llms import OpenAI

from dotenv import load_dotenv
# import os

load_dotenv()


llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")
memory = ConversationSummaryMemory(llm=llm, return_messages=True)
memory.save_context({"input": "Why are load balancers needed?"},
                    {"output": "Past a certain point, web applications outgrow a single server deployment. Companies either want to increase their availability, scalability, or both! To do this, they deploy their application across multiple servers with a load balancer in front to distribute incoming requests. Big companies may need thousands of servers running their web application to handle the load."})
memory.save_context({"input": "Give names of a few common load balancing algorithms."},
                    {"output": "Round Robin, Weighted Round Robin, Dynamic Weighted Round Robin, Least Connections, Peak Exponentially Weighted Moving Average(PEWMA)"})

print(memory.load_memory_variables({}))
# {'history': [SystemMessage(content='The human asks why load balancers are needed. The AI explains that as web applications grow, companies need to increase their availability and scalability by deploying their application across multiple servers with a load balancer in front to distribute incoming requests. Big companies may need thousands of servers running their web application to handle the load. The human then asks for common load balancing algorithms, and the AI lists Round Robin, Weighted Round Robin, Dynamic Weighted Round Robin, Least Connections, and Peak Exponentially Weighted Moving Average(PEWMA).', additional_kwargs={})]}
