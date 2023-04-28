# Create a .env file and add "OPENAI_API_KEY=..." to it"
# Run Motorhead parallelly
# docker run --name motorhead -p 8080:8080 -e MOTORHEAD_PORT=8080 -e REDIS_URL='redis://redis:6379' -e MOTORHEAD_LONG_TERM_MEMORY=true -e MOTORHEAD_MAX_WINDOW_SIZE=2 -e OPENAI_API_KEY='sk-...' -d ghcr.io/getmetal/motorhead:latest

from pprint import pprint
import asyncio
from flask import Flask, request
from langchain.memory.motorhead_memory import MotorheadMemory
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


memory = MotorheadMemory(
    session_id="testing-1", url="http://localhost:8080", memory_key="chat_history", timeout=5)
print('Initializing memory')
asyncio.run(memory.init())  # loads previous state from Motörhead
print('Done initializing memory')
pprint(vars(memory))


template = f"""You are a chatbot having a conversation with a human. Answer the question based on the context below.
If the question cannot be answered using the information provided answer with "I don't know"

{memory.context}
"""

template += """
{chat_history}
Human: {human_input}
AI:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"], template=template)
llm_chain = LLMChain(llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
                     prompt=prompt, verbose=True, memory=memory)


app = Flask(__name__)


@app.route("/")
def hello_world():
    message = request.json.get('message')
    if not message:
        return {'error': 'Message not provided'}, 400

    return {'response': llm_chain.run(message)}
