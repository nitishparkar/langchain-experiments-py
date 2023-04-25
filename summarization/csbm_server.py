# Create a .env file and add "OPENAI_API_KEY=..." to it"
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from flask import Flask, request
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model_name="gpt-3.5-turbo", request_timeout=10, max_retries=2)

chain = ConversationChain(
    llm=llm,
    # low max_token_limit for testing.
    memory=ConversationSummaryBufferMemory(llm=llm, max_token_limit=100),
    verbose=True
)


app = Flask(__name__)


@app.route("/")
def hello_world():
  message = request.json.get('message')
  if not message:
      return {'error': 'Message not provided'}, 400

  return {'response': chain.run(message)}


# Example 1

# Current conversation:
# System: Coco introduces themselves and asks for the AI's name. The AI responds that it is OpenAI, designed to communicate with humans and provide helpful responses. The AI expresses pleasure in meeting Coco and offers to answer any questions. Coco asks the AI for their name, to which the AI responds.
# AI: Your name is Coco. You just introduced yourself to me a few moments ago.
# Human: WhatOh, right. I am a boy. I like cycling, trekking and reading.
# AI: That's great to hear! Cycling and trekking are both excellent ways to stay active and explore the outdoors. As for reading, I can recommend many different books depending on your interests. What type of books do you enjoy?
# Human: I am currently reading Ego is the enemy. Recently finished Indian Summer: The Secret History of the End of an Empire

# After Summary it lost the details of my interests:

# Current conversation:
# System: Coco introduces themselves to OpenAI and the AI responds, expressing pleasure in meeting Coco and offering to answer any questions. When Coco asks for the AI's name, the AI reveals it is OpenAI and designed to communicate with humans. The AI then compliments Coco's interests and recommends books similar to the ones Coco has recently read.
# Human: I have read Sapiens. I liked it.


# Lost details of books mentioned

# Current conversation:
# System: Coco introduces themselves to OpenAI and the AI offers book recommendations to Coco, who is interested in reading "Guns, Germs, and Steel". When Coco asks for recommendations regarding their other interests, which include cycling, trekking, and sketching, the AI suggests local bike trails and group rides, hiking trails and backpacking trips, and art classes/workshops, as well as online resources and tutorials. The AI offers to provide more specific recommendations if needed.
# Human: Thanks, that was helpful.
