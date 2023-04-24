from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
# import os

load_dotenv()

loader = TextLoader('test_chunks.txt')
documents = loader.load()
print("Total documents:")
print(len(documents))

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=100,
    chunk_overlap=0,
    length_function=len,
)

texts = text_splitter.split_documents(documents)
print("\nChunks:")
print(len(texts))
print(texts)

# os._exit(0)

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)

query = "What's the final release date for UPI autopay?"
print(query)

retriever = db.as_retriever()
docs = retriever.get_relevant_documents(query)
print("\nRelevant documents:")
print(len(docs))
print(docs)

chain = load_qa_chain(OpenAI(temperature=0),
                      chain_type="stuff")#, return_refine_steps=True)
op = chain({"input_documents": docs, "question": query},
           return_only_outputs=True)
print("\nAnswer:")
print(op)

# op = chain.run(input_documents=docs, question=query)
# print(op)

# query = "What's the release date for UPI autopay?"
# print(query)
# docs_and_scores = db.similarity_search_with_score(query)

# print(len(docs_and_scores))
# print(docs_and_scores)


# -------------------------------------------------------------------------

# Outputs sample_chats.txt

# Refine
# Answer:
# {'intermediate_steps': ['\nThe final release date for UPI autopay is 21st May.',
#                         '\n\nThe final release date for UPI autopay is 18th May.'], 'output_text': '\n\nThe final release date for UPI autopay is 18th May.'}

# Stuff
# Answer:
# {'output_text': ' 18th May.'}


# Output test_chunks.txt

# Stuff
# Relevant documents:
# 4
# [Document(page_content='(2023-04-20 20:10:00) The release date for UPI autopay is 3rd March', metadata={'source': 'test_chunks.txt'}),
#   Document(page_content='(2023-04-20 20:50:00) The release date for UPI autopay is 7th March', metadata={'source': 'test_chunks.txt'}),
#   Document(page_content='(2023-04-20 20:40:00) The release date for UPI autopay is 6th March', metadata={'source': 'test_chunks.txt'}),
#   Document(page_content='(2023-04-20 20:20:00) The release date for UPI autopay is 4th March', metadata={'source': 'test_chunks.txt'})]
# Answer:
# {'output_text': ' The final release date for UPI autopay is 6th March.'}
