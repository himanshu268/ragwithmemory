import boto3
from langchain_aws import BedrockLLM
import os
from dotenv import load_dotenv
import warnings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import BedrockEmbeddings
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder)
from langchain_core.messages import HumanMessage, SystemMessage


# Suppress all warnings
warnings.filterwarnings("ignore")

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

curr_dir=os.path.dirname(os.path.abspath(__file__))
db_dir=os.path.join(curr_dir, "db")
persistant_dir=os.path.join(db_dir, "chromadb")

loader=PyPDFLoader("attention.pdf")
documents=loader.load()

# print(documents)

text_splitter=RecursiveCharacterTextSplitter(chunk_size=500)
doc_split=text_splitter.split_documents(documents)
print(len(doc_split))
# print(doc_split)
# print(len(doc_split[0]))

embeddings = BedrockEmbeddings(model_id="amazon.titan-text-lite-v1",region_name="ap-south-1")
if not os.path.exists(persistant_dir):
    print("creating database")
    vectorstore=Chroma.from_documents(documents=doc_split[2:5],embedding=embeddings,persist_directory=persistant_dir)
    print("database created successfully")
else:
    vectorstore=Chroma(embedding_function=embeddings,persist_directory=persistant_dir)
retriever=vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3}
    )

client=boto3.client("bedrock-runtime",region_name="ap-south-1")
model_id="amazon.titan-text-express-v1"
llm = BedrockLLM(client=client, model_id=model_id, max_tokens=200, temperature=0.7)

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


def continual_chat():
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    chat_history = []  # Collect chat history here (a sequence of messages)
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        # Process the user's query through the retrieval chain
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        # Display the AI's response
        print(f"AI: {result['answer']}")
        # Update the chat history
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result["answer"]))

if __name__ == "__main__":
    continual_chat()