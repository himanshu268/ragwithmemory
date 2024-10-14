import boto3
from langchain_aws import BedrockLLM
import os
from dotenv import load_dotenv
import warnings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import BedrockEmbeddings
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder)
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader

# Suppress all warnings
warnings.filterwarnings("ignore")
load_dotenv()

# Set up environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Define paths for persistence
curr_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(curr_dir, "db")
persistant_dir = os.path.join(db_dir, "chromadb")

# Load PDF and extract text
try:
    text = ""
    loader = PdfReader("attention.pdf")
    for page in loader.pages:
        text += page.extract_text()
except Exception as e:
    print(f"Error loading PDF: {e}")
    exit()

# Split the extracted text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
doc_split = text_splitter.split_text(text=text)

# Create or load the vector store with FAISS
embeddings = BedrockEmbeddings(model_id="amazon.titan-text-lite-v1", region_name="ap-south-1")
if not os.path.exists(persistant_dir):
    print("Creating FAISS database...")
    vectorstore = FAISS.from_texts(texts=doc_split, embedding=embeddings)
    vectorstore.save_local(persistant_dir)
    print("Database created successfully.")
else:
    vectorstore = FAISS.load_local(persistant_dir, embedding=embeddings)

# Set up the Bedrock LLM
client = boto3.client("bedrock-runtime", region_name="ap-south-1")
model_id = "amazon.titan-text-express-v1"
llm = BedrockLLM(client=client, model_id=model_id, max_tokens=200, temperature=0.7)

# System and history prompts
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question, "
    "reformulate a standalone question without the chat history."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the retrieved context to answer the question in three sentences maximum."
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

# Main chat loop
if __name__ == "__main__":
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    chat_history = []  # Store chat history

    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break

        # Search in vector store
        retriever = vectorstore.similarity_search(query=query, k=3)

        # Create chains
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Get the result and display AI's response
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        print(f"AI: {result.get('answer', 'No answer available')}")

        # Update chat history
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result["answer"]))
