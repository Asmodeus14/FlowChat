import json
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document  
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate    
from langchain.chains import LLMChain
load_dotenv()


#DATA LOADING AND CHUNKING

#Loading summaries
loader = TextLoader("argo_mock_summaries.txt", encoding="utf-8")
documents = loader.load()

print(f"Loaded {len(documents)} documents")
print(documents[0].page_content)

# Load metadata from JSON
with open("argo_profile_metadata.json", "r") as f:
    metadata_list = json.load(f)  # list of dicts
print(f"Loaded {len(metadata_list)} metadata entries")

all_summaries = documents[0].page_content.strip().split("\n")
print(f"Found {len(all_summaries)} summaries in the text file")

# Create individual Document objects for each summary with corresponding metadata
summary_docs = []
for summary in all_summaries:
    # Extract profile_id from line: "ARGO profile 12 recorded ..."
    try:
        pid = int(summary.split()[1])
    except:
        pid = None

    meta = next((m for m in metadata_list if m.get("profile_id") == pid), {})
    summary_docs.append(Document(page_content=summary, metadata=meta))

print(f"Created {len(summary_docs)} individual summary Documents")

# Chunk summaries
splitter = RecursiveCharacterTextSplitter(
    chunk_size=130,
    chunk_overlap=0,
)

chunked_docs = []
for doc in summary_docs:
    chunks = splitter.split_text(doc.page_content)
    for chunk in chunks:
        chunked_docs.append(Document(page_content=chunk, metadata=doc.metadata))

print(f"Final: {len(chunked_docs)} chunked Documents")
print("Sample chunk with metadata:")
print(chunked_docs[0].page_content)
print(chunked_docs[0].metadata)

#EMBEDDING AND VECTOR STORE

embedding= HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    )

vectorstore = FAISS.from_documents(chunked_docs, embedding)
vectorstore.save_local("argo_faiss_index")
print("FAISS index built and saved successfully.")

vectorstore = FAISS.load_local(
    "argo_faiss_index",
    embedding,
    allow_dangerous_deserialization=True  
)

mmr_retriever = vectorstore.as_retriever(
    search_type="mmr", 
    search_kwargs={
        "k": 2,           
        "fetch_k": 15,    
        "lambda_mult": 0.5  
    }
)


query = "Find profiles near Bay of Bengal with warm surface temperature"
results = mmr_retriever.get_relevant_documents(query)

print(f"Retrieved {len(results)} results using MMR")
for r in results:
    print(r.page_content, "| Metadata:", r.metadata)

import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


format_prompt = PromptTemplate(
    input_variables=["query", "candidates"],
    template=(
        "You are an ocean data assistant. A user asked the following query:\n"
        "Query: {query}\n\n"
        "Here are the top candidate ARGO profiles retrieved from the database.\n"
        "Analyze them carefully and display the BEST matching profile to the user "
        "in a clean, human-friendly format like a short report.\n"
        "Include:\n"
        "• Profile ID\n"
        "• Location (lat, lon)\n"
        "• Date\n"
        "• Depth Range\n"
        "• Region\n"
        "• Key observations (temperature, salinity)\n\n"
        "Candidates:\n{candidates}\n\n"
        "Choose the single best match and format your answer clearly like this:\n\n"
        "Profile ID: <id>\n"
        "Date: <date>\n"
        "Location: (lat, lon)\n"
        "Depth Range: <range>\n"
        "Region: <region>\n"
        "Key Observations:\n"
        "• observation 1\n"
        "• observation 2\n"
    )
)

def format_results_for_user(query, results):
    """
    Takes MMR retriever results and formats them nicely using Gemini.
    Returns LLM's formatted response (string).
    """
    # Build candidates string once
    candidates_text = ""
    for i, r in enumerate(results, start=1):
        candidates_text += (
            f"\nCandidate {i}:\n"
            f"Summary: {r.page_content}\n"
            f"Metadata: {json.dumps(r.metadata)}\n"
        )

    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)

    # Create chain
    chain = LLMChain(llm=llm, prompt=format_prompt)

    # Run chain
    response = chain.run(query=query, candidates=candidates_text)
    print("LLM formatted response:")
    print(type(response))
    return response

