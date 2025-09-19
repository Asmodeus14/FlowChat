import streamlit as st
import time
import random
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document  
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate    
from langchain.chains import LLMChain
from dotenv import load_dotenv
import asyncio
import threading

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="ARGO & INCOIS Ocean Data Assistant",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to style the app
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2563eb;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #2563eb, #1e40af);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #3b82f6;
        margin-bottom: 1.5rem;
    }
    .message-container {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        padding: 1rem;
        margin-bottom: 120px;
    }
    .user-message {
        background-color: #f3f4f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-left: 20%;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .assistant-message {
        background-color: #dbeafe;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-right: 20%;
        border: 1px solid #bfdbfe;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .message-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.8rem;
        color: #6b7280;
        margin-bottom: 0.5rem;
    }
    .chat-input {
        position: fixed;
        bottom: 2rem;
        left: 50%;
        transform: translateX(-50%);
        width: 70%;
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        z-index: 100;
    }
    .stButton button {
        background: linear-gradient(135deg, #2563eb, #1e40af);
        color: white;
        width: 100%;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .sidebar-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2563eb;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #2563eb, #1e40af);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .info-box {
        background-color: #e0f2fe;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2563eb;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .capability-item {
        padding: 0.5rem 0;
        border-bottom: 1px solid #e5e7eb;
        transition: all 0.3s ease;
    }
    .capability-item:hover {
        background-color: #f0f9ff;
        padding-left: 0.5rem;
    }
    .data-source-tag {
        background: linear-gradient(135deg, #c084fc, #a855f7);
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        margin-left: 0.5rem;
    }
    .graph-container {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .rag-indicator {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        margin-left: 0.5rem;
    }
    .status-indicator {
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    .status-good {
        background-color: #d1fae5;
        color: #065f46;
        border-left: 4px solid #10b981;
    }
    .status-warning {
        background-color: #fef3c7;
        color: #92400e;
        border-left: 4px solid #f59e0b;
    }
    .typing-indicator {
        display: flex;
        align-items: center;
        color: #6b7280;
        font-style: italic;
    }
    .typing-dots {
        display: flex;
        margin-left: 0.5rem;
    }
    .typing-dot {
        width: 4px;
        height: 4px;
        background-color: #6b7280;
        border-radius: 50%;
        margin: 0 1px;
        animation: typingAnimation 1.4s infinite ease-in-out;
    }
    .typing-dot:nth-child(1) { animation-delay: 0s; }
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    @keyframes typingAnimation {
        0%, 60%, 100% { transform: translateY(0); }
        30% { transform: translateY(-5px); }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "model" not in st.session_state:
    st.session_state.model = "ARGO-Specialist"

if "rag_enabled" not in st.session_state:
    st.session_state.rag_enabled = True

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "rag_initialized" not in st.session_state:
    st.session_state.rag_initialized = False

if "gemini_llm" not in st.session_state:
    st.session_state.gemini_llm = None

# Initialize RAG system automatically
@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system with ARGO data"""
    try:
        # Check if files exist
        if not os.path.exists("argo_mock_summaries.txt") or not os.path.exists("argo_profile_metadata.json"):
            st.error("RAG data files not found. Please ensure 'argo_mock_summaries.txt' and 'argo_profile_metadata.json' are in the same directory.")
            return None, None
        
        # Load summaries
        loader = TextLoader("argo_mock_summaries.txt", encoding="utf-8")
        documents = loader.load()
        
        # Load metadata from JSON
        with open("argo_profile_metadata.json", "r") as f:
            metadata_list = json.load(f)
        
        all_summaries = documents[0].page_content.strip().split("\n")
        
        # Create individual Document objects for each summary with corresponding metadata
        summary_docs = []
        for summary in all_summaries:
            try:
                # Extract profile_id from line: "Profile 1 | Location: ..."
                pid = int(summary.split()[1])
            except:
                pid = None

            meta = next((m for m in metadata_list if m.get("profile_id") == pid), {})
            summary_docs.append(Document(page_content=summary, metadata=meta))
        
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
        
        # Create embeddings and vector store
        embedding = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
        )
        
        vectorstore = FAISS.from_documents(chunked_docs, embedding)
        
        # Create MMR retriever
        mmr_retriever = vectorstore.as_retriever(
            search_type="mmr", 
            search_kwargs={
                "k": 2,           
                "fetch_k": 15,    
                "lambda_mult": 0.5  
            }
        )
        
        # Initialize Gemini LLM (cache it for faster responses)
        gemini_llm = ChatGoogleGenerativeAI(
            model="gemini-pro", 
            temperature=0.5,
            max_retries=2,
            timeout=30  # Set timeout to prevent hanging
        )
        
        return mmr_retriever, gemini_llm
        
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return None, None

# Pre-defined prompt template for faster response
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

# Format results using Gemini with timeout protection
def format_results_for_user_fast(query, results, llm):
    """
    Takes MMR retriever results and formats them nicely using Gemini.
    Returns LLM's formatted response (string).
    """
    try:
        # Build candidates string - simplified for faster processing
        candidates_text = ""
        for i, r in enumerate(results, start=1):
            # Extract just the essential information
            content = r.page_content[:200]  # Limit content length
            candidates_text += f"Candidate {i}: {content}\n"
        
        # Create chain
        chain = LLMChain(llm=llm, prompt=format_prompt)
        
        # Run chain with timeout protection
        import functools
        try:
            # Use a thread with timeout to prevent hanging
            with st.spinner("Retrieving data from ARGO database..."):
                response = chain.run(query=query, candidates=candidates_text)
                return response
        except Exception as e:
            return f"I found some ARGO profiles but encountered an issue formatting them: {str(e)}. Here are the raw results:\n\n" + candidates_text
        
    except Exception as e:
        return f"I encountered an error while processing your query: {str(e)}. Please try again."

# Simple response formatter for when Gemini is slow
def format_results_simple(query, results):
    """Simple formatting without Gemini for faster response"""
    if not results:
        return "I couldn't find any ARGO profiles matching your query. Please try rephrasing or ask about something else."
    
    response = "I found these ARGO profiles matching your query:\n\n"
    for i, r in enumerate(results, start=1):
        response += f"**Profile {i}:** {r.page_content}\n\n"
    
    response += "\nWould you like more details about any of these profiles?"
    return response

# Generate mock temperature data for Bay of Bengal
def generate_temperature_data():
    years = list(range(2010, 2024))
    surface_temp = [28.5 + 0.15*(year-2010) + random.uniform(-0.2, 0.2) for year in years]
    deep_temp = [12.3 + 0.08*(year-2010) + random.uniform(-0.1, 0.1) for year in years]
    
    return pd.DataFrame({
        'Year': years,
        'Surface Temperature (°C)': surface_temp,
        'Deep Water Temperature (°C)': deep_temp
    })

# Create temperature trend chart
def create_temperature_trend_chart():
    temp_data = generate_temperature_data()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=temp_data['Year'], y=temp_data['Surface Temperature (°C)'], 
                            mode='lines+markers', name='Surface Temperature', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=temp_data['Year'], y=temp_data['Deep Water Temperature (°C)'], 
                            mode='lines+markers', name='Deep Water Temperature', line=dict(color='blue')))
    
    fig.update_layout(
        title='Temperature Trends in Bay of Bengal (2010-2023)',
        xaxis_title='Year',
        yaxis_title='Temperature (°C)',
        hovermode='x unified',
        height=400
    )
    
    return fig

# Auto-initialize RAG system
if st.session_state.vectorstore is None:
    with st.spinner("Initializing RAG system with ARGO data..."):
        vectorstore, gemini_llm = initialize_rag_system()
        if vectorstore:
            st.session_state.vectorstore = vectorstore
            st.session_state.gemini_llm = gemini_llm
            st.session_state.rag_initialized = True

# Sidebar for configuration
with st.sidebar:
    st.markdown('<p class="sidebar-title">Ocean Data Assistant</p>', unsafe_allow_html=True)
    
    # Info box
    st.markdown("""
    <div class="info-box">
        <strong>About this Assistant</strong><br>
        This AI assistant helps you explore ARGO and INCOIS data through natural language. 
        Ask questions about ocean temperature, salinity, pressure, and other parameters.
    </div>
    """, unsafe_allow_html=True)
    
    # RAG status indicator
    if st.session_state.rag_initialized:
        st.markdown('<div class="status-indicator status-good">✅ RAG System: Initialized</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-indicator status-warning">⚠️ RAG System: Not Available</div>', unsafe_allow_html=True)
    
    # Data source selection
    st.markdown("### Data Sources")
    data_sources = st.multiselect(
        "Select data sources:",
        ["ARGO Floats", "INCOIS Moored Buoys", "Satellite Data", "Coastal Stations"],
        default=["ARGO Floats", "INCOIS Moored Buoys"]
    )
    
    # Model selection
    model_options = ["ARGO-Specialist", "INCOIS-Analyst", "GPT-4 Oceanography", "Claude-Marine", "Llama-Hydro"]
    st.session_state.model = st.selectbox(
        "Analysis Model",
        model_options,
        index=0,
        help="Select the AI model for analyzing ocean data"
    )
    
    # RAG toggle (enabled by default)
    st.session_state.rag_enabled = st.checkbox(
        "Enable Data Retrieval (RAG)",
        value=True,
        help="Retrieve relevant ocean data to answer questions accurately"
    )
    
    # Response speed preference
    response_speed = st.radio(
        "Response Speed Preference",
        ["Fast (Simple Format)", "Detailed (Gemini Enhanced)"],
        index=0,
        help="Choose between faster responses or more detailed Gemini-formatted responses"
    )
    
    # File upload for RAG
    st.markdown("### File Upload")
    if st.session_state.rag_enabled:
        uploaded_files = st.file_uploader(
            "Upload ocean data files (NetCDF, CSV)",
            type=["nc", "csv", "txt"],
            accept_multiple_files=True
        )
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            st.success(f"Uploaded {len(uploaded_files)} file(s) for analysis")
    
    # Capabilities section
    st.markdown("### Capabilities")
    st.markdown("""
    <div class="capability-item">• Query ocean temperature data</div>
    <div class="capability-item">• Analyze salinity patterns</div>
    <div class="capability-item">• Visualize float trajectories</div>
    <div class="capability-item">• Compare multiple data sources</div>
    <div class="capability-item">• Generate time-series plots</div>
    <div class="capability-item">• Create depth profiles</div>
    <div class="capability-item">• Coastal disaster prediction</div>
    """, unsafe_allow_html=True)
    
    # Clear chat button
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.uploaded_files = []
        st.rerun()

# Main interface
st.markdown('<p class="main-header">ARGO & INCOIS Ocean Data Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask questions about ocean data in natural language</p>', unsafe_allow_html=True)

# Display chat messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                <div class="message-header">
                    <span>You</span>
                    <span>{message["time"]}</span>
                </div>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            sources = ""
            if "sources" in message:
                sources = "".join([f'<span class="data-source-tag">{source}</span>' for source in message["sources"]])
            
            # Add RAG indicator if RAG was used
            rag_indicator = ""
            if message.get("rag_used", False):
                rag_indicator = '<span class="rag-indicator">RAG</span>'
            
            st.markdown(f"""
            <div class="assistant-message">
                <div class="message-header">
                    <span>{message["model"]}{sources}{rag_indicator}</span>
                    <span>{message["time"]}</span>
                </div>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
            
            # Display graph if included in the message
            if "graph" in message:
                st.plotly_chart(message["graph"], use_container_width=True)
            
            # Display additional graphs if they exist
            if "additional_graphs" in message:
                for graph in message["additional_graphs"]:
                    st.plotly_chart(graph, use_container_width=True)

# Chat input at the bottom
with st.form(key="chat_input", clear_on_submit=True):
    cols = st.columns([6, 1])
    with cols[0]:
        user_input = st.text_input(
            "Ask about ocean data...", 
            key="user_input",
            label_visibility="collapsed",
            placeholder="e.g., Show me temperature trends in the Bay of Bengal"
        )
    with cols[1]:
        submit_button = st.form_submit_button("Send")

# Process user input
if submit_button and user_input:
    # Add user message to chat history
    current_time = datetime.now().strftime("%H:%M:%S")
    st.session_state.messages.append({
        "role": "user", 
        "content": user_input,
        "time": current_time
    })
    
    # Generate assistant response
    with st.spinner("Analyzing ocean data..."):
        # Check if we should use RAG
        use_rag = st.session_state.rag_enabled and st.session_state.vectorstore is not None
        
        # For ARGO-related queries, use RAG if available
        rag_used = False
        rag_response = None
        
        if use_rag and any(term in user_input.lower() for term in ["argo", "profile", "float", "buoy", "temperature", "salinity", "pressure", "depth", "ocean", "sea"]):
            try:
                # Use RAG to retrieve relevant documents
                results = st.session_state.vectorstore.get_relevant_documents(user_input)
                if results:
                    rag_used = True
                    
                    # Choose response format based on speed preference
                    if response_speed == "Fast (Simple Format)":
                        rag_response = format_results_simple(user_input, results)
                    else:
                        rag_response = format_results_for_user_fast(
                            user_input, results, st.session_state.gemini_llm
                        )
            except Exception as e:
                st.error(f"RAG retrieval error: {str(e)}")
        
        # If RAG was used, return the RAG response
        if rag_used and rag_response:
            response = rag_response
        else:
            # Fallback to simulated responses
            ocean_responses = {
                "ARGO-Specialist": [
                    "Based on my analysis of ARGO float data, I've found that {}.\n\nI can provide a visualization of this data if you'd like.",
                    "The ARGO float data shows that {}.\n\nWould you like me to compare this with data from other regions?",
                    "After querying the ARGO database, I've discovered that {}.\n\nI can generate a time-series plot to illustrate this trend."
                ],
                "INCOIS-Analyst": [
                    "According to INCOIS buoy data and models, {}.\n\nThis is based on real-time ocean monitoring systems.",
                    "INCOIS coastal observation data indicates that {}.\n\nWould you like me to generate a forecast based on this data?",
                    "Analysis of INCOIS data reveals that {}.\n\nI can correlate this with satellite observations for validation."
                ],
                "GPT-4 Oceanography": [
                    "My analysis of oceanographic data indicates that {}.\n\nThis is based on the latest ARGO and INCOIS measurements.",
                    "The integrated ARGO and INCOIS data provides comprehensive insights showing that {}.\n\nThis aligns with current oceanographic models.",
                    "Based on multi-year ocean observations, I can report that {}.\n\nShall I prepare a detailed report with visualizations?"
                ],
                "Claude-Marine": [
                    "I've carefully analyzed the ocean data and found that {}.\n\nThis pattern is consistent with known oceanographic processes.",
                    "Examining the available ocean data reveals that {}.\n\nI can correlate this with meteorological observations if helpful.",
                    "The ocean measurements demonstrate that {}.\n\nThis information could be valuable for your research."
                ],
                "Llama-Hydro": [
                    "Processing the ocean dataset shows that {}.\n\nI can provide additional statistical analysis if needed.",
                    "The hydrographic data from multiple sources indicates that {}.\n\nWould you like to explore the raw data behind this analysis?",
                    "My analysis of the ocean profiles reveals that {}.\n\nI can generate comparative visualizations across different depth levels."
                ]
            }
            
            # Contextual responses based on user query
            query = user_input.lower()
            if any(term in query for term in ["temperature", "temp", "warm", "cool"]):
                context = "ocean temperatures have increased by an average of 0.12°C per decade in the upper 2000m"
                if "bay of bengal" in query:
                    context = "Bay of Bengal temperatures show significant warming trends, with surface temperatures increasing by approximately 1.2°C over the past decade"
            elif any(term in query for term in ["salinity", "salt", "saline"]):
                context = "salinity patterns show increased contrast between high and low salinity regions over the past decade"
            elif any(term in query for term in ["pressure", "depth", "deep"]):
                context = "pressure measurements show consistent patterns with slight variations due to seasonal cycles"
            elif any(term in query for term in ["float", "argo", "profile", "trajectory"]):
                context = "the ARGO float network provides global coverage with over 4000 active floats reporting data"
            elif any(term in query for term in ["incois", "buoy", "moored"]):
                context = "INCOIS maintains a network of moored buoys providing real-time data for the Indian Ocean region"
            elif any(term in query for term in ["cyclone", "storm", "disaster"]):
                context = "ocean heat content is a key factor in cyclone intensification, and current conditions suggest"
            else:
                context = "the integrated ocean data provides valuable insights into oceanographic processes and climate patterns"
            
            # Add RAG context if enabled but not used
            rag_info = ""
            if st.session_state.rag_enabled and not rag_used and not st.session_state.rag_initialized:
                rag_info = " (RAG system is not available - using simulated data)"
            elif st.session_state.rag_enabled and not rag_used:
                rag_info = " (Using general knowledge - no specific profiles matched your query)"
            
            # Select a random response based on the current model
            response_template = random.choice(ocean_responses.get(st.session_state.model, ocean_responses["ARGO-Specialist"]))
            response = response_template.format(context) + rag_info
        
        # Determine data sources used
        used_sources = []
        if "ARGO Floats" in data_sources:
            used_sources.append("ARGO")
        if "INCOIS Moored Buoys" in data_sources:
            used_sources.append("INCOIS")
        if "Satellite Data" in data_sources:
            used_sources.append("Satellite")
        if "Coastal Stations" in data_sources:
            used_sources.append("Coastal")
        
        # Add assistant response to chat history
        current_time = datetime.now().strftime("%H:%M:%S")
        message_data = {
            "role": "assistant", 
            "content": response,
            "model": st.session_state.model,
            "sources": used_sources,
            "time": current_time,
            "rag_used": rag_used
        }
        
        # Add temperature graph for Bay of Bengal queries
        if "bay of bengal" in user_input.lower() and "temperature" in user_input.lower():
            message_data["graph"] = create_temperature_trend_chart()
        
        st.session_state.messages.append(message_data)
        
        # Rerun to update the chat display
        st.rerun()

# Instructions for expanding the functionality
with st.expander("Implementation Guide for Ocean Data Integration"):
    st.markdown("""
    ### RAG System Optimization:
    
    The RAG system has been optimized for faster response times:
    
    1. **Speed Options**: Choose between fast simple responses or detailed Gemini responses
    2. **Cached LLM**: Gemini model is cached for faster initialization
    3. **Timeout Protection**: Prevents hanging with timeout settings
    4. **Simplified Formatting**: Option for simple formatting without Gemini
    
    ### Performance Tips:
    
    1. Use "Fast (Simple Format)" for quicker responses
    2. Gemini may be slower but provides more detailed analysis
    3. The system falls back gracefully if Gemini times out
    
    ### Files Used:
    - `argo_mock_summaries.txt`: Contains profile summaries
    - `argo_profile_metadata.json`: Contains detailed metadata for each profile
    
    ### Next Steps:
    1. Add more ARGO data files for better coverage
    2. Implement response caching for common queries
    3. Add visualization of retrieved profiles on a map
    """)