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
                "k": 3,           # Increased from 2 to 3 for more context
                "fetch_k": 20,    # Increased from 15 to 20 for better diversity
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

# Dynamic prompt template that adapts to different query types
def get_dynamic_prompt_template(query):
    """Return appropriate prompt template based on query content"""
    query_lower = query.lower()
    
    if any(term in query_lower for term in ["temperature", "temp", "warm", "cool", "heat"]):
        return PromptTemplate(
            input_variables=["query", "candidates"],
            template=(
                "You are an ocean data assistant specializing in temperature analysis. A user asked:\n"
                "Query: {query}\n\n"
                "Here are relevant ARGO profiles from the database:\n"
                "{candidates}\n\n"
                "Provide a concise analysis focusing on temperature patterns. Include:\n"
                "• Key temperature observations\n"
                "• Spatial patterns if available\n"
                "• Notable trends or anomalies\n"
                "• Recommendations for further exploration\n"
                "Format your response in clear, readable paragraphs."
            )
        )
    elif any(term in query_lower for term in ["salinity", "salt", "saline"]):
        return PromptTemplate(
            input_variables=["query", "candidates"],
            template=(
                "You are an ocean data assistant specializing in salinity analysis. A user asked:\n"
                "Query: {query}\n\n"
                "Here are relevant ARGO profiles from the database:\n"
                "{candidates}\n\n"
                "Provide a concise analysis focusing on salinity patterns. Include:\n"
                "• Key salinity observations\n"
                "• Freshwater influence indicators\n"
                "• Notable patterns or anomalies\n"
                "• Connections to broader oceanographic processes\n"
                "Format your response in clear, readable paragraphs."
            )
        )
    elif any(term in query_lower for term in ["profile", "float", "buoy", "argo"]):
        return PromptTemplate(
            input_variables=["query", "candidates"],
            template=(
                "You are an ARGO float data assistant. A user asked:\n"
                "Query: {query}\n\n"
                "Here are relevant ARGO profiles from the database:\n"
                "{candidates}\n\n"
                "Provide information about the ARGO profiles. Include:\n"
                "• Profile locations and dates\n"
                "• Key measurements and observations\n"
                "• Patterns across different profiles\n"
                "• Significance of the findings\n"
                "Format your response in clear, readable paragraphs."
            )
        )
    else:
        # General purpose template for other queries
        return PromptTemplate(
            input_variables=["query", "candidates"],
            template=(
                "You are an ocean data assistant. A user asked:\n"
                "Query: {query}\n\n"
                "Here are relevant ARGO profiles from the database:\n"
                "{candidates}\n\n"
                "Provide a helpful response that incorporates the ARGO data. Include:\n"
                "• Key insights from the data\n"
                "• Relevant patterns or observations\n"
                "• Context about what the data means\n"
                "• Suggestions for further exploration if appropriate\n"
                "Format your response in clear, readable paragraphs."
            )
        )

# Format results using Gemini with dynamic prompts
def format_results_dynamic(query, results, llm):
    """
    Takes MMR retriever results and formats them using dynamic prompts.
    Returns LLM's formatted response (string).
    """
    try:
        # Build candidates string
        candidates_text = ""
        for i, r in enumerate(results, start=1):
            candidates_text += (
                f"\nProfile {i}:\n"
                f"Data: {r.page_content}\n"
                f"Metadata: {json.dumps(r.metadata)}\n"
            )

        # Get appropriate prompt template based on query
        prompt_template = get_dynamic_prompt_template(query)
        
        # Create chain
        chain = LLMChain(llm=llm, prompt=prompt_template)
        
        # Run chain
        response = chain.run(query=query, candidates=candidates_text)
        return response
        
    except Exception as e:
        return f"I found some ARGO profiles but encountered an issue formatting them: {str(e)}. Here are the raw results:\n\n" + candidates_text

# Simple response formatter for when Gemini is slow
def format_results_simple(query, results):
    """Simple formatting without Gemini for faster response"""
    if not results:
        return "I couldn't find any ARGO profiles matching your query. Please try rephrasing or ask about something else."
    
    response = "Based on ARGO data, I found these relevant profiles:\n\n"
    for i, r in enumerate(results, start=1):
        # Extract the most relevant part of the profile
        profile_text = r.page_content
        if len(profile_text) > 150:
            profile_text = profile_text[:150] + "..."
        
        response += f"**Profile {i}:** {profile_text}\n\n"
    
    response += "\nWould you like more specific information about any of these profiles?"
    return response

# Check if a query is relevant for RAG
def should_use_rag(query):
    """Determine if a query is relevant for RAG based on content"""
    query_lower = query.lower()
    
    # List of ocean-related terms that might benefit from RAG
    ocean_terms = [
        "argo", "float", "buoy", "profile", "incois", "ocean", "sea", 
        "temperature", "temp", "salinity", "salt", "pressure", "depth",
        "current", "wave", "coastal", "marine", "water", "thermocline",
        "halocline", "bengal", "arabian", "indian ocean", "pacific", "atlantic",
        "southern ocean", "polar", "tropical", "subtropical", "climate",
        "warming", "cooling", "trend", "pattern", "data", "measurement",
        "observation", "profile", "cast", "ctd", "conductivity", "density"
    ]
    
    # Check if query contains any ocean-related terms
    return any(term in query_lower for term in ocean_terms)

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
        
        # For ocean-related queries, use RAG if available and relevant
        rag_used = False
        rag_response = None
        
        if use_rag and should_use_rag(user_input):
            try:
                # Use RAG to retrieve relevant documents
                results = st.session_state.vectorstore.get_relevant_documents(user_input)
                if results and len(results) > 0:
                    rag_used = True
                    
                    # Choose response format based on speed preference
                    if response_speed == "Fast (Simple Format)":
                        rag_response = format_results_simple(user_input, results)
                    else:
                        rag_response = format_results_dynamic(
                            user_input, results, st.session_state.gemini_llm
                        )
            except Exception as e:
                st.error(f"RAG retrieval error: {str(e)}")
        
        # If RAG was used, return the RAG response
        if rag_used and rag_response:
            response = rag_response
        else:
            # Fallback to simulated responses for non-ocean queries or when RAG fails
            ocean_responses = {
                "ARGO-Specialist": [
                    "Based on my analysis of ocean data, I've found that {}.\n\nI can provide more specific information if you'd like.",
                    "The oceanographic data shows that {}.\n\nWould you like me to compare this with data from other regions?",
                    "After examining available data, I've discovered that {}.\n\nI can generate visualizations to illustrate these patterns."
                ],
                "INCOIS-Analyst": [
                    "According to ocean monitoring systems, {}.\n\nThis is based on real-time data collection.",
                    "Coastal observation data indicates that {}.\n\nWould you like me to generate a forecast based on this data?",
                    "Analysis of marine data reveals that {}.\n\nI can correlate this with satellite observations for validation."
                ],
                "GPT-4 Oceanography": [
                    "My analysis of oceanographic information indicates that {}.\n\nThis is based on the latest available measurements.",
                    "The integrated ocean data provides comprehensive insights showing that {}.\n\nThis aligns with current oceanographic models.",
                    "Based on multi-year observations, I can report that {}.\n\nShall I prepare a detailed report with visualizations?"
                ],
                "Claude-Marine": [
                    "I've carefully analyzed the available data and found that {}.\n\nThis pattern is consistent with known oceanographic processes.",
                    "Examining the ocean data reveals that {}.\n\nI can correlate this with meteorological observations if helpful.",
                    "The measurements demonstrate that {}.\n\nThis information could be valuable for your research."
                ],
                "Llama-Hydro": [
                    "Processing the available dataset shows that {}.\n\nI can provide additional statistical analysis if needed.",
                    "The hydrographic data indicates that {}.\n\nWould you like to explore the data behind this analysis?",
                    "My analysis reveals that {}.\n\nI can generate comparative visualizations across different parameters."
                ]
            }
            
            # Contextual responses based on user query
            query = user_input.lower()
            if any(term in query for term in ["temperature", "temp", "warm", "cool"]):
                context = "ocean temperatures have shown variability with some regions experiencing warming trends"
                if "bay of bengal" in query:
                    context = "Bay of Bengal temperatures show interesting patterns with surface variations"
            elif any(term in query for term in ["salinity", "salt", "saline"]):
                context = "salinity patterns vary across different ocean regions based on multiple factors"
            elif any(term in query for term in ["pressure", "depth", "deep"]):
                context = "pressure and depth measurements reveal important information about ocean structure"
            elif any(term in query for term in ["float", "argo", "profile", "trajectory"]):
                context = "the ARGO float network provides valuable data about global ocean conditions"
            elif any(term in query for term in ["incois", "buoy", "moored"]):
                context = "INCOIS monitoring systems contribute important data for understanding ocean processes"
            elif any(term in query for term in ["cyclone", "storm", "disaster"]):
                context = "ocean conditions play a significant role in cyclone formation and intensification"
            else:
                context = "ocean data provides valuable insights into marine environments and climate patterns"
            
            # Add RAG context if enabled but not used
            rag_info = ""
            if st.session_state.rag_enabled and not rag_used and not st.session_state.rag_initialized:
                rag_info = " (RAG system is not available - using general knowledge)"
            elif st.session_state.rag_enabled and not rag_used:
                rag_info = " (Using general ocean knowledge)"
            
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
    ### Dynamic RAG System:
    
    The RAG system now dynamically adapts to different types of queries:
    
    1. **Smart Query Detection**: Automatically detects ocean-related queries
    2. **Dynamic Prompts**: Uses different prompt templates based on query content
    3. **Flexible Response**: Works with both specific and general ocean questions
    
    ### Query Types Handled:
    
    - **Temperature queries**: Focus on thermal patterns and trends
    - **Salinity queries**: Analyze salt concentration patterns  
    - **ARGO profile queries**: Provide information about specific floats
    - **General ocean queries**: Contextual responses with relevant data
    
    ### How It Works:
    
    1. User asks any ocean-related question
    2. System detects if RAG would be beneficial
    3. Retrieves relevant ARGO profiles
    4. Formats response based on query type
    5. Provides appropriate visualizations when relevant
    
    ### Examples of Questions You Can Ask:
    
    - "What's the temperature in the Bay of Bengal?"
    - "Show me ARGO profiles with high salinity"
    - "Tell me about ocean currents in the Indian Ocean"
    - "How does depth affect temperature?"
    - "What ARGO data is available from last year?"
    """)