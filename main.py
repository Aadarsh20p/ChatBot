#main.py
import streamlit as st
import os

# Import the chatbot class - make sure this matches your file structure
try:
    from utils import HallucinationResistantChatbot
    st.success("✅ Successfully imported chatbot class")
except ImportError as e:
    st.error(f"❌ Failed to import chatbot class: {e}")
    st.error("Make sure utils.py is in the same directory as main.py")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Hallucination-Resistant AI Chatbot",
    page_icon="🧭",
    layout="wide"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'confidence_scores' not in st.session_state:
    st.session_state.confidence_scores = []
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None

@st.cache_resource
def load_chatbot():
    """Load and cache the chatbot"""
    with st.spinner("🤖 Loading AI models... This may take a few minutes on first run."):
        return HallucinationResistantChatbot()

# UI Layout
st.title("🧭 Hallucination-Resistant AI Chatbot")
st.markdown("*Using RAG, NLI verification, and multi-tier fallbacks*")

# Sidebar with information
with st.sidebar:
    st.header("📊 System Information")
    st.markdown("""
    **Features:**
    - 🎯 Intent Classification
    - 📚 RAG (Retrieval-Augmented Generation)
    - ✅ NLI Fact Verification
    - 🔄 Multi-tier Fallbacks
    - 📈 Confidence Scoring
    """)
    
    st.header("🔧 Model Configuration")
    
    # Response Mode Selection Dropdown
    st.subheader("🎯 Response Mode")
    
    response_modes = {
        "⚡ Fast Mode (Instant Responses)": {
            "mode": "fast",
            "description": "Lightning-fast intelligent responses using knowledge base + AI patterns",
            "speed": "< 1 second",
            "quality": "High",
            "use_case": "Demos, Testing, General Use"
        },
        "🚀 Real OpenChat-3.5 (Authentic 7B)": {
            "mode": "real",
            "description": "Authentic responses from the full 7B parameter OpenChat model",
            "speed": "30-60 seconds",
            "quality": "Highest",
            "use_case": "Research, Benchmarking"
        },
        "🔄 Hybrid Mode (Best of Both)": {
            "mode": "hybrid",
            "description": "Try real OpenChat first, fallback to fast simulation if too slow",
            "speed": "1-30 seconds",
            "quality": "Adaptive",
            "use_case": "Production, Presentations"
        },
        "🏃 Speed Demo (Ultra Fast)": {
            "mode": "demo",
            "description": "Instant responses optimized for live demonstrations",
            "speed": "< 0.5 seconds",
            "quality": "Good",
            "use_case": "Live Demos, Quick Tests"
        }
    }
    
    selected_mode = st.selectbox(
        "Choose Response Mode:",
        options=list(response_modes.keys()),
        index=0,  # Default to Fast Mode
        help="Select how you want the chatbot to generate responses"
    )
    
    # Display mode information
    mode_info = response_modes[selected_mode]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("⏱️ Speed", mode_info["speed"])
    with col2:
        st.metric("🎯 Quality", mode_info["quality"])
    with col3:
        st.metric("📊 Best For", mode_info["use_case"])
    
    st.info(f"**{selected_mode}**: {mode_info['description']}")
    
    # Store the selected mode
    st.session_state['response_mode'] = mode_info['mode']
    
    # Show warnings based on mode
    if mode_info['mode'] == 'real':
        st.warning("⚠️ **Real OpenChat Mode**: Responses will be very slow (30-60 seconds) but completely authentic. Make sure you have 16GB+ RAM.")
    elif mode_info['mode'] == 'hybrid':
        st.info("🔄 **Hybrid Mode**: Will attempt real OpenChat but timeout after 10 seconds and use fast simulation.")
    elif mode_info['mode'] == 'demo':
        st.success("🏃 **Speed Demo Mode**: Optimized for the fastest possible responses during live presentations.")
    else:
        st.success("⚡ **Fast Mode**: Perfect balance of speed and quality for most use cases.")
    
    # OpenChat model path (only needed for real/hybrid modes)
    if mode_info['mode'] in ['real', 'hybrid']:
        st.subheader("📁 OpenChat Model Path")
        default_path = os.path.join(os.getcwd(), "models", "openchat_3.5")
        openchat_path = st.text_input(
            "OpenChat Model Path", 
            value=default_path,
            placeholder="models/openchat_3.5",
            help="Path to your OpenChat model folder (required for Real and Hybrid modes)"
        )
        
        if openchat_path:
            st.session_state['openchat_path'] = openchat_path
            if os.path.exists(openchat_path):
                # Check if required files exist
                required_files = ['config.json', 'tokenizer_config.json']
                missing_files = [f for f in required_files if not os.path.exists(os.path.join(openchat_path, f))]
                
                if not missing_files:
                    st.success("🟢 OpenChat Model: Ready")
                else:
                    st.warning(f"🟡 OpenChat Model: Missing files: {missing_files}")
            else:
                st.error("🔴 OpenChat Path: Folder not found")
        else:
            st.warning("🟡 OpenChat Path: Not set")
    else:
        st.session_state['openchat_path'] = ""  # Not needed for fast modes
    
    st.markdown("""
    **🤖 Intelligent Model Chain:**
    - ⚡ Fast Simulation: Instant intelligent responses
    - 🚀 Real OpenChat-3.5: Authentic 7B model (slow)
    - 🔄 Hybrid: Smart timeout + fallback
    - 🏃 Demo: Ultra-fast for presentations
    
    **📊 Current Mode Features:**
    """)
    
    # Show current mode info
    current_mode = st.session_state.get('response_mode', 'fast')
    mode_descriptions = {
        'fast': "⚡ **Fast Mode**: Knowledge base + AI patterns for instant responses",
        'real': "🚀 **Real Mode**: Authentic 7B OpenChat model (30-60s per response)",
        'hybrid': "🔄 **Hybrid Mode**: Real OpenChat with 10s timeout → Fast fallback",
        'demo': "🏃 **Demo Mode**: Ultra-optimized for live presentations (<0.5s)"
    }
    
    st.info(mode_descriptions.get(current_mode, "⚡ Fast Mode Active"))
    
    st.markdown("""
    **✅ Core Features:**
    - Complete RAG pipeline with FAISS search
    - NLI verification and confidence scoring  
    - Multi-tier fallback system
    - Comprehensive knowledge base (20+ topics)
    - Response time tracking
    - No API dependencies (fully local)
    """)
    
    # Performance comparison
    with st.expander("📊 Mode Comparison"):
        st.markdown("""
        | Mode | Speed | Quality | RAM Usage | Best For |
        |------|-------|---------|-----------|----------|
        | 🏃 Demo | <0.5s | Good | <2GB | Live Presentations |
        | ⚡ Fast | <1s | High | <4GB | General Use, Testing |
        | 🔄 Hybrid | 1-30s | Adaptive | <8GB | Production |
        | 🚀 Real | 30-60s | Highest | 14GB+ | Research, Benchmarks |
        """)
    
    st.header("🔧 Settings")
    show_technical_details = st.checkbox("Show Technical Details", value=False)
    max_context_docs = st.slider("Max Context Documents", 1, 5, 3)
    
    # Show mode-specific info
    current_mode = st.session_state.get('response_mode', 'fast')
    if current_mode == 'real':
        st.warning("💡 Real mode: Requires 16GB+ RAM and patience!")
    elif current_mode == 'hybrid':
        st.info("💡 Hybrid mode: Best of both worlds!")
    elif current_mode == 'demo':
        st.success("💡 Demo mode: Perfect for live presentations!")
    else:
        st.success("💡 Fast mode: Optimal for most users!")

# Load chatbot
if st.session_state.chatbot is None:
    try:
        st.session_state.chatbot = load_chatbot()
        st.success("✅ Chatbot loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading chatbot: {str(e)}")
        st.stop()

chatbot = st.session_state.chatbot

# Chat interface
st.header("💬 Chat Interface")

# Display chat history
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        # Show confidence and verification status for assistant messages
        if message["role"] == "assistant" and i < len(st.session_state.confidence_scores):
            score_info = st.session_state.confidence_scores[i]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence", f"{score_info['confidence']:.1%}")
            with col2:
                st.write(f"Status: {score_info['status']}")
            with col3:
                if show_technical_details:
                    st.write(f"Intent: {score_info.get('intent', 'N/A')}")

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking and verifying..."):
            response, confidence, status = chatbot.process_query(prompt)
            intent = chatbot.classify_input(prompt)
        
        st.write(response)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Confidence", f"{confidence:.1%}")
        with col2:
            st.write(f"Status: {status}")
        with col3:
            if show_technical_details:
                st.write(f"Intent: {intent}")
    
    # Save assistant message and scores
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.confidence_scores.append({
        "confidence": confidence,
        "status": status,
        "intent": intent
    })

# Clear chat button
if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.session_state.confidence_scores = []
    st.rerun()

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit • Designed to minimize AI hallucinations*")