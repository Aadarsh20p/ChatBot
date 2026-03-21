# Hallucination-Resistant AI Chatbot 🧭

A sophisticated, hallucination-resistant AI chatbot built with Streamlit and PyTorch. This project leverages the OpenChat-3.5 7B model locally, combining Retrieval-Augmented Generation (RAG) and Natural Language Inference (NLI) verification to ensure accurate, fact-based responses with multi-tier fallbacks and confidence scoring.

## Features ✨

- **Local LLM Inference**: Runs OpenChat-3.5 locally optimized for CPU footprint (float16, inference mode).
- **Multiple Response Modes**: 
  - ⚡ Fast Mode (Instant responses)
  - 🚀 Real OpenChat-3.5 (Authentic 7B inference)
  - 🔄 Hybrid Mode (Dynamic fallback)
  - 🏃 Speed Demo
- **Hallucination Resistance**: Utilizes RAG techniques and NLI fact-checking to minimize AI hallucinations.
- **Interactive UI**: Built with Streamlit for a clean, responsive chat interface.

## Project Structure 📁

- `main.py`: The Streamlit application and frontend UI.
- `utils.py`: Contains the `HallucinationResistantChatbot` class for model loading and optimized local inference.
- `models/`: Directory housing the locally downloaded OpenChat-3.5 weights and tokenizers.
- `requirements.txt`: Python dependencies.

## Installation 🛠️

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Aadarsh20p/ChatBot.git
   cd ChatBot
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/Mac:
   source .venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure Model Files are Present:**
   Ensure the `models/openchat_3.5` directory contains the model weights (`pytorch_model.bin`, `config.json`, etc.). 

## Running the App 🚀

Run the Streamlit application using:
```bash
streamlit run main.py
```
*(Or use the provided `run_fast.bat` script on Windows).*

## Technical Stack 🧰

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Machine Learning**: [PyTorch](https://pytorch.org/), [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- **Vector Search / RAG**: [FAISS](https://faiss.ai/), [Sentence-Transformers](https://sbert.net/)

## License 📄
This project is open-source and available under the MIT License.
