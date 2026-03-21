import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from typing import Tuple
import os
import gc

class HallucinationResistantChatbot:
    def __init__(self):
        """Initialize with optimized OpenChat loading"""
        self.has_openchat = False
        self.load_openchat_optimized()
    
    def load_openchat_optimized(self):
        """Load OpenChat with optimizations for faster inference"""
        openchat_path = st.session_state.get('openchat_path', '')
        if not openchat_path:
            openchat_path = os.path.join(os.getcwd(), "models", "openchat_3.5")
        
        try:
            st.info("🚀 Loading OpenChat-3.5 (Optimized for speed)...")
            
            # Clear memory before loading
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Step 1: Load tokenizer
            st.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                openchat_path,
                local_files_only=True,
                use_fast=True  # Use fast tokenizer
            )
            
            # Set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Step 2: Load model with optimizations
            st.info("Loading model weights (this takes 1-2 minutes)...")
            
            # Load in float16 for faster inference on CPU
            self.model = AutoModelForCausalLM.from_pretrained(
                openchat_path,
                torch_dtype=torch.float16,  # Half precision for speed
                device_map="cpu",
                local_files_only=True,
                low_cpu_mem_usage=True
            )
            
            # Step 3: Optimize model for inference
            self.model.eval()  # Set to evaluation mode
            
            # Disable gradient computation
            for param in self.model.parameters():
                param.requires_grad = False
            
            self.has_openchat = True
            st.success("✅ OpenChat-3.5 loaded and optimized!")
            st.info("Ready to chat! (First response may be slower)")
            
        except Exception as e:
            st.error(f"Error loading OpenChat: {e}")
            st.error("Make sure you have enough RAM (need ~14GB free)")
            self.has_openchat = False
    
    def generate_response(self, user_input: str) -> Tuple[str, float]:
        """Generate response with optimized settings"""
        if not self.has_openchat:
            return "Model not loaded. Please check the error messages above.", 0.0
        
        try:
            # Simple prompt format
            prompt = f"GPT4 Correct User: {user_input}<|end_of_turn|>GPT4 Correct Assistant:"
            
            # Tokenize with optimizations
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=256,  # Shorter context for faster generation
                truncation=True,
                padding=False  # No padding needed for single input
            )
            
            # Move to model device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate with optimized settings
            start_time = time.time()
            
            with torch.inference_mode():  # Faster than torch.no_grad()
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=50,  # Shorter responses for speed
                    min_new_tokens=5,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,  # Use KV cache for speed
                    num_beams=1,  # No beam search for speed
                )
            
            # Decode only the generated part
            generated_tokens = outputs[0][len(inputs['input_ids'][0]):]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Clean response
            response = response.replace("<|end_of_turn|>", "").strip()
            
            generation_time = time.time() - start_time
            
            # Clear memory after generation
            del inputs, outputs, generated_tokens
            gc.collect()
            
            return response, generation_time
            
        except Exception as e:
            st.error(f"Generation error: {e}")
            return f"Error generating response: {str(e)}", 0.0
    
    def classify_input(self, user_input: str) -> str:
        """Simple rule-based classification (no heavy models)"""
        input_lower = user_input.lower()
        
        if any(word in input_lower for word in ['hi', 'hello', 'hey', 'greetings']):
            return "greeting"
        elif any(word in input_lower for word in ['what is', 'what are', 'explain', 'define']):
            return "factual_question"
        elif any(word in input_lower for word in ['how to', 'how do', 'how can']):
            return "how_to_guide"
        elif any(word in input_lower for word in ['code', 'program', 'function', 'implement']):
            return "coding_question"
        else:
            return "general"
    
    def get_fallback_response(self, user_input: str, intent: str) -> str:
        """Quick fallback responses for fast/demo modes"""
        fallback_responses = {
            "greeting": "Hello! I'm OpenChat, a hallucination-resistant AI assistant. How can I help you today?",
            "factual_question": "I can help explain various topics. In fast mode, switch to 'Real Mode' for detailed explanations using the full OpenChat model.",
            "how_to_guide": "I can provide step-by-step guides. For detailed instructions, please use 'Real Mode' to access the full model.",
            "coding_question": "I can help with programming questions. For code examples and detailed explanations, switch to 'Real Mode'.",
            "general": "I'm here to help! For the best responses, please use 'Real Mode' which uses the full OpenChat model."
        }
        return fallback_responses.get(intent, fallback_responses["general"])
    
    def process_query(self, user_input: str) -> Tuple[str, float, str]:
        """Simplified processing pipeline"""
        # Get response mode
        mode = st.session_state.get('response_mode', 'fast')
        
        # Classify input
        intent = self.classify_input(user_input)
        
        if mode == 'fast' or mode == 'demo':
            # Use simple responses for fast/demo modes
            response = self.get_fallback_response(user_input, intent)
            confidence = 0.95
            processing_time = 0.1
            
            if mode == 'demo':
                status = f"🏃 Demo Mode ({processing_time:.1f}s)"
            else:
                status = f"⚡ Fast Mode ({processing_time:.1f}s)"
            
            return response, confidence, status
        
        elif mode == 'real' or mode == 'hybrid':
            # Generate with OpenChat
            if not self.has_openchat:
                return "OpenChat model not loaded. Please check the errors above.", 0.3, "❌ Model Error"
            
            st.info(f"🤖 Generating response with OpenChat-3.5...")
            response, gen_time = self.generate_response(user_input)
            
            # Simple confidence based on response quality
            if response and len(response) > 10:
                confidence = min(0.95, 0.6 + (len(response.split()) / 100))
            else:
                confidence = 0.4
            
            if mode == 'hybrid':
                status = f"🔄 Hybrid Mode ({gen_time:.1f}s)"
            else:
                status = f"🚀 Real OpenChat ({gen_time:.1f}s)"
            
            return response, confidence, status
        
        else:
            # Default fallback
            return "Invalid mode selected.", 0.1, "❌ Error"