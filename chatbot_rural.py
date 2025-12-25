"""
Rural/Gramin Chatbot for INDRA
Focuses on rural water management, crop selection, and farming practices
"""
import os
import sys
from typing import Optional
from pydantic import BaseModel

# Import config
from config import (
    QDRANT_URL, QDRANT_API_KEY, OPENROUTER_API_KEY,
    RAG_COLLECTION_NAME, LLM_MODEL, LLM_TEMPERATURE,
    EMBEDDING_MODEL_PATH, OPENROUTER_BASE_URL, RAG_RETRIEVER_K
)

# --- 2. IMPORTS ---
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
except ImportError as e:
    print(f"Error: Missing libraries - {e}")
    sys.exit(1)

# Pydantic Models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

RURAL_PROMPT = """You are INDRA AI, expert in rural water management and farming.

Context: {context}

Farmer: {question}

Guidelines:
- Rural/farm focus
- Simple Hindi-friendly language
- Crop selection, water saving
- Rainwater harvesting for farms
- Max 60 words

Answer:"""

# Chatbot Engine
class RuralChatbot:
    def __init__(self):
        self.embeddings = None
        self.retriever = None
        self.llm = None
        self.chain = None
        self._initialized = False
    
    def initialize(self):
        if self._initialized:
            return
        
        try:
            print("Initializing Rural Chatbot...")
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_PATH,
                encode_kwargs={'normalize_embeddings': False}
            )
            
            client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            collections = client.get_collections().collections
            
            if not any(c.name == RAG_COLLECTION_NAME for c in collections):
                print(f"Warning: Collection '{RAG_COLLECTION_NAME}' not found")
                return
            
            vector_store = QdrantVectorStore.from_existing_collection(
                embedding=self.embeddings,
                collection_name=RAG_COLLECTION_NAME,
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY
            )
            
            self.retriever = vector_store.as_retriever(search_kwargs={"k": RAG_RETRIEVER_K})
            
            self.llm = ChatOpenAI(
                model=LLM_MODEL,
                openai_api_key=OPENROUTER_API_KEY,
                openai_api_base=OPENROUTER_BASE_URL,
                temperature=LLM_TEMPERATURE,
                max_tokens=300
            )
            
            prompt = ChatPromptTemplate.from_template(RURAL_PROMPT)
            
            def format_docs(docs):
                return " ".join(doc.page_content[:200] for doc in docs)
            
            self.chain = (
                {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )
            
            self._initialized = True
            print("Rural Chatbot ready")
            
        except Exception as e:
            print(f"Error initializing Rural Chatbot: {e}")
    
    async def get_response(self, message: str) -> str:
        if not self._initialized:
            self.initialize()
        
        if not self._initialized or not self.chain:
            return "Main abhi uplabdh nahi hoon. Kripya baad mein prayas karein."
        
        try:
            response = self.chain.invoke(message)
            return response
        except Exception as e:
            print(f"Error in Rural Chatbot: {e}")
            return "Mujhe samajhne mein dikkat hui. Kripya apna sawal dobara puchein."

# Singleton
rural_chatbot = RuralChatbot()

async def chat_rural(request: ChatRequest) -> ChatResponse:
    """API function for rural chatbot"""
    if not rural_chatbot._initialized:
        rural_chatbot.initialize()
    
    response = await rural_chatbot.get_response(request.message)
    return ChatResponse(response=response)