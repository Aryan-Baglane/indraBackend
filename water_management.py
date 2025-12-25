"""
Water Management AI - Production Ready for Rural India
Uses buckets/day, GIS rainfall data, and token-optimized prompts
"""

import os
import sys
import re
import time
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
import traceback

# API KEYS
QDRANT_URL = "https://50052f68-a3f2-4fce-91b2-9e140737db61.us-east4-0.gcp.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.5XzTwGgID_B55AVH-lIM9QYK2PEceMbMkcUMHWCgTDU"
OPENROUTER_API_KEY = "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"

# MODEL CONFIG
COLLECTION_NAME = "standrd_rag"
LLM_MODEL = "nvidia/nemotron-nano-12b-v2-vl:free"
TEMPERATURE = 0.3
MAX_TOKENS = 600  # Increased for better responses
MAX_RETRIES = 3  # Retry count for LLM calls
RETRY_DELAY = 1.0  # Seconds between retries

# DEPENDENCIES
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from gis_utils import gis_manager
except ImportError as e:
    print(f"Error: Missing libraries - {e}")
    sys.exit(1)


# MODELS
class WaterManagementRequest(BaseModel):
    location: Optional[str] = None
    pincode: Optional[str] = None
    season: str = "monsoon"
    crop_type: Optional[str] = None
    cattle_count: int = 10
    household_members: int = 4
    farm_size_acres: float = 2.0


class WaterDistribution(BaseModel):
    irrigation_buckets: int
    cattle_buckets: int
    drinking_buckets: int
    irrigation_pct: float
    cattle_pct: float
    drinking_pct: float


class WaterManagementResponse(BaseModel):
    distribution: WaterDistribution
    recommendations: List[str]
    ai_insights: str
    water_status: str
    gis_summary: str


# AI ENGINE
class WaterManagementAI:
    def __init__(self):
        self.embeddings = None
        self.vector_store = None
        self.retriever = None
        self.llm = None
        self.prompt = None
        self._initialized = False
    
    def initialize(self):
        if self._initialized:
            return
        
        try:
            print("Initializing Water Management AI...")
            gis_manager.load_data()
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_PATH,
                encode_kwargs={'normalize_embeddings': False}
            )
            
            client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            collections = client.get_collections().collections
            if not any(c.name == RAG_COLLECTION_NAME for c in collections):
                print(f"Warning: Collection '{RAG_COLLECTION_NAME}' not found")
                return
            
            self.vector_store = QdrantVectorStore.from_existing_collection(
                embedding=self.embeddings,
                collection_name=RAG_COLLECTION_NAME,
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY
            )
            
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": RAG_RETRIEVER_K})
            
            self.llm = ChatOpenAI(
                model=LLM_MODEL,
                openai_api_key=OPENROUTER_API_KEY,
                openai_api_base=OPENROUTER_BASE_URL,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS
            )
            
            # Improved prompt with specific GIS data reference
            prompt_template = """You are a water management advisor for rural India.

GIS DATA (use this for rainfall info):
{gis_data}

RAG CONTEXT:
{context}

FARM DETAILS:
- Location: {location}
- Season: {season}
- Crop: {crop_type}
- Cattle: {cattle_count}
- Family members: {household_members}

Provide 2 specific water management tips based on the GIS rainfall data shown above.
Keep response under 80 words. Be practical and actionable."""

            self.prompt = ChatPromptTemplate.from_template(prompt_template)
            self._initialized = True
            print("Water Management AI initialized")
            
        except Exception as e:
            print(f"Error initializing AI: {e}")
    
    def get_ai_insights(self, request: WaterManagementRequest) -> str:
        """
        Get AI insights using GIS + RAG + LLM.
        Uses pincode as primary identifier with smart fallbacks.
        """
        if not self._initialized:
            self.initialize()
        
        if not self._initialized or not self.prompt:
            raise ValueError("AI system not initialized. Please check Qdrant connection and model files.")
        
        # GIS data - REQUIRED (using pincode as primary)
        gis_data_str = None
        gis_data = None
        
        if request.pincode or request.location:
            location_parts = (request.location or "").split(",")
            district = location_parts[0].strip() if location_parts else None
            state = location_parts[1].strip() if len(location_parts) > 1 else None
            
            gis_data = gis_manager.get_location_data(
                pincode=request.pincode,
                district=district,
                state=state
            )
            
            if gis_data:
                rainfall = gis_data.get('rainfall', {})
                stress = gis_manager.get_water_stress_level(gis_data)
                match_note = gis_data.get('note', '')
                
                gis_data_str = f"{gis_data.get('district', 'Unknown')}, {rainfall.get('total_annual', 0):.0f}mm rain, {stress} stress"
                if match_note:
                    gis_data_str += f" ({match_note})"
        
        if not gis_data_str:
            raise ValueError(f"GIS data unavailable for pincode: {request.pincode}. Please verify your pincode is correct.")
        
        # RAG context - REQUIRED
        query = f"water management {request.season} {request.crop_type or 'farming'} rural india"
        docs = self.retriever.invoke(query)
        
        if not docs:
            raise ValueError("RAG retrieval failed. No relevant documents found in knowledge base.")
        
        context = " ".join(doc.page_content[:150] for doc in docs[:2])
        
        if not context.strip():
            raise ValueError("RAG context is empty. Knowledge base may need to be populated.")
        
        # Invoke LLM
        prompt_vars = {
            "context": context,
            "gis_data": gis_data_str,
            "location": gis_data.get('district', 'Rural') if gis_data else "Rural",
            "season": request.season,
            "crop_type": request.crop_type or "General",
            "cattle_count": request.cattle_count,
            "household_members": request.household_members
        }
        
        formatted_prompt = self.prompt.format(**prompt_vars)
        
        # Retry logic for LLM calls
        result = None
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self.llm.invoke(formatted_prompt)
                result = response.content if hasattr(response, 'content') else str(response)
                
                # Clean any emojis from response
                result = self._clean_text(result)
                
                if result and result.strip():
                    break
                    
                print(f"[WaterManagement] Empty response on attempt {attempt + 1}/{MAX_RETRIES}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
            except Exception as e:
                last_error = e
                print(f"[WaterManagement] LLM error on attempt {attempt + 1}: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
        
        if not result or not result.strip():
            error_msg = f"LLM failed after {MAX_RETRIES} attempts"
            if last_error:
                error_msg += f": {str(last_error)}"
            raise ValueError(error_msg)
        
        return result
    
    def _clean_text(self, text: str) -> str:
        """Remove emojis and clean text output"""
        if not text:
            return text
        # Remove emojis and special unicode characters
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        text = emoji_pattern.sub('', text)
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def calculate_optimal_distribution(self, request: WaterManagementRequest) -> Tuple[WaterDistribution, str, str]:
        """
        Calculate water distribution using GIS data.
        Uses pincode as primary identifier, falls back to nearest available data.
        """
        # REQUIRE at least pincode OR location
        if not request.pincode and not request.location:
            raise ValueError("Pincode or location is required for GIS-based water distribution analysis.")
        
        # Extract district/state from location string for fallback
        location_parts = (request.location or "").split(",")
        district = location_parts[0].strip() if location_parts else None
        state = location_parts[1].strip() if len(location_parts) > 1 else None
        
        # Fetch GIS data - prioritize pincode, with smart fallbacks
        gis_data = gis_manager.get_location_data(
            pincode=request.pincode,
            district=district,
            state=state
        )
        
        if not gis_data:
            error_msg = f"GIS data not found for pincode: {request.pincode}" if request.pincode else f"GIS data not found for location: {request.location}"
            raise ValueError(f"{error_msg}. Please verify your pincode is correct.")
        
        # Log if using fallback data
        match_type = gis_data.get('match_type', 'unknown')
        note = gis_data.get('note', '')
        if match_type != 'exact_pincode':
            print(f"[WaterManagement] Using {match_type} data: {note}")
        
        # Extract GIS data
        rainfall = gis_data.get('rainfall', {})
        gw = gis_data.get('groundwater', {})
        stress = gis_manager.get_water_stress_level(gis_data)
        
        annual_rain = rainfall.get('total_annual', 0)
        season_rain = rainfall.get(request.season.lower(), 0)
        extraction_pct = gw.get('extraction_percentage', 0)
        
        # Base needs (1 bucket ≈ 20L) - adjusted by GIS data
        drinking_base = request.household_members * 2
        cattle_base = request.cattle_count * 2
        irrigation_base = int(request.farm_size_acres * 10)
        
        # Season adjustments based on ACTUAL GIS rainfall data
        if request.season == "summer":
            # Adjust based on actual summer rainfall
            if season_rain < 30:
                drinking_base = int(drinking_base * 1.3)
                cattle_base = int(cattle_base * 1.4)
                irrigation_base = int(irrigation_base * 1.5)
            else:
                drinking_base = int(drinking_base * 1.2)
                cattle_base = int(cattle_base * 1.3)
                irrigation_base = int(irrigation_base * 1.4)
        elif request.season == "winter":
            irrigation_base = int(irrigation_base * 0.7)
            cattle_base = int(cattle_base * 0.9)
        elif request.season == "monsoon":
            # Adjust based on actual monsoon rainfall
            if season_rain > 300:
                irrigation_base = int(irrigation_base * 0.3)  # Heavy rain, less irrigation
            else:
                irrigation_base = int(irrigation_base * 0.5)
        
        # Further adjust based on groundwater stress
        if stress == "Over-Exploited" or extraction_pct > 100:
            # Critical water shortage - reduce all usage
            drinking_base = int(drinking_base * 0.9)
            cattle_base = int(cattle_base * 0.8)
            irrigation_base = int(irrigation_base * 0.6)
        elif stress == "Critical" or extraction_pct > 90:
            irrigation_base = int(irrigation_base * 0.7)
        
        total_buckets = max(drinking_base + cattle_base + irrigation_base, 1)  # Avoid division by zero
        
        # Percentages
        irrigation_pct = round((irrigation_base / total_buckets) * 100, 1)
        cattle_pct = round((cattle_base / total_buckets) * 100, 1)
        drinking_pct = round((drinking_base / total_buckets) * 100, 1)
        
        # Water status based on GIS data
        if stress == "Over-Exploited" or extraction_pct > 100:
            water_status = "Critical - Water Outsourcing Needed"
        elif stress == "Critical" or (request.season == "summer" and season_rain < 30):
            water_status = "Moderate - Strict Conservation Required"
        elif request.season == "monsoon" and season_rain > 300:
            water_status = "Surplus - Rainwater Harvesting Recommended"
        elif stress == "Safe" and annual_rain > 800:
            water_status = "Sufficient - Manage Wisely"
        else:
            water_status = "Normal - Conservation Advised"
        
        gis_summary = f"{gis_data.get('district', 'Unknown')}, {stress} stress level, {annual_rain:.0f}mm annual rainfall, {extraction_pct:.1f}% groundwater extraction"
        
        distribution = WaterDistribution(
            irrigation_buckets=irrigation_base,
            cattle_buckets=cattle_base,
            drinking_buckets=drinking_base,
            irrigation_pct=irrigation_pct,
            cattle_pct=cattle_pct,
            drinking_pct=drinking_pct
        )
        
        return distribution, water_status, gis_summary
    
    def generate_recommendations_with_rag(self, request: WaterManagementRequest, water_status: str, gis_summary: str) -> List[str]:
        """
        Generate recommendations using RAG + LLM based on GIS data.
        Raises ValueError if RAG/LLM fails.
        """
        if not self._initialized:
            self.initialize()
        
        if not self._initialized or not self.retriever:
            raise ValueError("RAG system not initialized. Cannot generate recommendations.")
        
        # Build context-rich query
        query = f"water management recommendations {request.season} {request.crop_type or 'farming'} {water_status} rural india agriculture"
        
        # Get RAG documents
        docs = self.retriever.invoke(query)
        if not docs:
            raise ValueError("Failed to retrieve relevant documents from knowledge base.")
        
        context = " ".join(doc.page_content[:200] for doc in docs[:2])
        
        if not context.strip():
            raise ValueError("RAG context is empty. Knowledge base may need data.")
        
        # Generate recommendations via LLM with retry logic
        rec_prompt = f"""Based on the following context and GIS data, provide 3 specific actionable water management recommendations.

Context: {context}

GIS Data: {gis_summary}
Water Status: {water_status}
Season: {request.season}
Crop: {request.crop_type or 'General'}
Cattle Count: {request.cattle_count}
Family Members: {request.household_members}
Farm Size: {request.farm_size_acres} acres

Provide exactly 3 recommendations, one per line. Each should be specific and actionable. No numbering, bullets or emojis."""
        
        result = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self.llm.invoke(rec_prompt)
                result = response.content if hasattr(response, 'content') else str(response)
                result = self._clean_text(result)  # Remove emojis
                if result and result.strip():
                    break
                print(f"[WaterManagement] Empty recommendations on attempt {attempt + 1}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
            except Exception as e:
                print(f"[WaterManagement] Recommendation error attempt {attempt + 1}: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
        
        if not result or not result.strip():
            raise ValueError("LLM failed to generate recommendations after multiple attempts.")
        
        # Parse recommendations (split by newlines, filter empty, clean)
        recs = []
        for line in result.strip().split('\n'):
            line = line.strip()
            # Remove numbering like "1.", "1)", "-", "*"
            line = re.sub(r'^[\d]+[.)]\s*', '', line)
            line = re.sub(r'^[-*]\s*', '', line)
            if line:
                recs.append(line)
        
        if not recs:
            raise ValueError("LLM returned empty recommendations.")
        
        return recs[:3]  # Return top 3


# SINGLETON
water_ai = WaterManagementAI()


# API FUNCTION
async def predict_water_distribution(request: WaterManagementRequest) -> WaterManagementResponse:
    """
    AI-powered water distribution prediction using GIS + RAG + LLM.
    Raises HTTPException with specific error messages - NO hardcoded fallbacks.
    """
    try:
        # Initialize AI system
        if not water_ai._initialized:
            water_ai.initialize()
        
        if not water_ai._initialized:
            raise ValueError("Water Management AI failed to initialize. Check Qdrant connection and model files.")
        
        # Step 1: Calculate distribution using GIS data (REQUIRED)
        distribution, water_status, gis_summary = water_ai.calculate_optimal_distribution(request)
        
        # Step 2: Generate AI insights using GIS + RAG + LLM (REQUIRED)
        ai_insights = water_ai.get_ai_insights(request)
        
        # Step 3: Generate recommendations using RAG + LLM (REQUIRED)
        recommendations = water_ai.generate_recommendations_with_rag(request, water_status, gis_summary)
        
        return WaterManagementResponse(
            distribution=distribution,
            recommendations=recommendations,
            ai_insights=ai_insights,
            water_status=water_status,
            gis_summary=gis_summary
        )
        
    except ValueError as ve:
        # Re-raise ValueError with original message (user-friendly errors)
        print(f"[WaterManagement] Validation error: {ve}")
        raise ve
    except Exception as e:
        # Log unexpected errors but don't expose internal details
        print(f"[WaterManagement] Unexpected error: {e}")
        traceback.print_exc()
        raise ValueError(f"Water management analysis failed: {str(e)}. Please try again or contact support.")


async def get_water_tips(season: str, crop_type: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Get water tips using RAG + LLM - NO hardcoded tips.
    """
    try:
        if not water_ai._initialized:
            water_ai.initialize()
        
        if not water_ai._initialized or not water_ai.retriever:
            raise ValueError("Water Management AI not initialized. Cannot fetch tips.")
        
        # Build query for RAG
        query = f"water conservation tips {season} {crop_type or 'general'} rural india agriculture"
        
        # Get relevant documents
        docs = water_ai.retriever.invoke(query)
        if not docs:
            raise ValueError(f"No tips found for {season} season. Knowledge base may need more data.")
        
        context = " ".join(doc.page_content[:200] for doc in docs[:2])
        
        # Generate tips via LLM
        tips_prompt = f"""Based on this context, provide 3 practical water conservation tips for {season} season{f' and {crop_type} crop' if crop_type else ''}.

Context: {context}

Provide exactly 3 tips, one per line. Keep each tip under 10 words. No numbering."""
        
        response = water_ai.llm.invoke(tips_prompt)
        result = response.content if hasattr(response, 'content') else str(response)
        
        if not result or not result.strip():
            raise ValueError("Failed to generate water tips. LLM returned empty response.")
        
        tips = [line.strip() for line in result.strip().split('\n') if line.strip()]
        
        if not tips:
            raise ValueError("Failed to parse water tips from LLM response.")
        
        return {"tips": tips[:3]}
        
    except ValueError as ve:
        raise ve
    except Exception as e:
        print(f"[WaterTips] Error: {e}")
        traceback.print_exc()
        raise ValueError(f"Failed to fetch water tips: {str(e)}")
