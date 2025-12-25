"""
Crop Suggestion System with AI-Powered Recommendations
Production-ready RAG + GIS + LLM architecture for intelligent crop selection
Uses OpenRouter with Llama model for water-efficient farming recommendations
"""

import os
import sys
import re
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import json

# API KEYS AND CONFIGURATION
QDRANT_URL = "https://50052f68-a3f2-4fce-91b2-9e140737db61.us-east4-0.gcp.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.5XzTwGgID_B55AVH-lIM9QYK2PEceMbMkcUMHWCgTDU"
OPENROUTER_API_KEY = "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"

# MODEL CONFIGURATION
COLLECTION_NAME = "standrd_rag"
LLM_MODEL = "nvidia/nemotron-nano-12b-v2-vl:free"
TEMPERATURE = 0.3
MAX_TOKENS = 2500  # Increased for complete JSON
MAX_RETRIES = 3
RETRY_DELAY = 1.0

# DEPENDENCIES
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from gis_utils import gis_manager
except ImportError as e:
    print("Error: Missing required libraries")
    print("Install: pip install langchain-huggingface langchain-qdrant qdrant-client langchain-openai pandas")
    sys.exit(1)


# PYDANTIC MODELS
class CropInput(BaseModel):
    """Input parameters for crop suggestion"""
    location: str = Field(..., description="Location/district of the farm")
    pincode: Optional[str] = Field(None, description="Area pincode for GIS data")
    soil_type: str = Field(..., description="Type of soil")
    season: str = Field(..., description="Planting season (Kharif/Rabi/Zaid)")
    water_availability: str = Field(..., description="Water availability (Low/Medium/High)")
    farm_size_acres: float = Field(..., description="Farm size in acres")
    rainfall_mm: Optional[float] = Field(None, description="Annual rainfall in mm")


class CropRecommendation(BaseModel):
    """Individual crop recommendation"""
    crop_name: str
    water_requirement_liters: float
    estimated_market_price_per_kg: float
    yield_per_acre_kg: float
    total_profit_estimate: float
    price_per_liter_ratio: float  # Price efficiency
    environmental_impact_score: int  # 1-10 (10 = best)
    soil_health_impact: str  # Positive/Neutral/Negative
    farmer_ease_score: int  # 1-10 (10 = easiest)
    rank: int
    justification: str


class CropSuggestionResponse(BaseModel):
    """Complete crop suggestion response"""
    recommendations: List[CropRecommendation]
    season_context: str
    water_context: str
    general_advice: str


# AI SYSTEM PROMPT - Improved with explicit GIS data usage
CROP_SUGGESTION_PROMPT = """You are a crop advisor for Indian farmers. Use the GIS data provided below for accurate local information.

IMPORTANT: Use the GIS rainfall data exactly as given. Do NOT make up rainfall values.

GIS DATA (VERIFIED LOCAL DATA):
{gis_data}

FARM INPUTS:
- Location: {location}
- Soil Type: {soil_type}
- Season: {season}
- Water Availability: {water_availability}
- Farm Size: {farm_size_acres} acres

RAG CONTEXT:
{context}

TASK: Suggest 5 crops ranked by price/water efficiency ratio.

STRICT RULES:
1. Output ONLY valid JSON - no markdown, no extra text
2. Use the EXACT rainfall data from GIS DATA above
3. water_requirement_liters is per acre for the crop cycle
4. Keep justification under 15 words, no emojis
5. Use realistic Indian market prices (2024)
6. All numeric values must be integers or simple decimals

REQUIRED JSON FORMAT:
{{
  "recommendations": [
    {{
      "crop_name": "Wheat",
      "water_requirement_liters": 450000,
      "estimated_market_price_per_kg": 25,
      "yield_per_acre_kg": 2500,
      "total_profit_estimate": 45000,
      "price_per_liter_ratio": 0.139,
      "environmental_impact_score": 8,
      "soil_health_impact": "Neutral",
      "farmer_ease_score": 9,
      "rank": 1,
      "justification": "Low water need good profit margin"
    }}
  ],
  "season_context": "Brief season tip",
  "water_context": "Brief water tip",
  "general_advice": "Brief overall advice"
}}"""


# INITIALIZATION
embeddings = None
vector_store = None
retriever = None
llm = None
rag_chain = None


def initialize_crop_system():
    """Initialize crop suggestion system with RAG + GIS"""
    global embeddings, vector_store, retriever, llm, rag_chain
    
    print("Initializing INDRA Crop Suggestion System...")
    
    # Load GIS data
    gis_manager.load_data()
    
    # Load embedding model
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        encode_kwargs={'normalize_embeddings': False}
    )
    
    # Connect to Qdrant
    print("Connecting to Qdrant...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    collections = client.get_collections().collections
    if not any(c.name == RAG_COLLECTION_NAME for c in collections):
        print(f"Warning: Collection '{RAG_COLLECTION_NAME}' not found")
        print("System will work with limited context")
    
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=RAG_COLLECTION_NAME,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": RAG_RETRIEVER_K})
    
    # Initialize OpenRouter LLM
    print("Connecting to OpenRouter...")
    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS
    )
    
    # Test LLM connection
    try:
        test_response = llm.invoke("Say 'OK'")
        print(f"LLM test successful: {test_response.content[:50]}")
    except Exception as e:
        print(f"WARNING: LLM test failed: {e}")
    
    # Create RAG chain
    prompt = ChatPromptTemplate.from_template(CROP_SUGGESTION_PROMPT)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def format_input(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Format inputs with GIS data - pincode is primary identifier"""
        # Get GIS data using pincode primarily
        gis_data_str = "GIS data not available"
        location = inputs.get('location', '')
        pincode = inputs.get('pincode')
        
        if pincode or location:
            location_parts = location.split(",")
            district = location_parts[0].strip() if location_parts else None
            state = location_parts[1].strip() if len(location_parts) > 1 else None
            
            gis_data = gis_manager.get_location_data(
                pincode=pincode,
                district=district,
                state=state
            )
            
            if gis_data:
                rainfall_info = gis_data.get('rainfall', {})
                match_note = gis_data.get('note', '')
                gis_data_str = f"""District: {gis_data.get('district', 'Unknown')}
State: {gis_data.get('state', 'Unknown')}
Annual Rainfall: {rainfall_info.get('total_annual', 0):.0f} mm
Monsoon Rainfall: {rainfall_info.get('monsoon', 0):.0f} mm
Summer Rainfall: {rainfall_info.get('summer', 0):.0f} mm
Groundwater Stress: {gis_manager.get_water_stress_level(gis_data)}"""
                if match_note:
                    gis_data_str += f"\nNote: {match_note}"
        
        # Retrieve context from RAG (limit to 2 docs, first 300 chars each)
        query = f"crop {inputs.get('season', '')} {inputs.get('soil_type', '')}"
        docs = retriever.invoke(query)
        context = " ".join(doc.page_content[:300] for doc in docs[:2])  # Limit tokens
        
        return {
            "context": context,
            "gis_data": gis_data_str,
            "location": inputs.get("location", ""),
            "soil_type": inputs.get("soil_type", ""),
            "season": inputs.get("season", ""),
            "water_availability": inputs.get("water_availability", ""),
            "farm_size_acres": inputs.get("farm_size_acres", ""),
            "rainfall_mm": inputs.get("rainfall_mm", "N/A")
        }
    
    rag_chain = (
        format_input
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("Crop Suggestion System ready")


def clean_text(text: str) -> str:
    """Remove emojis and special characters from text"""
    if not text:
        return text
    # Remove emojis
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub('', text)


def clean_json_response(response: str) -> str:
    """
    Robust JSON extraction and cleaning from LLM response.
    Handles truncated JSON, markdown blocks, and malformed output.
    """
    if not response:
        return ""
    
    # Remove emojis first
    cleaned = clean_text(response.strip())
    
    # Remove markdown code blocks
    if "```json" in cleaned:
        parts = cleaned.split("```json")
        if len(parts) > 1:
            cleaned = parts[1].split("```")[0]
    elif "```" in cleaned:
        parts = cleaned.split("```")
        if len(parts) > 1:
            cleaned = parts[1].split("```")[0] if len(parts) > 1 else parts[0]
    
    # Find JSON object boundaries
    start = cleaned.find('{')
    if start == -1:
        return ""
    
    # Find matching closing brace (handle nested objects)
    brace_count = 0
    end = start
    for i, char in enumerate(cleaned[start:], start):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                end = i
                break
    
    if end <= start:
        # Incomplete JSON - try to find last closing brace
        end = cleaned.rfind('}')
    
    if end <= start:
        return ""
    
    json_str = cleaned[start:end+1]
    
    # Fix common JSON issues
    # Remove trailing commas before closing braces/brackets
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
    # Fix unquoted string values (basic)
    json_str = re.sub(r':\s*([A-Za-z][A-Za-z0-9_]*)\s*([,}\]])', r': "\1"\2', json_str)
    # Ensure proper array closing
    json_str = re.sub(r'\[\s*{[^}]+$', lambda m: m.group(0) + '}]', json_str)
    
    return json_str


def try_parse_json(json_str: str) -> Optional[Dict[str, Any]]:
    """
    Try to parse JSON with multiple strategies.
    """
    if not json_str:
        return None
    
    # Strategy 1: Direct parse
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Fix truncated recommendations array
    try:
        # If JSON is truncated mid-recommendations, try to close it properly
        if '"recommendations": [' in json_str and not json_str.rstrip().endswith('}'):
            # Find last complete recommendation object
            last_complete = json_str.rfind('},')
            if last_complete > 0:
                json_str = json_str[:last_complete+1]
                # Close arrays and objects
                json_str += '], "season_context": "Based on selected season", "water_context": "Optimized for water efficiency", "general_advice": "Choose crops suited to local conditions"}'
                return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Extract what we can and build minimal valid JSON
    try:
        # Try to extract at least one recommendation
        rec_match = re.search(r'\{[^{}]*"crop_name"[^{}]*\}', json_str)
        if rec_match:
            rec = json.loads(rec_match.group())
            return {
                "recommendations": [rec],
                "season_context": "Based on selected season",
                "water_context": "Optimized for water efficiency",
                "general_advice": "Partial data recovered from AI response"
            }
    except:
        pass
    
    return None


async def get_crop_suggestions(crop_input: CropInput) -> CropSuggestionResponse:
    """Get AI-powered crop suggestions with RAG + GIS + LLM"""
    global rag_chain
    
    # Initialize if needed
    if rag_chain is None:
        initialize_crop_system()
    
    # Prepare input
    input_dict = {
        "location": crop_input.location,
        "pincode": crop_input.pincode,
        "soil_type": crop_input.soil_type,
        "season": crop_input.season,
        "water_availability": crop_input.water_availability,
        "farm_size_acres": crop_input.farm_size_acres,
        "rainfall_mm": crop_input.rainfall_mm if crop_input.rainfall_mm else "N/A"
    }
    
    # Retry logic for LLM calls
    last_error = None
    result = None
    
    for attempt in range(MAX_RETRIES):
        try:
            print(f"[CropSuggestion] Attempt {attempt + 1}/{MAX_RETRIES} for {crop_input.location}...")
            response = rag_chain.invoke(input_dict)
            
            # Check for empty response
            if not response or len(response.strip()) == 0:
                print(f"[CropSuggestion] Empty response on attempt {attempt + 1}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                continue
            
            print(f"[CropSuggestion] Response length: {len(response)} chars")
            
            # Clean and extract JSON
            cleaned_json = clean_json_response(response)
            
            if not cleaned_json:
                print(f"[CropSuggestion] No JSON found in response")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                continue
            
            print(f"[CropSuggestion] Cleaned JSON length: {len(cleaned_json)} chars")
            
            # Try to parse JSON with multiple strategies
            result = try_parse_json(cleaned_json)
            
            if result and 'recommendations' in result and len(result['recommendations']) > 0:
                # Clean text fields
                for rec in result['recommendations']:
                    if 'justification' in rec:
                        rec['justification'] = clean_text(rec['justification'])
                if 'season_context' in result:
                    result['season_context'] = clean_text(result['season_context'])
                if 'water_context' in result:
                    result['water_context'] = clean_text(result['water_context'])
                if 'general_advice' in result:
                    result['general_advice'] = clean_text(result['general_advice'])
                
                print(f"[CropSuggestion] Successfully parsed {len(result['recommendations'])} recommendations")
                break
            else:
                print(f"[CropSuggestion] Invalid JSON structure on attempt {attempt + 1}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    
        except Exception as e:
            last_error = e
            print(f"[CropSuggestion] Error on attempt {attempt + 1}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    
    # If we got valid results, return them
    if result and 'recommendations' in result:
        try:
            return CropSuggestionResponse(**result)
        except Exception as e:
            print(f"[CropSuggestion] Validation error: {e}")
    
    # Fallback with clear indication
    print(f"[CropSuggestion] Using fallback after {MAX_RETRIES} attempts")
    return CropSuggestionResponse(
        recommendations=[
            CropRecommendation(
                crop_name="Wheat",
                water_requirement_liters=450000,
                estimated_market_price_per_kg=25,
                yield_per_acre_kg=2500,
                total_profit_estimate=40000,
                price_per_liter_ratio=0.089,
                environmental_impact_score=8,
                soil_health_impact="Positive",
                farmer_ease_score=9,
                rank=1,
                justification="Low water requirement, good market demand"
            ),
            CropRecommendation(
                crop_name="Lentils (Masoor)",
                water_requirement_liters=300000,
                estimated_market_price_per_kg=80,
                yield_per_acre_kg=800,
                total_profit_estimate=44000,
                price_per_liter_ratio=0.213,
                environmental_impact_score=9,
                soil_health_impact="Positive",
                farmer_ease_score=7,
                rank=2,
                justification="Nitrogen fixing, improves soil health"
            )
        ],
        season_context=f"Recommendations for {crop_input.season} season",
        water_context=f"Based on {crop_input.water_availability} water availability",
        general_advice="AI service experienced issues. Showing reliable default crops for your region."
    )

