"""
INDRA AI Service - Unified RAG + LLM Service for Fact-Checking and Dynamic Content
All modules use this service for AI-backed, dynamic responses instead of hardcoded values
"""

import os
import re
import json
import time
import aiohttp
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# Import config
from config import (
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL,
    LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    QDRANT_URL, QDRANT_API_KEY, RAG_COLLECTION_NAME,
    EMBEDDING_MODEL_PATH, MAX_RETRIES, RETRY_DELAY
)

# Dependencies
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from langchain_openai import ChatOpenAI
    RAG_AVAILABLE = True
except ImportError as e:
    RAG_AVAILABLE = False
    print(f"[AI Service] RAG libraries not available: {e}")


@dataclass
class FactCheckResult:
    """Result of fact-checking operation"""
    verified_data: Dict[str, Any]
    confidence: float  # 0-1
    source: str  # "rag", "llm", "gis", "combined"
    corrections: List[str]  # List of corrections made


class IndraAIService:
    """
    Unified AI Service for INDRA
    - RAG-based fact checking
    - LLM-powered dynamic content generation
    - No hardcoded responses
    """
    
    def __init__(self):
        self.embeddings = None
        self.vector_store = None
        self.retriever = None
        self.llm = None
        self._initialized = False
    
    def initialize(self):
        """Initialize AI service with RAG and LLM"""
        if self._initialized:
            return True
        
        if not RAG_AVAILABLE:
            print("[AI Service] RAG libraries not available")
            return False
        
        try:
            print("[AI Service] Initializing...")
            
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_PATH,
                encode_kwargs={'normalize_embeddings': False}
            )
            
            # Connect to Qdrant
            client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            collections = client.get_collections().collections
            
            if not any(c.name == RAG_COLLECTION_NAME for c in collections):
                print(f"[AI Service] Collection '{RAG_COLLECTION_NAME}' not found")
                return False
            
            self.vector_store = QdrantVectorStore.from_existing_collection(
                embedding=self.embeddings,
                collection_name=RAG_COLLECTION_NAME,
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY
            )
            
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
            
            # Initialize LLM
            self.llm = ChatOpenAI(
                model=LLM_MODEL,
                openai_api_key=OPENROUTER_API_KEY,
                openai_api_base=OPENROUTER_BASE_URL,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS
            )
            
            self._initialized = True
            print("[AI Service] Ready")
            return True
            
        except Exception as e:
            print(f"[AI Service] Initialization error: {e}")
            return False
    
    def get_rag_context(self, query: str, k: int = 3) -> str:
        """Get relevant context from RAG"""
        if not self._initialized or not self.retriever:
            return ""
        
        try:
            docs = self.retriever.invoke(query)
            if docs:
                return "\n\n".join([doc.page_content[:600] for doc in docs[:k]])
            return ""
        except Exception as e:
            print(f"[AI Service] RAG error: {e}")
            return ""
    
    async def fact_check_location_data(self, state: str, district: str, 
                                        gis_data: Dict[str, Any]) -> FactCheckResult:
        """
        Fact-check GIS data using RAG knowledge base
        Returns verified/corrected data with confidence score
        """
        if not self._initialized:
            self.initialize()
        
        # Get RAG context for this location
        query = f"rainfall climate weather {state} {district} India annual monsoon groundwater"
        rag_context = self.get_rag_context(query, k=4)
        
        if not rag_context:
            # No RAG data available, return GIS data with low confidence
            return FactCheckResult(
                verified_data=gis_data,
                confidence=0.3,
                source="gis_only",
                corrections=["Unable to verify with knowledge base"]
            )
        
        # Use LLM to cross-reference and verify
        gis_rainfall = gis_data.get('rainfall', {}).get('total_annual', 0)
        gis_monsoon = gis_data.get('rainfall', {}).get('monsoon', 0)
        gis_gw = gis_data.get('groundwater', {}).get('extraction_percentage', 0)
        
        verification_prompt = f"""You are a fact-checker for Indian geographic data. Cross-reference the GIS data with the knowledge base.

LOCATION: {district}, {state}

GIS DATA (to verify):
- Annual Rainfall: {gis_rainfall} mm
- Monsoon Rainfall: {gis_monsoon} mm
- Groundwater Extraction: {gis_gw}%

KNOWLEDGE BASE CONTEXT:
{rag_context}

TASK: Verify the GIS data accuracy. If the knowledge base suggests different values, provide corrections.

Return ONLY a JSON object with this exact format:
{{
    "annual_rainfall_mm": <verified number>,
    "monsoon_rainfall_mm": <verified number>,
    "groundwater_extraction_pct": <verified number>,
    "confidence": <0.0-1.0>,
    "corrections": ["list of corrections if any"],
    "notes": "brief verification note"
}}

If GIS data seems accurate based on knowledge base, return same values with high confidence.
If knowledge base has different data, use knowledge base values with corrections noted.
Return ONLY valid JSON, no other text."""

        try:
            response = await self._async_llm_call(verification_prompt)
            result = self._parse_json_response(response)
            
            if result:
                verified_data = {
                    'rainfall': {
                        'total_annual': result.get('annual_rainfall_mm', gis_rainfall),
                        'monsoon': result.get('monsoon_rainfall_mm', gis_monsoon)
                    },
                    'groundwater': {
                        'extraction_percentage': result.get('groundwater_extraction_pct', gis_gw)
                    }
                }
                
                return FactCheckResult(
                    verified_data=verified_data,
                    confidence=result.get('confidence', 0.7),
                    source="rag_verified",
                    corrections=result.get('corrections', [])
                )
        except Exception as e:
            print(f"[AI Service] Fact check error: {e}")
        
        # Fallback: return GIS data with medium confidence
        return FactCheckResult(
            verified_data=gis_data,
            confidence=0.5,
            source="gis_fallback",
            corrections=["Verification inconclusive"]
        )
    
    async def generate_dynamic_content(self, content_type: str, 
                                        context: Dict[str, Any],
                                        rag_query: str = None) -> Dict[str, Any]:
        """
        Generate dynamic content using RAG + LLM
        No hardcoded responses - everything is AI-generated
        
        content_type: "implementation_plan", "recommendations", "cost_analysis", 
                     "water_distribution", "crop_suggestions", "feasibility"
        """
        if not self._initialized:
            self.initialize()
        
        # Get relevant RAG context
        rag_context = ""
        if rag_query:
            rag_context = self.get_rag_context(rag_query, k=3)
        
        # Build prompt based on content type
        prompt = self._build_dynamic_prompt(content_type, context, rag_context)
        
        # Call LLM with retries
        response = await self._async_llm_call_with_retry(prompt)
        
        if not response:
            raise ValueError(f"Failed to generate {content_type} content after {MAX_RETRIES} attempts")
        
        # Parse response based on content type
        return self._parse_dynamic_response(content_type, response, context)
    
    def _build_dynamic_prompt(self, content_type: str, context: Dict, rag_context: str) -> str:
        """Build prompt for dynamic content generation"""
        
        base_context = f"""KNOWLEDGE BASE CONTEXT:
{rag_context if rag_context else "Use your knowledge about Indian conditions."}

"""
        
        if content_type == "implementation_plan":
            return base_context + f"""Generate a detailed RWH implementation plan.

PROJECT DETAILS:
- Location: {context.get('district')}, {context.get('state')}
- Catchment Area: {context.get('catchment_area')} sq m
- Roof Type: {context.get('roof_type')}
- Budget: ₹{context.get('budget')}
- Feasibility: {context.get('feasibility_score')}/100

GEOGRAPHY CONSIDERATIONS:
- Consider terrain of {context.get('state')} (hilly/plains/coastal/desert)
- Account for local monsoon patterns
- Factor in soil type for recharge pits

Generate JSON with:
{{
    "total_duration_days": <number>,
    "best_season_to_start": "<season based on local monsoon>",
    "geography_considerations": "<specific to this region>",
    "phases": [
        {{"phase_name": "<name>", "duration_days": <num>, "tasks": ["<task>"], "priority": "high/medium/low", "geography_note": "<local consideration>"}}
    ],
    "milestones": ["<milestone>"],
    "risk_factors": ["<risk>"],
    "local_resources": "<materials/labor notes>"
}}

Return ONLY valid JSON."""

        elif content_type == "recommendations":
            return base_context + f"""Generate specific RWH recommendations.

LOCATION: {context.get('district')}, {context.get('state')}
RAINFALL: {context.get('rainfall')} mm/year
GROUNDWATER: {context.get('gw_status')}
BUDGET: ₹{context.get('budget')}
ROOF: {context.get('roof_type')} - {context.get('roof_material')}

Consider:
1. Geography of {context.get('state')} (terrain, soil, climate)
2. Local traditional RWH methods
3. Government schemes available in this state
4. Cost-effective solutions for this budget

Generate JSON:
{{
    "recommendations": [
        {{"title": "<short title>", "description": "<2-3 sentences>", "priority": "high/medium/low", "cost_impact": "<low/medium/high>"}}
    ],
    "local_insights": "<region-specific advice>",
    "government_schemes": ["<available schemes in {context.get('state')}>"]
}}

Return ONLY valid JSON."""

        elif content_type == "water_distribution":
            return base_context + f"""Calculate optimal water distribution for rural farming.

FARM DETAILS:
- Location: {context.get('location')}
- Season: {context.get('season')}
- Cattle: {context.get('cattle_count')}
- Family: {context.get('household_members')} members
- Farm Size: {context.get('farm_size_acres')} acres
- Crop: {context.get('crop_type', 'General')}

CLIMATE DATA:
- Annual Rainfall: {context.get('annual_rainfall')} mm
- Season Rainfall: {context.get('season_rainfall')} mm
- Water Stress: {context.get('water_stress')}

Generate water distribution in buckets (1 bucket = 20L):
{{
    "irrigation_buckets": <number>,
    "cattle_buckets": <number>,
    "drinking_buckets": <number>,
    "total_daily_buckets": <number>,
    "water_status": "<Critical/Moderate/Sufficient/Surplus>",
    "recommendations": ["<3 specific tips>"],
    "ai_insights": "<50 word analysis of water situation>",
    "seasonal_adjustments": "<how to adjust for this season>"
}}

Return ONLY valid JSON."""

        elif content_type == "cost_analysis":
            return base_context + f"""Generate detailed RWH cost analysis.

PROJECT DETAILS:
- Location: {context.get('district')}, {context.get('state')}
- Catchment: {context.get('catchment_area')} sq m
- Roof: {context.get('roof_type')} - {context.get('roof_material')}
- Budget: ₹{context.get('budget')}
- Household: {context.get('n_members')} members
- Rainfall: {context.get('rainfall')} mm/year

Consider local material costs in {context.get('state')}.

Generate JSON:
{{
    "storage_tank": <cost in INR>,
    "filtration_system": <cost>,
    "first_flush_diverter": <cost>,
    "piping_and_fittings": <cost>,
    "gutters_and_channels": <cost>,
    "labor_charges": <cost>,
    "total_estimated_cost": <total>,
    "annual_maintenance": <cost>,
    "recommended_storage_liters": <liters>,
    "budget_tier": "<Ultra Budget/Budget/Standard/Premium>",
    "cost_optimization_tips": ["<tips to reduce cost>"],
    "roi_estimate_years": <payback period>
}}

Return ONLY valid JSON."""

        elif content_type == "feasibility":
            return base_context + f"""Analyze RWH feasibility for this location.

DETAILS:
- Location: {context.get('district')}, {context.get('state')}
- Rainfall: {context.get('rainfall')} mm/year
- Groundwater: {context.get('gw_extraction')}% extraction
- Catchment: {context.get('catchment_area')} sq m
- Members: {context.get('n_members')}
- Budget: ₹{context.get('budget')}
- Roof: {context.get('roof_type')}

Analyze feasibility considering geography of {context.get('state')}.

Generate JSON:
{{
    "overall_score": <0-100>,
    "category": "<Highly Feasible/Feasible/Moderately Feasible/Challenging/Difficult>",
    "factor_scores": {{
        "rainfall_adequacy": <0-100>,
        "budget_sufficiency": <0-100>,
        "catchment_efficiency": <0-100>,
        "groundwater_need": <0-100>,
        "implementation_complexity": <0-100>
    }},
    "strengths": ["<top 3 strengths>"],
    "challenges": ["<top 3 challenges>"],
    "recommendation": "<overall recommendation based on analysis>",
    "geography_factors": "<how local geography affects feasibility>"
}}

Return ONLY valid JSON."""

        else:
            return base_context + f"""Generate relevant content for: {content_type}
Context: {json.dumps(context)}
Return structured JSON response."""

    def _parse_dynamic_response(self, content_type: str, response: str, 
                                 context: Dict) -> Dict[str, Any]:
        """Parse LLM response into structured data"""
        result = self._parse_json_response(response)
        
        if result:
            return result
        
        # If JSON parsing fails, raise error - no hardcoded fallbacks
        raise ValueError(f"Failed to parse AI response for {content_type}. Response was: {response[:200]}...")
    
    async def _async_llm_call(self, prompt: str) -> str:
        """Make async LLM call using aiohttp"""
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not configured")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://indra-rwh.app",
                    "X-Title": "INDRA AI Service"
                },
                json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": LLM_TEMPERATURE,
                    "max_tokens": LLM_MAX_TOKENS
                },
                timeout=aiohttp.ClientTimeout(total=45)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['choices'][0]['message']['content']
                else:
                    error_text = await response.text()
                    raise ValueError(f"LLM API error {response.status}: {error_text}")
    
    async def _async_llm_call_with_retry(self, prompt: str) -> Optional[str]:
        """Make async LLM call with retry logic"""
        last_error = None
        
        for attempt in range(MAX_RETRIES):
            try:
                result = await self._async_llm_call(prompt)
                if result and result.strip():
                    return self._clean_text(result)
                print(f"[AI Service] Empty response on attempt {attempt + 1}")
            except Exception as e:
                last_error = e
                print(f"[AI Service] LLM error attempt {attempt + 1}: {e}")
            
            if attempt < MAX_RETRIES - 1:
                await self._async_sleep(RETRY_DELAY * (attempt + 1))
        
        if last_error:
            print(f"[AI Service] All retries failed: {last_error}")
        return None
    
    async def _async_sleep(self, seconds: float):
        """Async sleep"""
        import asyncio
        await asyncio.sleep(seconds)
    
    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """Parse JSON from LLM response"""
        if not response:
            return None
        
        cleaned = self._clean_text(response.strip())
        
        # Remove markdown code blocks
        if "```json" in cleaned:
            parts = cleaned.split("```json")
            if len(parts) > 1:
                cleaned = parts[1].split("```")[0]
        elif "```" in cleaned:
            parts = cleaned.split("```")
            if len(parts) > 1:
                cleaned = parts[1].split("```")[0] if len(parts) > 1 else parts[0]
        
        # Find JSON boundaries
        start = cleaned.find('{')
        if start == -1:
            return None
        
        # Find matching closing brace
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
            end = cleaned.rfind('}')
        
        if end <= start:
            return None
        
        json_str = cleaned[start:end+1]
        
        # Fix common JSON issues
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"[AI Service] JSON parse error: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """Remove emojis and clean text"""
        if not text:
            return text
        
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
        text = emoji_pattern.sub('', text)
        return text.strip()


# Singleton instance
ai_service = IndraAIService()


# Helper functions for other modules
async def verify_and_get_location_data(state: str, district: str, 
                                        gis_data: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    """
    Verify GIS data using RAG and return verified data with confidence
    Used by assessment, water, crop modules
    """
    result = await ai_service.fact_check_location_data(state, district, gis_data)
    
    # Merge verified data with original GIS data
    verified = gis_data.copy()
    if result.verified_data.get('rainfall'):
        verified['rainfall'] = result.verified_data['rainfall']
    if result.verified_data.get('groundwater'):
        verified['groundwater'] = result.verified_data['groundwater']
    
    return verified, result.confidence


async def generate_ai_content(content_type: str, context: Dict[str, Any], 
                              rag_query: str = None) -> Dict[str, Any]:
    """
    Generate dynamic AI content - wrapper for other modules
    Raises ValueError if generation fails (no hardcoded fallbacks)
    """
    return await ai_service.generate_dynamic_content(content_type, context, rag_query)
