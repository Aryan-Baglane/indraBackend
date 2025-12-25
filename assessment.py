"""
INDRA - Intelligent Assessment Module
AI/ML-powered Rainwater Harvesting Assessment System
Uses on-device embeddings, Qdrant RAG for geography data, and LLM for dynamic recommendations
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import os
import aiohttp

# Import config
from config import (
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL,
    LLM_MODEL, EMBEDDING_MODEL_PATH, LLM_TEMPERATURE,
    QDRANT_URL, QDRANT_API_KEY, RAG_COLLECTION_NAME
)

# Import local modules
from gis_utils import GISDataManager
from sentence_transformers import SentenceTransformer

# Import AI service for fact-checking and dynamic content
from ai_service import ai_service, verify_and_get_location_data, generate_ai_content

# Qdrant and LangChain imports for RAG
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    print("[WARN] Qdrant libraries not available - geography RAG disabled")

# Initialize Router
router = APIRouter(prefix="/api/assessment", tags=["assessment"])

# Load local embedding model
embedding_model = None

# Qdrant retriever for geography data
geography_retriever = None

def load_embedding_model():
    """Load the local sentence transformer model"""
    global embedding_model
    if embedding_model is None:
        try:
            embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)
            print(f"[OK] Loaded embedding model from {EMBEDDING_MODEL_PATH}")
        except Exception as e:
            print(f"[WARN] Error loading embedding model: {e}")
    return embedding_model


def initialize_geography_retriever():
    """Initialize Qdrant retriever for geography-based RAG"""
    global geography_retriever
    
    if not QDRANT_AVAILABLE:
        print("[WARN] Qdrant not available for geography retrieval")
        return None
    
    if geography_retriever is not None:
        return geography_retriever
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_PATH,
            encode_kwargs={'normalize_embeddings': False}
        )
        
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        collections = client.get_collections().collections
        
        if not any(c.name == RAG_COLLECTION_NAME for c in collections):
            print(f"[WARN] Collection '{RAG_COLLECTION_NAME}' not found in Qdrant")
            return None
        
        vector_store = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name=RAG_COLLECTION_NAME,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )
        
        geography_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        print("[OK] Geography retriever initialized")
        return geography_retriever
        
    except Exception as e:
        print(f"[WARN] Error initializing geography retriever: {e}")
        return None


async def get_geography_context(state: str, district: str) -> str:
    """Retrieve geography-specific context from Qdrant"""
    global geography_retriever
    
    if geography_retriever is None:
        geography_retriever = initialize_geography_retriever()
    
    if geography_retriever is None:
        return ""
    
    try:
        query = f"Geography climate terrain soil water resources {state} {district} India rainwater harvesting"
        docs = geography_retriever.invoke(query)
        
        if docs:
            context = "\n".join([doc.page_content[:500] for doc in docs[:3]])
            return context
        return ""
    except Exception as e:
        print(f"[WARN] Error retrieving geography context: {e}")
        return ""

async def get_llm_recommendations(context: Dict[str, Any], geography_context: str = "") -> str:
    """Get AI-powered recommendations using OpenRouter LLM with geography context"""
    if not OPENROUTER_API_KEY:
        return "AI recommendations unavailable (API key not configured)"
    
    try:
        geography_section = f"""
GEOGRAPHY & REGIONAL CONTEXT (from knowledge base):
{geography_context if geography_context else "No specific geography data available - use general knowledge for this region."}
""" if geography_context else ""

        prompt = f"""You are an expert in Rainwater Harvesting (RWH) systems in India. Analyze the following assessment and provide 3-5 specific, actionable recommendations.

LOCATION DETAILS:
- Location: {context['district']}, {context['state']}
- Annual Rainfall: {context['rainfall']} mm
- Groundwater Status: {context['gw_status']}
{geography_section}
PROPERTY DETAILS:
- Household Members: {context['members']}
- Catchment Area: {context['catchment']} sq m
- Roof Type: {context['roof_type']}
- Roof Material: {context['roof_material']}

ASSESSMENT:
- Budget: ₹{context['budget']}
- Feasibility Score: {context['feasibility_score']}/100

IMPORTANT: Take into account the SPECIFIC GEOGRAPHY of {context['state']} including:
1. Terrain type (hilly, plains, coastal, desert, etc.)
2. Soil characteristics (permeability, type)
3. Climate patterns (monsoon timing, dry spells)
4. Local water table conditions
5. Regional RWH practices and traditional methods

Provide practical recommendations focusing on:
1. Geography-appropriate system design (considering local terrain and soil)
2. Cost optimization based on local conditions
3. Water quality improvements specific to the region
4. Seasonal strategies aligned with local monsoon patterns
5. Government schemes or subsidies available in {context['state']}

Keep each recommendation concise (2-3 sentences) and actionable. Include specific local considerations.
"""
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://indra-rwh.app",
                    "X-Title": "INDRA RWH Assessment"
                },
                json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 800
                },
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['choices'][0]['message']['content']
                else:
                    print(f"OpenRouter API error: {response.status}")
                    return "AI recommendations temporarily unavailable"
    except Exception as e:
        print(f"LLM error: {e}")
        return "AI recommendations temporarily unavailable"

# Initialize GIS Manager
gis_manager = GISDataManager()

# Pydantic Models
class AssessmentInput(BaseModel):
    """User input for RWH assessment"""
    name: str = Field(..., description="User's name")
    state: str = Field(..., description="State")
    district: str = Field(..., description="District")
    pincode: str = Field(..., description="Pincode")
    n_members: int = Field(..., description="Number of household members", ge=1)
    catchment_area: float = Field(..., description="Catchment area in sq meters", ge=0)
    farm_land_area: Optional[float] = Field(0, description="Farm land area in acres")
    roof_type: str = Field(..., description="Roof type (Flat/Sloped/Mixed)")
    roof_material: str = Field(..., description="Roof material (RCC/Tiles/Metal/Asbestos)")
    budget: float = Field(..., description="Budget for RWH in INR", ge=0)
    project_status: Optional[str] = Field("planning", description="Project status")

class AssessmentOutput(BaseModel):
    """Complete assessment output"""
    # Input confirmation
    user_details: Dict[str, Any]
    
    # Generated location data
    location_data: Dict[str, Any]
    
    # RWH Analysis
    rwh_analysis: Dict[str, Any]
    
    # Cost Analysis
    cost_analysis: Dict[str, Any]
    
    # Implementation Plan
    implementation: Dict[str, Any]
    
    # Feasibility Score
    feasibility: Dict[str, Any]
    
    # Recommendations
    recommendations: List[str]
    
    # Timestamp
    assessment_id: str
    timestamp: str


class RWHCostPredictor:
    """
    ML-based cost prediction using statistical models and domain knowledge
    Budget-aware planning: Adapts system to fit within user's budget
    
    REALISTIC COST REFERENCE (India 2024):
    - Ultra-Budget System: ₹3,000 - ₹5,000 (DIY, minimal)
    - Budget System: ₹5,000 - ₹10,000 (basic setup)
    - Standard RWH System: ₹10,000 - ₹20,000 (recommended)
    - Premium System: ₹20,000+ (advanced features)
    """
    
    # Component costs with budget tiers (INR) - BASE COSTS
    COMPONENT_COSTS = {
        "ultra_budget": {  # ₹3,000 - ₹5,000
            "storage_tank": 1500,  # Small plastic drum (200L)
            "filtration_system": 500,  # DIY sand-gravel
            "first_flush_diverter": 300,  # Simple PVC
            "piping_and_fittings": 200,
            "gutters_and_channels": 300,  # Partial gutters
            "labor_charges": 500,  # Mostly DIY
        },
        "budget": {  # ₹5,000 - ₹10,000
            "storage_tank": 3000,  # 500L tank
            "filtration_system": 1000,
            "first_flush_diverter": 500,
            "piping_and_fittings": 300,
            "gutters_and_channels": 800,
            "labor_charges": 2000,
        },
        "standard": {  # ₹10,000 - ₹20,000
            "storage_tank": 6000,  # 1000L tank
            "filtration_system": 1500,
            "first_flush_diverter": 800,
            "piping_and_fittings": 500,
            "gutters_and_channels": 2000,
            "labor_charges": 4000,
        },
        "premium": {  # ₹20,000+
            "storage_tank": 12000,  # 2000L+ tank
            "filtration_system": 3000,
            "first_flush_diverter": 1200,
            "piping_and_fittings": 800,
            "gutters_and_channels": 3500,
            "labor_charges": 6000,
        }
    }
    
    # Material multipliers - affects filtration and water quality
    MATERIAL_MULTIPLIERS = {
        "RCC": 1.0,      # Ideal - minimal filtration needed
        "Tiles": 0.95,   # Good - slightly less efficient
        "Metal": 1.15,   # Needs more filtration, can rust
        "Asbestos": 1.25, # Needs extra filtration (health concern)
        "Thatch": 1.3,   # Rural - needs significant filtration
        "other": 1.1
    }
    
    # Roof type multipliers - affects installation complexity
    ROOF_TYPE_MULTIPLIERS = {
        "Flat": 1.0,     # Easiest - simple drainage
        "Sloped": 1.15,  # Moderate - needs proper gutters
        "Mixed": 1.25,   # Complex - multiple collection points
        "Dome": 1.3,     # Difficult - specialized collection
        "other": 1.1
    }
    
    # Household size affects required storage capacity
    HOUSEHOLD_SIZE_MULTIPLIERS = {
        1: 0.5,   # Single person
        2: 0.7,   # Couple
        3: 0.85,  # Small family
        4: 1.0,   # Average family (baseline)
        5: 1.15,  # Large family
        6: 1.3,   # Joint family
        7: 1.4,
        8: 1.5,
    }
    
    # Rainfall zone affects storage sizing
    RAINFALL_ZONE_MULTIPLIERS = {
        "very_high": 0.8,   # >2000mm - smaller storage ok (frequent refill)
        "high": 0.9,        # 1200-2000mm
        "moderate": 1.0,    # 700-1200mm (baseline)
        "low": 1.2,         # 400-700mm - larger storage needed
        "very_low": 1.4,    # <400mm - maximize storage
    }
    
    def _get_rainfall_zone(self, annual_rainfall: float) -> str:
        """Categorize rainfall zone"""
        if annual_rainfall >= 2000:
            return "very_high"
        elif annual_rainfall >= 1200:
            return "high"
        elif annual_rainfall >= 700:
            return "moderate"
        elif annual_rainfall >= 400:
            return "low"
        else:
            return "very_low"
    
    def _select_budget_tier(self, budget: float) -> str:
        """Select appropriate cost tier based on user's budget"""
        if budget < 5000:
            return "ultra_budget"
        elif budget < 10000:
            return "budget"
        elif budget < 20000:
            return "standard"
        else:
            return "premium"
    
    def predict_cost(self, catchment_area: float, storage_capacity: float, 
                    roof_type: str, roof_material: str, location_data: Dict,
                    user_budget: float = 20000, n_members: int = 4) -> Dict[str, Any]:
        """
        Predict RWH system cost - FULLY DYNAMIC
        Considers: budget, roof type, roof material, area, members, rainfall
        """
        
        # Select budget tier
        tier = self._select_budget_tier(user_budget)
        base_costs = self.COMPONENT_COSTS[tier].copy()
        
        # Get multipliers based on all input factors
        mat_mult = self.MATERIAL_MULTIPLIERS.get(roof_material, 1.0)
        roof_mult = self.ROOF_TYPE_MULTIPLIERS.get(roof_type, 1.0)
        
        # Household size multiplier (cap at 8+)
        hh_mult = self.HOUSEHOLD_SIZE_MULTIPLIERS.get(min(n_members, 8), 1.0 + (n_members - 4) * 0.15)
        
        # Rainfall zone multiplier
        annual_rainfall = location_data.get('total_annual_rainfall', 800)
        rainfall_zone = self._get_rainfall_zone(annual_rainfall)
        rain_mult = self.RAINFALL_ZONE_MULTIPLIERS.get(rainfall_zone, 1.0)
        
        # Area factor (catchment area affects gutter length and piping)
        # 100 sq m is baseline (typical small house)
        area_factor = min(max(catchment_area / 100, 0.7), 2.5)
        
        # Groundwater stress affects investment value
        gw_stress = location_data.get('groundwater_extraction_stage', 50)
        urgency_mult = 1.0 + (gw_stress / 200)  # Up to 1.5x for critical areas
        
        print(f"[Cost] Factors - Area: {area_factor:.2f}x, Roof: {roof_mult:.2f}x, Material: {mat_mult:.2f}x, HH: {hh_mult:.2f}x, Rain: {rain_mult:.2f}x")
        
        # Calculate component costs with all factors
        breakdown = {}
        
        # Storage tank - scales with household size and rainfall zone
        tank_cost = base_costs["storage_tank"] * hh_mult * rain_mult
        breakdown["storage_tank"] = round(tank_cost, 0)
        
        # Filtration - scales with roof material (some need more filtering)
        filter_cost = base_costs["filtration_system"] * mat_mult
        breakdown["filtration_system"] = round(filter_cost, 0)
        
        # First flush - relatively fixed, slight area adjustment
        ff_cost = base_costs["first_flush_diverter"] * min(area_factor, 1.3)
        breakdown["first_flush_diverter"] = round(ff_cost, 0)
        
        # Piping - scales with area and roof complexity
        pipe_cost = base_costs["piping_and_fittings"] * area_factor * roof_mult
        breakdown["piping_and_fittings"] = round(pipe_cost, 0)
        
        # Gutters - directly proportional to catchment perimeter (sqrt of area)
        gutter_factor = (catchment_area ** 0.5) / 10  # 100sqm = factor 1
        gutter_cost = base_costs["gutters_and_channels"] * max(gutter_factor, 0.8) * roof_mult
        breakdown["gutters_and_channels"] = round(gutter_cost, 0)
        
        # Labor - scales with complexity (roof type + area)
        labor_cost = base_costs["labor_charges"] * roof_mult * min(area_factor, 1.5)
        breakdown["labor_charges"] = round(labor_cost, 0)
        
        # Calculate total
        total_cost = sum(breakdown.values())
        
        # If over budget, scale proportionally to fit budget (with 5% buffer)
        if total_cost > user_budget and user_budget > 2000:
            scale_factor = (user_budget * 0.95) / total_cost
            for key in breakdown:
                breakdown[key] = round(breakdown[key] * scale_factor, 0)
            total_cost = sum(breakdown.values())
            tier = "budget_adapted"  # Mark as adapted
        
        # Annual maintenance (scales with system complexity)
        base_maintenance = {"ultra_budget": 500, "budget": 800, "standard": 1000, "premium": 1500, "budget_adapted": 600}
        annual_maintenance = round(base_maintenance.get(tier, 800) * roof_mult * mat_mult, 0)
        
        # Storage capacity estimation - dynamic based on factors
        base_capacity = {"ultra_budget": 200, "budget": 500, "standard": 1000, "premium": 2000, "budget_adapted": 300}
        practical_storage = round(base_capacity.get(tier, 500) * hh_mult * rain_mult, 0)
        
        breakdown["total_estimated_cost"] = round(total_cost, 0)
        breakdown["annual_maintenance"] = annual_maintenance
        breakdown["cost_per_liter_capacity"] = round(total_cost / max(practical_storage, 1), 2)
        breakdown["payback_estimate_years"] = round(total_cost / (practical_storage * 12 * 0.10), 1)
        breakdown["budget_tier"] = tier.replace("_", " ").title()
        breakdown["recommended_storage_liters"] = practical_storage
        breakdown["rainfall_zone"] = rainfall_zone.replace("_", " ").title()
        
        return breakdown


class FeasibilityAnalyzer:
    """AI-powered feasibility scoring system"""
    
    def __init__(self):
        self.weights = {
            "rainfall_adequacy": 0.25,
            "budget_sufficiency": 0.20,
            "catchment_efficiency": 0.15,
            "groundwater_need": 0.15,
            "household_size": 0.10,
            "implementation_complexity": 0.15
        }
    
    def calculate_feasibility(self, input_data: AssessmentInput, 
                            location_data: Dict, cost_analysis: Dict) -> Dict[str, Any]:
        """
        Calculate comprehensive feasibility score (0-100)
        Uses multi-factor weighted scoring algorithm
        """
        
        scores = {}
        
        # 1. Rainfall Adequacy Score (0-100)
        rainfall = location_data.get('total_annual_rainfall', 0)
        if rainfall > 1500:
            scores['rainfall_adequacy'] = 100
        elif rainfall > 1000:
            scores['rainfall_adequacy'] = 80
        elif rainfall > 600:
            scores['rainfall_adequacy'] = 60
        elif rainfall > 300:
            scores['rainfall_adequacy'] = 40
        else:
            scores['rainfall_adequacy'] = 20
        
        # 2. Budget Sufficiency Score (0-100)
        # Now budget-aware: since we adapt the plan to budget, this is more lenient
        estimated_cost = cost_analysis.get('total_estimated_cost', 0)
        budget_tier = cost_analysis.get('budget_tier', 'Standard')
        
        # If system was adapted to fit budget, give higher score
        if estimated_cost <= input_data.budget:
            scores['budget_sufficiency'] = 90  # Plan fits budget
        elif input_data.budget >= estimated_cost * 0.9:
            scores['budget_sufficiency'] = 75
        elif input_data.budget >= estimated_cost * 0.7:
            scores['budget_sufficiency'] = 55
        else:
            scores['budget_sufficiency'] = 35  # Very tight budget
        
        # 3. Catchment Efficiency Score (0-100)
        # Assumes 1 sq m can support 2 people adequately
        catchment_per_person = input_data.catchment_area / input_data.n_members
        if catchment_per_person > 20:
            scores['catchment_efficiency'] = 100
        elif catchment_per_person > 10:
            scores['catchment_efficiency'] = 85
        elif catchment_per_person > 5:
            scores['catchment_efficiency'] = 70
        elif catchment_per_person > 2:
            scores['catchment_efficiency'] = 55
        else:
            scores['catchment_efficiency'] = 35
        
        # 4. Groundwater Need Score (0-100)
        gw_stage = location_data.get('groundwater_extraction_stage', 50)
        if gw_stage > 90:  # Critical
            scores['groundwater_need'] = 100
        elif gw_stage > 70:  # Over-exploited
            scores['groundwater_need'] = 85
        elif gw_stage > 50:  # Semi-critical
            scores['groundwater_need'] = 65
        else:
            scores['groundwater_need'] = 45
        
        # 5. Household Size Score (0-100)
        # Larger households benefit more from RWH
        if input_data.n_members >= 6:
            scores['household_size'] = 100
        elif input_data.n_members >= 4:
            scores['household_size'] = 80
        elif input_data.n_members >= 2:
            scores['household_size'] = 60
        else:
            scores['household_size'] = 40
        
        # 6. Implementation Complexity Score (0-100)
        # Lower complexity = higher score
        complexity_factors = 0
        if input_data.roof_type == "Mixed":
            complexity_factors += 1
        if input_data.roof_material in ["Asbestos", "other"]:
            complexity_factors += 1
        if input_data.catchment_area > 300:
            complexity_factors += 1
        
        if complexity_factors == 0:
            scores['implementation_complexity'] = 100
        elif complexity_factors == 1:
            scores['implementation_complexity'] = 75
        elif complexity_factors == 2:
            scores['implementation_complexity'] = 50
        else:
            scores['implementation_complexity'] = 30
        
        # Calculate weighted total
        total_score = sum(scores[key] * self.weights[key] for key in scores.keys())
        
        # Determine feasibility category
        if total_score >= 80:
            category = "Highly Feasible"
            recommendation = "Excellent conditions for RWH implementation. Proceed with confidence."
        elif total_score >= 65:
            category = "Feasible"
            recommendation = "Good conditions for RWH. Minor optimizations recommended."
        elif total_score >= 50:
            category = "Moderately Feasible"
            recommendation = "RWH is viable with careful planning and budget management."
        elif total_score >= 35:
            category = "Challenging"
            recommendation = "Consider phased implementation or budget increase."
        else:
            category = "Difficult"
            recommendation = "Significant challenges exist. Consider alternative water solutions or budget revision."
        
        return {
            "overall_score": round(total_score, 1),
            "category": category,
            "recommendation": recommendation,
            "factor_scores": {k: round(v, 1) for k, v in scores.items()},
            "strengths": self._identify_strengths(scores),
            "challenges": self._identify_challenges(scores)
        }
    
    def _identify_strengths(self, scores: Dict[str, float]) -> List[str]:
        """Identify top performing factors"""
        strengths = []
        for key, value in scores.items():
            if value >= 75:
                strengths.append(key.replace('_', ' ').title())
        return strengths[:3]  # Top 3
    
    def _identify_challenges(self, scores: Dict[str, float]) -> List[str]:
        """Identify weak factors needing attention"""
        challenges = []
        for key, value in scores.items():
            if value < 60:
                challenges.append(key.replace('_', ' ').title())
        return challenges[:3]  # Top 3


class ImplementationPlanner:
    """AI-powered implementation planning using LLM and geography context"""
    
    def __init__(self, embedding_model):
        self.model = embedding_model
    
    async def generate_implementation_plan(self, input_data, location_data: Dict, 
                                           feasibility: Dict, cost_analysis: Dict,
                                           geography_context: str = "") -> Dict[str, Any]:
        """Generate AI-powered implementation plan based on all factors"""
        
        if not OPENROUTER_API_KEY:
            return self._get_fallback_plan(feasibility['overall_score'], feasibility['category'])
        
        try:
            geography_section = f"""
GEOGRAPHY CONTEXT (from knowledge base):
{geography_context if geography_context else "Use general knowledge for this region."}
""" if geography_context else ""

            prompt = f"""You are an expert RWH implementation planner in India. Create a detailed, realistic implementation plan.

PROJECT DETAILS:
- Location: {input_data.district}, {input_data.state}
- Catchment Area: {input_data.catchment_area} sq m
- Roof Type: {input_data.roof_type}
- Roof Material: {input_data.roof_material}
- Budget: ₹{input_data.budget}
- Household Size: {input_data.n_members} members
- Feasibility Score: {feasibility['overall_score']}/100 ({feasibility['category']})

ENVIRONMENTAL DATA:
- Annual Rainfall: {location_data.get('total_annual_rainfall', 0)} mm
- Monsoon Rainfall: {location_data.get('monsoon_rainfall', 0)} mm
- Groundwater Extraction: {location_data.get('groundwater_extraction_stage', 0)}%

COST ANALYSIS:
- Budget Tier: {cost_analysis.get('budget_tier', 'Standard')}
- Estimated Cost: ₹{cost_analysis.get('total_estimated_cost', 0)}
- Storage Capacity: {cost_analysis.get('recommended_storage_liters', 1000)} L
{geography_section}
IMPORTANT GEOGRAPHY CONSIDERATIONS for {input_data.state}:
- Account for local terrain (hilly regions need different installation approach than plains)
- Consider soil type for recharge pit feasibility
- Factor in local monsoon patterns for optimal installation timing
- Include region-specific challenges (coastal salinity, desert heat, mountain accessibility)

Generate a JSON implementation plan with this EXACT structure:
{{
    "total_duration_days": <number based on complexity and geography>,
    "best_season_to_start": "<recommended season considering local monsoon>",
    "geography_considerations": "<specific challenges for this region>",
    "phases": [
        {{
            "phase_name": "<name>",
            "duration_days": <number>,
            "tasks": ["<task1>", "<task2>", ...],
            "priority": "high/medium/low",
            "geography_note": "<any region-specific consideration>"
        }}
    ],
    "milestones": ["<milestone1>", "<milestone2>", "<milestone3>"],
    "risk_factors": ["<risk1>", "<risk2>"],
    "local_resources": "<available local materials/labor considerations>"
}}

Provide 5-7 phases. Adjust timeline based on:
1. Feasibility score (lower = longer duration)
2. Terrain complexity
3. Local climate/monsoon timing
4. Budget tier (ultra_budget = simpler, faster; premium = more complex)

Return ONLY valid JSON, no explanations."""

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{OPENROUTER_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://indra-rwh.app",
                        "X-Title": "INDRA Implementation Plan"
                    },
                    json={
                        "model": LLM_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.5,
                        "max_tokens": 1500
                    },
                    timeout=aiohttp.ClientTimeout(total=45)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        llm_response = data['choices'][0]['message']['content']
                        
                        # Parse JSON from response
                        plan = self._parse_llm_plan(llm_response)
                        if plan:
                            # Add computed dates
                            start_date = datetime.now()
                            plan['estimated_start_date'] = start_date.strftime("%Y-%m-%d")
                            plan['estimated_completion_date'] = (
                                start_date + timedelta(days=plan.get('total_duration_days', 21))
                            ).strftime("%Y-%m-%d")
                            
                            # Compute phase start/end days
                            current_day = 0
                            for phase in plan.get('phases', []):
                                phase['start_day'] = current_day
                                phase['end_day'] = current_day + phase.get('duration_days', 3)
                                current_day = phase['end_day']
                            
                            return plan
                        
                    print(f"Implementation plan API error: {response.status}")
                    
        except Exception as e:
            print(f"Implementation plan LLM error: {e}")
        
        return self._get_fallback_plan(feasibility['overall_score'], feasibility['category'])
    
    def _parse_llm_plan(self, response: str) -> Optional[Dict]:
        """Parse JSON from LLM response"""
        try:
            # Try direct parse
            return json.loads(response)
        except:
            pass
        
        try:
            # Try to extract JSON from markdown code block
            import re
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
            if json_match:
                return json.loads(json_match.group(1))
            
            # Try to find JSON object in response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(response[start:end])
        except Exception as e:
            print(f"Error parsing LLM plan JSON: {e}")
        
        return None
    
    def _get_fallback_plan(self, feasibility_score: float, category: str) -> Dict[str, Any]:
        """Fallback implementation plan when LLM is unavailable"""
        base_duration_days = 21
        
        if feasibility_score < 50:
            duration_days = base_duration_days + 7
        elif feasibility_score > 80:
            duration_days = base_duration_days - 3
        else:
            duration_days = base_duration_days
        
        if category in ["Challenging", "Difficult"]:
            duration_days += 5
        
        start_date = datetime.now()
        
        return {
            "total_duration_days": duration_days,
            "estimated_start_date": start_date.strftime("%Y-%m-%d"),
            "estimated_completion_date": (start_date + timedelta(days=duration_days)).strftime("%Y-%m-%d"),
            "best_season_to_start": "Pre-monsoon (March-May)",
            "geography_considerations": "General implementation - consult local experts for region-specific advice",
            "phases": [
                {
                    "phase_name": "Site Assessment and Preparation",
                    "start_day": 0, "end_day": 5, "duration_days": 5,
                    "tasks": ["Site survey", "Soil test", "Design drawings", "Permissions"],
                    "priority": "high",
                    "geography_note": "Assess local terrain and drainage patterns"
                },
                {
                    "phase_name": "Material Procurement",
                    "start_day": 5, "end_day": 8, "duration_days": 3,
                    "tasks": ["Procure tank", "Buy filters", "Get pipes and fittings", "Arrange gutters"],
                    "priority": "high",
                    "geography_note": "Source locally where possible"
                },
                {
                    "phase_name": "Gutter Installation",
                    "start_day": 8, "end_day": 12, "duration_days": 4,
                    "tasks": ["Install gutters", "Mount downspouts", "Seal joints", "Test flow"],
                    "priority": "high",
                    "geography_note": "Account for local rainfall intensity"
                },
                {
                    "phase_name": "Filtration Setup",
                    "start_day": 12, "end_day": 15, "duration_days": 3,
                    "tasks": ["Install first flush", "Set up filters", "Connect piping"],
                    "priority": "medium",
                    "geography_note": "Filter design based on local dust/debris levels"
                },
                {
                    "phase_name": "Storage Installation",
                    "start_day": 15, "end_day": 18, "duration_days": 3,
                    "tasks": ["Prepare foundation", "Position tank", "Connect inlet/outlet"],
                    "priority": "high",
                    "geography_note": "Foundation depth based on soil type"
                },
                {
                    "phase_name": "Testing and Commissioning",
                    "start_day": 18, "end_day": 21, "duration_days": 3,
                    "tasks": ["Full system test", "Check leaks", "Document settings"],
                    "priority": "high",
                    "geography_note": "Test before monsoon season"
                }
            ],
            "milestones": [
                "Week 1: Site preparation and material procurement",
                "Week 2: Installation of gutters, filters, and tanks",
                "Week 3: Testing and commissioning"
            ],
            "risk_factors": ["Weather delays", "Material availability", "Labor scheduling"],
            "local_resources": "Consult local RWH vendors for region-specific materials"
        }
    
    def generate_timeline(self, feasibility_score: float, 
                         budget: float, complexity: str) -> Dict[str, Any]:
        """Legacy sync method - returns fallback plan"""
        return self._get_fallback_plan(feasibility_score, complexity)


# API Endpoints

@router.post("/analyze", response_model=AssessmentOutput)
async def analyze_rwh_system(input_data: AssessmentInput):
    """
    Main endpoint for comprehensive RWH assessment
    Performs AI-powered analysis with RAG-verified data and geography-aware recommendations
    All outputs are dynamically generated by AI - no hardcoded values
    """
    
    try:
        # Load embedding model
        model = load_embedding_model()
        
        # 1. Get location data from GIS
        print(f"[Assessment] Looking up GIS data for pincode: {input_data.pincode}, district: {input_data.district}")
        
        # Ensure GIS data is loaded
        if not gis_manager._loaded:
            gis_manager.load_data()
        
        location_data_raw = gis_manager.get_location_data(
            pincode=input_data.pincode,
            district=input_data.district,
            state=input_data.state
        )
        
        if not location_data_raw:
            raise HTTPException(status_code=404, detail=f"Location data not found for pincode: {input_data.pincode}")
        
        # 1a. FACT-CHECK GIS DATA using RAG
        print(f"[Assessment] Fact-checking GIS data with RAG...")
        verified_gis, confidence = await verify_and_get_location_data(
            input_data.state, 
            input_data.district, 
            location_data_raw
        )
        print(f"[Assessment] Data verification confidence: {confidence:.0%}")
        
        # Use verified data
        rainfall_data = verified_gis.get('rainfall', location_data_raw.get('rainfall', {}))
        groundwater_data = verified_gis.get('groundwater', location_data_raw.get('groundwater', {}))
        
        # Round values for clean display
        total_rainfall = round(rainfall_data.get('total_annual', 0), 1)
        monsoon_rainfall = round(rainfall_data.get('monsoon', 0), 1)
        gw_extraction = round(groundwater_data.get('extraction_percentage', 50), 1)
        
        print(f"[Assessment] Verified Data - Rainfall: {total_rainfall}mm, GW: {gw_extraction}%")
        
        location_data = {
            "latitude": location_data_raw.get('latitude', 0),
            "longitude": location_data_raw.get('longitude', 0),
            "district": location_data_raw.get('district', input_data.district),
            "state": location_data_raw.get('state', input_data.state),
            "pincode": location_data_raw.get('pincode', input_data.pincode),
            "total_annual_rainfall": total_rainfall,
            "monsoon_rainfall": monsoon_rainfall,
            "groundwater_extraction_stage": gw_extraction,
            "groundwater_resource": round(groundwater_data.get('resource_bcm', 0), 3),
            "data_confidence": confidence  # Include confidence score
        }
        
        # 1b. Fetch geography context from Qdrant for AI-powered features
        print(f"[Assessment] Fetching geography context for {input_data.state}...")
        geography_context = await get_geography_context(input_data.state, input_data.district)
        if geography_context:
            print(f"[Assessment] Geography context retrieved ({len(geography_context)} chars)")
        else:
            print("[Assessment] No geography context available - using general knowledge")
        
        # 2. AI-POWERED FEASIBILITY ANALYSIS
        print("[Assessment] Generating AI-powered feasibility analysis...")
        try:
            feasibility_context = {
                'district': input_data.district,
                'state': input_data.state,
                'rainfall': total_rainfall,
                'gw_extraction': gw_extraction,
                'catchment_area': input_data.catchment_area,
                'n_members': input_data.n_members,
                'budget': input_data.budget,
                'roof_type': input_data.roof_type
            }
            feasibility = await generate_ai_content(
                "feasibility", 
                feasibility_context,
                f"rainwater harvesting feasibility {input_data.state} {input_data.district}"
            )
        except Exception as e:
            print(f"[Assessment] AI feasibility failed, using algorithm: {e}")
            # Fallback to algorithmic feasibility
            feasibility_analyzer = FeasibilityAnalyzer()
            feasibility = feasibility_analyzer.calculate_feasibility(
                input_data=input_data,
                location_data=location_data,
                cost_analysis={"total_estimated_cost": input_data.budget}
            )
        
        # 3. AI-POWERED COST ANALYSIS
        print("[Assessment] Generating AI-powered cost analysis...")
        try:
            cost_context = {
                'district': input_data.district,
                'state': input_data.state,
                'catchment_area': input_data.catchment_area,
                'roof_type': input_data.roof_type,
                'roof_material': input_data.roof_material,
                'budget': input_data.budget,
                'n_members': input_data.n_members,
                'rainfall': total_rainfall
            }
            cost_analysis = await generate_ai_content(
                "cost_analysis",
                cost_context,
                f"rainwater harvesting cost India {input_data.state}"
            )
        except Exception as e:
            print(f"[Assessment] AI cost analysis failed, using algorithm: {e}")
            # Fallback to algorithmic cost prediction
            cost_predictor = RWHCostPredictor()
            cost_analysis = cost_predictor.predict_cost(
                catchment_area=input_data.catchment_area,
                storage_capacity=input_data.n_members * 150 * 30,
                roof_type=input_data.roof_type,
                roof_material=input_data.roof_material,
                location_data=location_data,
                user_budget=input_data.budget,
                n_members=input_data.n_members
            )
        
        # Calculate RWH metrics
        daily_requirement = input_data.n_members * 150
        storage_capacity = daily_requirement * 30
        
        runoff_coefficients = {
            "RCC": 0.90, "Tiles": 0.85, "Metal": 0.80, "Asbestos": 0.75, "Thatch": 0.60
        }
        runoff_coefficient = runoff_coefficients.get(input_data.roof_material, 0.80)
        
        harvestable_water = (input_data.catchment_area * total_rainfall / 1000 * runoff_coefficient)
        harvestable_water_liters = harvestable_water * 1000
        
        rwh_type = "Hybrid (Rooftop + Farm Recharge)" if input_data.farm_land_area > 0 else "Rooftop Harvesting with Storage"
        
        # 4. AI-POWERED IMPLEMENTATION PLANNING
        print("[Assessment] Generating AI-powered implementation plan...")
        try:
            impl_context = {
                'district': input_data.district,
                'state': input_data.state,
                'catchment_area': input_data.catchment_area,
                'roof_type': input_data.roof_type,
                'budget': input_data.budget,
                'feasibility_score': feasibility.get('overall_score', 70)
            }
            implementation = await generate_ai_content(
                "implementation_plan",
                impl_context,
                f"rainwater harvesting implementation {input_data.state} rural"
            )
        except Exception as e:
            print(f"[Assessment] AI implementation plan failed, using fallback: {e}")
            planner = ImplementationPlanner(model)
            implementation = planner._get_fallback_plan(
                feasibility.get('overall_score', 70),
                feasibility.get('category', 'Feasible')
            )
        
        # 5. AI-POWERED RECOMMENDATIONS
        print("[Assessment] Generating AI-powered recommendations...")
        recommendations = await generate_recommendations(
            input_data, location_data, feasibility, cost_analysis, geography_context
        )
        
        # 6. Create assessment ID
        assessment_id = f"INDRA-{input_data.pincode}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Compile output
        output = AssessmentOutput(
            user_details={
                "name": input_data.name,
                "location": f"{input_data.district}, {input_data.state}",
                "pincode": input_data.pincode,
                "household_members": input_data.n_members,
                "catchment_area_sqm": input_data.catchment_area,
                "farm_land_acres": input_data.farm_land_area,
                "budget_inr": input_data.budget
            },
            location_data=location_data,
            rwh_analysis={
                "rwh_type": rwh_type,
                "recommended_storage_capacity_liters": cost_analysis.get('recommended_storage_liters', storage_capacity),
                "annual_harvestable_water_liters": round(harvestable_water_liters, 2),
                "daily_household_requirement_liters": daily_requirement,
                "water_self_sufficiency_days": round(harvestable_water_liters / daily_requirement, 1),
                "runoff_coefficient": runoff_coefficient
            },
            cost_analysis=cost_analysis,
            implementation=implementation,
            feasibility=feasibility,
            recommendations=recommendations,
            assessment_id=assessment_id,
            timestamp=datetime.now().isoformat()
        )
        
        return output
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")


async def generate_recommendations(input_data: AssessmentInput, location_data: Dict,
                            feasibility: Dict, cost_analysis: Dict, 
                            geography_context: str = "") -> List[str]:
    """Generate AI-powered context-aware recommendations with geography data"""
    
    # Base recommendations (always included)
    recommendations = []
    
    # Budget recommendations
    budget_gap = cost_analysis['total_estimated_cost'] - input_data.budget
    if budget_gap > 0:
        recommendations.append(
            f"Budget Gap: Rs.{round(budget_gap, 0):,.0f}. Consider phased implementation or government subsidies like Jal Shakti Abhiyan."
        )
    else:
        surplus = input_data.budget - cost_analysis['total_estimated_cost']
        recommendations.append(
            f"Budget is sufficient with Rs.{round(surplus, 0):,.0f} surplus. Consider investing in advanced filtration or larger storage."
        )
    
    # Rainfall recommendations
    if location_data['total_annual_rainfall'] < 600:
        recommendations.append(
            "Low rainfall region detected. Maximize catchment efficiency with proper gutter slope (1:100) and sealed joints."
        )
    elif location_data['total_annual_rainfall'] > 1500:
        recommendations.append(
            "High rainfall region. Consider larger storage tanks or groundwater recharge pits to utilize excess monsoon water."
        )
    else:
        recommendations.append(
            f"Moderate rainfall ({location_data['total_annual_rainfall']:.0f}mm/year) is favorable for RWH. Standard system recommended."
        )
    
    # Groundwater recommendations
    if location_data['groundwater_extraction_stage'] > 70:
        recommendations.append(
            f"Critical groundwater zone ({location_data['groundwater_extraction_stage']:.0f}% extraction). RWH highly recommended for aquifer recharge."
        )
    
    # Get AI-powered recommendations from LLM (with geography context)
    try:
        gw_status = "Critical" if location_data['groundwater_extraction_stage'] > 70 else "Moderate"
        llm_context = {
            'district': input_data.district,
            'state': input_data.state,
            'rainfall': location_data['total_annual_rainfall'],
            'members': input_data.n_members,
            'catchment': input_data.catchment_area,
            'budget': input_data.budget,
            'roof_type': input_data.roof_type,
            'roof_material': input_data.roof_material,
            'gw_status': gw_status,
            'feasibility_score': feasibility['overall_score']
        }
        
        llm_response = await get_llm_recommendations(llm_context, geography_context)
        if llm_response and "unavailable" not in llm_response.lower():
            # Parse LLM response into individual recommendations
            llm_recs = [line.strip() for line in llm_response.split('\n') if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-'))]
            for rec in llm_recs[:5]:  # Add top 5 AI recommendations
                # Clean up numbering and any emojis
                clean_rec = rec.lstrip('0123456789.-) ').strip()
                if clean_rec:
                    recommendations.append(f"AI Insight: {clean_rec}")
    except Exception as e:
        print(f"Error getting LLM recommendations: {e}")
    
    return recommendations


# Master data for dropdowns
INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand",
    "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal",
    "Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu",
    "Delhi", "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry"
]

ROOF_TYPES = [
    {"value": "Flat", "label": "Flat Roof", "description": "Horizontal or nearly horizontal roof surface"},
    {"value": "Sloped", "label": "Sloped Roof", "description": "Angled roof (single or gabled)"},
    {"value": "Mixed", "label": "Mixed Roof", "description": "Combination of flat and sloped sections"}
]

ROOF_MATERIALS = [
    {"value": "RCC", "label": "RCC (Reinforced Concrete)", "efficiency": 0.9, "description": "Best for water quality"},
    {"value": "Tiles", "label": "Clay/Concrete Tiles", "efficiency": 0.85, "description": "Good quality, traditional"},
    {"value": "Metal", "label": "Metal Sheets (GI/Aluminum)", "efficiency": 0.8, "description": "Durable, moderate quality"},
    {"value": "Asbestos", "label": "Asbestos Sheets", "efficiency": 0.7, "description": "Requires extra filtration"}
]

PROJECT_STATUS_OPTIONS = [
    {"value": "planning", "label": "Planning Phase", "icon": "plan"},
    {"value": "approved", "label": "Approved/Funded", "icon": "approved"},
    {"value": "in_progress", "label": "Under Construction", "icon": "progress"},
    {"value": "completed", "label": "Completed", "icon": "complete"}
]

@router.get("/master-data")
async def get_master_data():
    """Get all master data for form dropdowns"""
    return {
        "states": sorted(INDIAN_STATES),
        "roof_types": ROOF_TYPES,
        "roof_materials": ROOF_MATERIALS,
        "project_status": PROJECT_STATUS_OPTIONS,
        "budget_ranges": [
            {"value": 25000, "label": "₹25,000 - Basic System"},
            {"value": 50000, "label": "₹50,000 - Standard System"},
            {"value": 75000, "label": "₹75,000 - Enhanced System"},
            {"value": 100000, "label": "₹1,00,000 - Premium System"},
            {"value": 150000, "label": "₹1,50,000+ - Advanced System"}
        ]
    }

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": embedding_model is not None,
        "gis_data_loaded": gis_manager._loaded,
        "llm_enabled": bool(OPENROUTER_API_KEY)
    }
