"""
Firebase Admin Backend for INDRA
Handles server-side authentication verification and database operations
"""

import firebase_admin
from firebase_admin import credentials, firestore, auth
from fastapi import APIRouter, HTTPException, Depends, Header, Query
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime
import os

# Import config
from config import FIREBASE_SERVICE_ACCOUNT_PATH

router = APIRouter(prefix="/api/users", tags=["users"])

# Initialize Firebase Admin SDK
# Check if Firebase is already initialized
if not firebase_admin._apps:
    try:
        if os.path.exists(FIREBASE_SERVICE_ACCOUNT_PATH):
            cred = credentials.Certificate(FIREBASE_SERVICE_ACCOUNT_PATH)
            firebase_admin.initialize_app(cred)
            print("✓ Firebase Admin SDK initialized successfully")
        else:
            # Initialize without credentials (for development)
            firebase_admin.initialize_app()
            print("⚠ Firebase Admin SDK initialized with default credentials")
    except Exception as e:
        print(f"⚠ Firebase Admin SDK initialization warning: {e}")

# Get Firestore client
try:
    db = firestore.client()
except Exception as e:
    db = None
    print(f"⚠ Firestore client warning: {e}")


# ==================== PYDANTIC MODELS ====================

class UserProfile(BaseModel):
    uid: str
    email: EmailStr
    name: str
    state: str
    district: str
    n_members: int
    catchment_area: float
    farmland_area: float
    roof_type: str
    roof_material: str
    budget: float

class UserProfileUpdate(BaseModel):
    name: Optional[str] = None
    state: Optional[str] = None
    district: Optional[str] = None
    n_members: Optional[int] = None
    catchment_area: Optional[float] = None
    farmland_area: Optional[float] = None
    roof_type: Optional[str] = None
    roof_material: Optional[str] = None
    budget: Optional[float] = None

class AssessmentCreate(BaseModel):
    user_id: str
    name: str
    state: str
    district: str
    pincode: str
    n_members: int
    catchment_area: float
    farmland_area: float
    roof_type: str
    roof_material: str
    budget: float
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    rwh_type: str
    avg_rainfall: float
    cost: float
    project_status: int = 1
    feasibility_score: float
    annual_harvestable_water: float
    recommended_storage_capacity: float
    water_self_sufficiency_days: int
    recommendations: List[str] = []

class AssessmentResponse(BaseModel):
    id: str
    user_id: str
    name: str
    state: str
    district: str
    rwh_type: str
    cost: float
    project_status: int
    feasibility_score: float
    annual_harvestable_water: float
    created_at: Optional[str] = None


# ==================== AUTH HELPERS ====================

async def verify_firebase_token(authorization: str = Header(None)) -> dict:
    """
    Verify Firebase ID token from Authorization header
    Returns decoded token with user info
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    try:
        # Extract token from "Bearer <token>"
        if authorization.startswith("Bearer "):
            token = authorization[7:]
        else:
            token = authorization
        
        # Verify the token
        decoded_token = auth.verify_id_token(token)
        return decoded_token
    except auth.InvalidIdTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except auth.ExpiredIdTokenError:
        raise HTTPException(status_code=401, detail="Token expired")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication failed: {str(e)}")


# ==================== USER ENDPOINTS ====================

@router.get("/profile/{uid}")
async def get_user_profile(uid: str):
    """Get user profile by UID"""
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    try:
        doc_ref = db.collection('users').document(uid)
        doc = doc_ref.get()
        
        if doc.exists:
            data = doc.to_dict()
            return {"success": True, "profile": data}
        else:
            raise HTTPException(status_code=404, detail="User not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/profile/{uid}/update")
async def update_user_profile(
    uid: str,
    name: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    district: Optional[str] = Query(None),
    n_members: Optional[int] = Query(None),
    catchment_area: Optional[float] = Query(None),
    farmland_area: Optional[float] = Query(None),
    roof_type: Optional[str] = Query(None),
    roof_material: Optional[str] = Query(None),
    budget: Optional[float] = Query(None)
):
    """Update user profile"""
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    try:
        doc_ref = db.collection('users').document(uid)
        
        # Build update data from non-None query parameters
        update_data = {}
        if name is not None:
            update_data['name'] = name
        if state is not None:
            update_data['state'] = state
        if district is not None:
            update_data['district'] = district
        if n_members is not None:
            update_data['n_members'] = n_members
        if catchment_area is not None:
            update_data['catchment_area'] = catchment_area
        if farmland_area is not None:
            update_data['farmland_area'] = farmland_area
        if roof_type is not None:
            update_data['roof_type'] = roof_type
        if roof_material is not None:
            update_data['roof_material'] = roof_material
        if budget is not None:
            update_data['budget'] = budget
        
        update_data['updatedAt'] = firestore.SERVER_TIMESTAMP
        
        doc_ref.update(update_data)
        return {"success": True, "message": "Profile updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== ASSESSMENT ENDPOINTS ====================

@router.get("/assessments/create")
async def create_assessment(
    user_id: str = Query(...),
    name: str = Query(...),
    state: str = Query(...),
    district: str = Query(...),
    pincode: str = Query(...),
    n_members: int = Query(...),
    catchment_area: float = Query(...),
    farmland_area: float = Query(...),
    roof_type: str = Query(...),
    roof_material: str = Query(...),
    budget: float = Query(...),
    rwh_type: str = Query(...),
    avg_rainfall: float = Query(...),
    cost: float = Query(...),
    feasibility_score: float = Query(...),
    annual_harvestable_water: float = Query(...),
    recommended_storage_capacity: float = Query(...),
    water_self_sufficiency_days: int = Query(...),
    latitude: Optional[float] = Query(None),
    longitude: Optional[float] = Query(None),
    project_status: int = Query(1),
    recommendations: str = Query("", description="Comma-separated recommendations")
):
    """Create a new assessment"""
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    try:
        # Parse recommendations from comma-separated string
        recommendations_list = [r.strip() for r in recommendations.split(",") if r.strip()] if recommendations else []
        
        assessment_data = {
            'user_id': user_id,
            'name': name,
            'state': state,
            'district': district,
            'pincode': pincode,
            'n_members': n_members,
            'catchment_area': catchment_area,
            'farmland_area': farmland_area,
            'roof_type': roof_type,
            'roof_material': roof_material,
            'budget': budget,
            'latitude': latitude,
            'longitude': longitude,
            'rwh_type': rwh_type,
            'avg_rainfall': avg_rainfall,
            'cost': cost,
            'project_status': project_status,
            'feasibility_score': feasibility_score,
            'annual_harvestable_water': annual_harvestable_water,
            'recommended_storage_capacity': recommended_storage_capacity,
            'water_self_sufficiency_days': water_self_sufficiency_days,
            'recommendations': recommendations_list,
            'createdAt': firestore.SERVER_TIMESTAMP,
            'updatedAt': firestore.SERVER_TIMESTAMP
        }
        
        doc_ref = db.collection('assessments').document()
        doc_ref.set(assessment_data)
        
        return {
            "success": True,
            "assessment_id": doc_ref.id,
            "message": "Assessment created successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/assessments/{user_id}")
async def get_user_assessments(user_id: str):
    """Get all assessments for a user"""
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    try:
        assessments_ref = db.collection('assessments')
        query = assessments_ref.where('user_id', '==', user_id).order_by('createdAt', direction=firestore.Query.DESCENDING)
        docs = query.stream()
        
        assessments = []
        for doc in docs:
            data = doc.to_dict()
            data['id'] = doc.id
            # Convert timestamps to strings
            if 'createdAt' in data and data['createdAt']:
                data['createdAt'] = data['createdAt'].isoformat() if hasattr(data['createdAt'], 'isoformat') else str(data['createdAt'])
            assessments.append(data)
        
        return {"success": True, "assessments": assessments}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/assessments/{assessment_id}/status")
async def toggle_assessment_status(assessment_id: str, status: int = Query(...)):
    """Toggle project status (0 or 1)"""
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    if status not in [0, 1]:
        raise HTTPException(status_code=400, detail="Status must be 0 or 1")
    
    try:
        doc_ref = db.collection('assessments').document(assessment_id)
        doc_ref.update({
            'project_status': status,
            'updatedAt': firestore.SERVER_TIMESTAMP
        })
        return {"success": True, "message": "Status updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/assessments/{assessment_id}/delete")
async def delete_assessment(assessment_id: str):
    """Delete an assessment"""
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    try:
        doc_ref = db.collection('assessments').document(assessment_id)
        doc_ref.delete()
        return {"success": True, "message": "Assessment deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== STATS ENDPOINTS ====================

@router.get("/stats/{user_id}")
async def get_user_stats(user_id: str):
    """Get user statistics"""
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    try:
        assessments_ref = db.collection('assessments')
        query = assessments_ref.where('user_id', '==', user_id)
        docs = list(query.stream())
        
        total_assessments = len(docs)
        active_projects = sum(1 for doc in docs if doc.to_dict().get('project_status') == 1)
        total_water = sum(doc.to_dict().get('annual_harvestable_water', 0) for doc in docs)
        total_cost = sum(doc.to_dict().get('cost', 0) for doc in docs if doc.to_dict().get('project_status') == 1)
        
        return {
            "success": True,
            "stats": {
                "total_assessments": total_assessments,
                "active_projects": active_projects,
                "total_harvestable_water": total_water,
                "total_investment": total_cost
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/community/{state}")
async def get_community_data(state: str):
    """Get community data by state (for Gramin dashboard clustering)"""
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    try:
        assessments_ref = db.collection('assessments')
        query = assessments_ref.where('state', '==', state).where('project_status', '==', 1)
        docs = query.stream()
        
        community_data = []
        for doc in docs:
            data = doc.to_dict()
            community_data.append({
                "id": doc.id,
                "district": data.get('district'),
                "latitude": data.get('latitude'),
                "longitude": data.get('longitude'),
                "rwh_type": data.get('rwh_type'),
                "annual_harvestable_water": data.get('annual_harvestable_water', 0)
            })
        
        return {"success": True, "community": community_data, "count": len(community_data)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
