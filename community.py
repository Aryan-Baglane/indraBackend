"""
INDRA Community API - Social Feed for Water Conservation
Firebase-backed community features with AIML clustering
Members are called "droplets"
Clean, minimalistic, no gamification
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
import sys
import os
import math

# Import config
from config import FIREBASE_SERVICE_ACCOUNT_PATH

# Add models directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from models.load_model import (
    initialize_clustering,
    train_clustering,
    predict_cluster,
    get_cluster_info,
    get_nearby_droplets
)

router = APIRouter(prefix="/api/community", tags=["community"])

# ==================== FIREBASE INITIALIZATION ====================

# Initialize Firebase Admin SDK (if not already initialized)
db = None
try:
    firebase_admin.get_app()
    db = firestore.client()
except ValueError:
    # Initialize with default credentials or service account
    if os.path.exists(FIREBASE_SERVICE_ACCOUNT_PATH):
        try:
            cred = credentials.Certificate(FIREBASE_SERVICE_ACCOUNT_PATH)
            firebase_admin.initialize_app(cred)
            db = firestore.client()
        except Exception as e:
            print(f"⚠ Firebase initialization error: {e}")
    else:
        print("⚠ Firebase service account not found. Community features will be limited.")
        print(f"  Set FIREBASE_SERVICE_ACCOUNT_PATH env var or create the file")
except Exception as e:
    print(f"⚠ Firebase client error: {e}")

# Collection names
USERS_COLLECTION = 'users'
POSTS_COLLECTION = 'community_posts'
COMMENTS_COLLECTION = 'post_comments'


# ==================== PYDANTIC MODELS ====================

class DropletData(BaseModel):
    """User/Droplet data model"""
    uid: str
    name: str
    email: Optional[str] = None
    state: str
    district: str
    n_members: int = 4
    catchment_area: float = 100
    farmland_area: float = 0
    roof_type: str = "Flat"
    roof_material: str = "RCC"
    budget: float = 50000
    avg_rainfall: Optional[float] = 800
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    bio: Optional[str] = None


class CreatePostRequest(BaseModel):
    """Request to create a new post"""
    author_uid: str
    author_name: str
    content: str
    post_type: str = "general"  # general, question, tip, event, achievement
    district: str
    state: str
    tags: List[str] = []


class CommentRequest(BaseModel):
    """Request to add a comment"""
    author_uid: str
    author_name: str
    content: str


class ClusterDropletsRequest(BaseModel):
    """Request to cluster droplets"""
    droplets: List[DropletData]


class NearbyDropletsRequest(BaseModel):
    """Request for nearby droplets"""
    uid: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    state: Optional[str] = None
    district: Optional[str] = None
    max_distance: float = 100  # km
    limit: int = 20


# ==================== HELPER FUNCTIONS ====================

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two coordinates using Haversine formula"""
    R = 6371  # Earth's radius in km
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2) ** 2 + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
         math.sin(d_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def serialize_firestore_doc(doc) -> Dict[str, Any]:
    """Convert Firestore document to dict with proper datetime handling"""
    data = doc.to_dict()
    data['id'] = doc.id
    
    # Convert timestamps to ISO strings
    for key, value in data.items():
        if hasattr(value, 'isoformat'):
            data[key] = value.isoformat()
        elif hasattr(value, 'timestamp'):
            data[key] = datetime.fromtimestamp(value.timestamp()).isoformat()
    
    return data


# ==================== HELPER FUNCTION ====================

def check_db():
    """Check if Firebase is initialized"""
    if db is None:
        raise HTTPException(
            status_code=503, 
            detail="Firebase not configured. Please set up firebase-service-account.json"
        )


# ==================== FEED ENDPOINTS ====================

@router.get("/feed")
async def get_feed(
    state: Optional[str] = None,
    district: Optional[str] = None,
    post_type: Optional[str] = None,
    limit_count: int = Query(default=50, le=100, alias="limit")
):
    """Get community feed posts from Firestore"""
    check_db()
    try:
        posts_ref = db.collection(POSTS_COLLECTION)
        query = posts_ref.order_by('createdAt', direction=firestore.Query.DESCENDING).limit(limit_count)
        
        docs = query.stream()
        posts = []
        
        for doc in docs:
            post = serialize_firestore_doc(doc)
            
            # Apply filters in memory (Firestore limitations)
            if state and post.get('authorState', '').lower() != state.lower():
                continue
            if district and post.get('authorDistrict', '').lower() != district.lower():
                continue
            if post_type and post_type != 'all' and post.get('postType') != post_type:
                continue
            
            # Normalize field names for frontend
            posts.append({
                "id": post.get('id'),
                "author_uid": post.get('authorUid'),
                "author_name": post.get('authorName'),
                "content": post.get('content'),
                "post_type": post.get('postType', 'general'),
                "district": post.get('authorDistrict'),
                "state": post.get('authorState'),
                "cluster_id": post.get('clusterId'),
                "likes": post.get('likeCount', 0),
                "comments": post.get('commentCount', 0),
                "created_at": post.get('createdAt'),
                "tags": post.get('tags', []),
                "liked_by": post.get('likes', [])
            })
        
        return {
            "posts": posts,
            "total": len(posts),
            "filters": {"state": state, "district": district, "post_type": post_type}
        }
    except Exception as e:
        print(f"Error fetching feed: {e}")
        # Return empty feed on error
        return {"posts": [], "total": 0, "filters": {}}


@router.get("/post/create")
async def create_post(
    author_uid: str = Query(..., description="Author's UID"),
    author_name: str = Query(..., description="Author's name"),
    content: str = Query(..., description="Post content"),
    district: str = Query(..., description="Author's district"),
    state: str = Query(..., description="Author's state"),
    post_type: str = Query("general", description="Post type"),
    tags: str = Query("", description="Comma-separated tags")
):
    """Create a new community post in Firestore"""
    try:
        # Parse tags from comma-separated string
        tags_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
        
        # Predict cluster for the post author
        cluster_id = None
        try:
            initialize_clustering()
            cluster_id = predict_cluster({
                "state": state,
                "district": district
            })
        except:
            pass  # Clustering is optional
        
        post_data = {
            "authorUid": author_uid,
            "authorName": author_name,
            "content": content,
            "postType": post_type,
            "authorDistrict": district,
            "authorState": state,
            "tags": tags_list,
            "likes": [],
            "likeCount": 0,
            "commentCount": 0,
            "clusterId": cluster_id,
            "createdAt": firestore.SERVER_TIMESTAMP,
            "updatedAt": firestore.SERVER_TIMESTAMP
        }
        
        # Add to Firestore
        doc_ref = db.collection(POSTS_COLLECTION).add(post_data)
        post_id = doc_ref[1].id
        
        return {
            "success": True,
            "post": {
                "id": post_id,
                "author_uid": author_uid,
                "author_name": author_name,
                "content": content,
                "post_type": post_type,
                "district": district,
                "state": state,
                "likes": 0,
                "comments": 0,
                "created_at": datetime.now().isoformat(),
                "tags": tags_list
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create post: {str(e)}")


@router.get("/post/{post_id}/like")
async def toggle_like_post(post_id: str, user_uid: str = Query(...)):
    """Toggle like on a post"""
    try:
        post_ref = db.collection(POSTS_COLLECTION).document(post_id)
        post_doc = post_ref.get()
        
        if not post_doc.exists:
            raise HTTPException(status_code=404, detail="Post not found")
        
        post_data = post_doc.to_dict()
        likes = post_data.get('likes', [])
        
        if user_uid in likes:
            # Unlike
            post_ref.update({
                'likes': firestore.ArrayRemove([user_uid]),
                'likeCount': firestore.Increment(-1),
                'updatedAt': firestore.SERVER_TIMESTAMP
            })
            return {"success": True, "liked": False, "likes": post_data.get('likeCount', 1) - 1}
        else:
            # Like
            post_ref.update({
                'likes': firestore.ArrayUnion([user_uid]),
                'likeCount': firestore.Increment(1),
                'updatedAt': firestore.SERVER_TIMESTAMP
            })
            return {"success": True, "liked": True, "likes": post_data.get('likeCount', 0) + 1}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to toggle like: {str(e)}")


@router.get("/post/{post_id}/delete")
async def delete_post(post_id: str, user_uid: str = Query(...)):
    """Delete a post (only by author)"""
    try:
        post_ref = db.collection(POSTS_COLLECTION).document(post_id)
        post_doc = post_ref.get()
        
        if not post_doc.exists:
            raise HTTPException(status_code=404, detail="Post not found")
        
        post_data = post_doc.to_dict()
        if post_data.get('authorUid') != user_uid:
            raise HTTPException(status_code=403, detail="Not authorized to delete this post")
        
        post_ref.delete()
        return {"success": True, "message": "Post deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete post: {str(e)}")


# ==================== COMMENT ENDPOINTS ====================

@router.get("/post/{post_id}/comments")
async def get_comments(post_id: str):
    """Get comments for a post"""
    try:
        comments_ref = db.collection(COMMENTS_COLLECTION)
        query = comments_ref.where('postId', '==', post_id).order_by('createdAt')
        
        docs = query.stream()
        comments = []
        
        for doc in docs:
            comment = serialize_firestore_doc(doc)
            comments.append({
                "id": comment.get('id'),
                "author_uid": comment.get('authorUid'),
                "author_name": comment.get('authorName'),
                "content": comment.get('content'),
                "created_at": comment.get('createdAt')
            })
        
        return {"comments": comments, "total": len(comments)}
    except Exception as e:
        return {"comments": [], "total": 0}


@router.get("/post/{post_id}/comment/add")
async def add_comment(
    post_id: str,
    author_uid: str = Query(..., description="Author's UID"),
    author_name: str = Query(..., description="Author's name"),
    content: str = Query(..., description="Comment content")
):
    """Add a comment to a post"""
    try:
        # Verify post exists
        post_ref = db.collection(POSTS_COLLECTION).document(post_id)
        post_doc = post_ref.get()
        
        if not post_doc.exists:
            raise HTTPException(status_code=404, detail="Post not found")
        
        # Add comment
        comment_data = {
            "postId": post_id,
            "authorUid": author_uid,
            "authorName": author_name,
            "content": content,
            "createdAt": firestore.SERVER_TIMESTAMP
        }
        
        doc_ref = db.collection(COMMENTS_COLLECTION).add(comment_data)
        
        # Increment comment count on post
        post_ref.update({
            'commentCount': firestore.Increment(1),
            'updatedAt': firestore.SERVER_TIMESTAMP
        })
        
        return {
            "success": True,
            "comment": {
                "id": doc_ref[1].id,
                "author_uid": author_uid,
                "author_name": author_name,
                "content": content,
                "created_at": datetime.now().isoformat()
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add comment: {str(e)}")


# ==================== NEARBY DROPLETS ENDPOINTS ====================

@router.get("/nearby")
async def get_nearby(
    uid: str = Query(..., description="User's UID"),
    latitude: Optional[float] = Query(None, description="User's latitude"),
    longitude: Optional[float] = Query(None, description="User's longitude"),
    state: Optional[str] = Query(None, description="User's state"),
    district: Optional[str] = Query(None, description="User's district"),
    max_distance: float = Query(100, description="Maximum distance in km"),
    limit: int = Query(20, description="Maximum number of results")
):
    """Get nearby droplets based on location using AIML clustering"""
    try:
        users_ref = db.collection(USERS_COLLECTION)
        docs = users_ref.stream()
        
        droplets = []
        current_user_data = None
        
        for doc in docs:
            data = doc.to_dict()
            if data.get('uid') == uid:
                current_user_data = data
                continue  # Skip current user
            
            droplet = {
                "uid": data.get('uid'),
                "name": data.get('name', 'Anonymous'),
                "state": data.get('state', ''),
                "district": data.get('district', ''),
                "n_members": data.get('n_members', 0),
                "catchment_area": data.get('catchment_area', 0),
                "farmland_area": data.get('farmland_area', 0),
                "budget": data.get('budget', 0),
                "latitude": data.get('latitude'),
                "longitude": data.get('longitude'),
                "bio": data.get('bio', '')
            }
            
            # Calculate distance if coordinates available
            if (latitude and longitude and 
                data.get('latitude') and data.get('longitude')):
                distance_val = calculate_distance(
                    latitude, longitude,
                    data['latitude'], data['longitude']
                )
                droplet['distance'] = round(distance_val, 1)
                
                # Filter by max distance
                if distance_val > max_distance:
                    continue
            else:
                # Fallback: filter by state
                if state and data.get('state', '').lower() != state.lower():
                    continue
            
            droplets.append(droplet)
        
        # Use AIML clustering if available
        try:
            initialize_clustering()
            if current_user_data or state:
                user_data = current_user_data or {
                    "state": state,
                    "district": district
                }
                droplets_with_clusters = get_nearby_droplets(
                    user_data,
                    droplets,
                    limit
                )
                if droplets_with_clusters:
                    droplets = droplets_with_clusters
        except:
            pass  # Fall back to distance-based sorting
        
        # Sort by distance (closest first)
        droplets.sort(key=lambda x: (
            x.get('distance', 9999),
            0 if x.get('district', '').lower() == (district or '').lower() else 1
        ))
        
        return {
            "droplets": droplets[:limit],
            "total": len(droplets),
            "has_location": bool(latitude and longitude)
        }
    except Exception as e:
        print(f"Error fetching nearby droplets: {e}")
        return {"droplets": [], "total": 0, "has_location": False}


@router.get("/nearby/{district}")
async def get_nearby_by_district(
    district: str, 
    state: Optional[str] = None,
    exclude_uid: Optional[str] = None,
    limit_count: int = Query(default=20, alias="limit")
):
    """Get droplets in a specific district"""
    try:
        users_ref = db.collection(USERS_COLLECTION)
        query = users_ref.where('district', '==', district)
        
        docs = query.stream()
        droplets = []
        
        for doc in docs:
            data = doc.to_dict()
            
            # Skip excluded user
            if exclude_uid and data.get('uid') == exclude_uid:
                continue
            
            # Filter by state if provided
            if state and data.get('state', '').lower() != state.lower():
                continue
            
            droplets.append({
                "uid": data.get('uid'),
                "name": data.get('name', 'Anonymous'),
                "state": data.get('state', ''),
                "district": data.get('district', ''),
                "n_members": data.get('n_members', 0),
                "catchment_area": data.get('catchment_area', 0),
                "farmland_area": data.get('farmland_area', 0),
                "budget": data.get('budget', 0),
                "bio": data.get('bio', '')
            })
        
        return {
            "droplets": droplets[:limit_count],
            "total": len(droplets),
            "district": district,
            "state": state
        }
    except Exception as e:
        return {"droplets": [], "total": 0, "district": district, "state": state}


# ==================== CLUSTERING ENDPOINTS ====================

@router.get("/clusters")
async def get_clusters():
    """Get all cluster information from AIML model"""
    try:
        initialize_clustering()
        return {
            "clusters": get_cluster_info(),
            "total_clusters": len(get_cluster_info())
        }
    except Exception as e:
        return {"clusters": {}, "total_clusters": 0}


@router.get("/clusters/train")
async def train_clusters():
    """Train clustering model with droplet data from Firestore"""
    try:
        droplets_dict = []
        
        # Fetch droplets from Firestore
        users_ref = db.collection(USERS_COLLECTION).limit(50)
        docs = users_ref.stream()
        for doc in docs:
            data = doc.to_dict()
            droplets_dict.append({
                "uid": data.get('uid'),
                "name": data.get('name'),
                "state": data.get('state'),
                "district": data.get('district'),
                "n_members": data.get('n_members', 4),
                "catchment_area": data.get('catchment_area', 100),
                "farmland_area": data.get('farmland_area', 0),
                "budget": data.get('budget', 50000),
                "avg_rainfall": data.get('avg_rainfall', 800)
            })
        
        result = train_clustering(droplets_dict)
        
        return {
            "success": True,
            "n_clusters": result.get("n_clusters"),
            "cluster_info": result.get("cluster_info"),
            "message": f"Model trained with {len(droplets_dict)} droplets"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/clusters/predict")
async def predict_droplet_cluster(
    uid: str = Query(..., description="User's UID"),
    name: str = Query(..., description="User's name"),
    state: str = Query(..., description="User's state"),
    district: str = Query(..., description="User's district"),
    n_members: int = Query(4, description="Number of household members"),
    catchment_area: float = Query(100, description="Catchment area"),
    farmland_area: float = Query(0, description="Farmland area"),
    roof_type: str = Query("Flat", description="Roof type"),
    roof_material: str = Query("RCC", description="Roof material"),
    budget: float = Query(50000, description="Budget"),
    avg_rainfall: Optional[float] = Query(800, description="Average rainfall"),
    latitude: Optional[float] = Query(None, description="Latitude"),
    longitude: Optional[float] = Query(None, description="Longitude"),
    email: Optional[str] = Query(None, description="Email"),
    bio: Optional[str] = Query(None, description="Bio")
):
    """Predict which cluster a droplet belongs to"""
    try:
        initialize_clustering()
        droplet_data = {
            "uid": uid,
            "name": name,
            "email": email,
            "state": state,
            "district": district,
            "n_members": n_members,
            "catchment_area": catchment_area,
            "farmland_area": farmland_area,
            "roof_type": roof_type,
            "roof_material": roof_material,
            "budget": budget,
            "avg_rainfall": avg_rainfall,
            "latitude": latitude,
            "longitude": longitude,
            "bio": bio
        }
        cluster_id = predict_cluster(droplet_data)
        cluster_info = get_cluster_info()
        
        return {
            "cluster_id": cluster_id,
            "cluster_info": cluster_info.get(str(cluster_id), {}),
            "droplet_uid": uid
        }
    except Exception as e:
        return {"cluster_id": 0, "cluster_info": {}, "droplet_uid": uid}


# ==================== STATS ENDPOINT ====================

@router.get("/stats")
async def get_community_stats():
    """Get community statistics from Firestore"""
    try:
        # Get users count and data
        users_ref = db.collection(USERS_COLLECTION)
        users_docs = list(users_ref.stream())
        
        # Get posts count
        posts_ref = db.collection(POSTS_COLLECTION)
        posts_docs = list(posts_ref.stream())
        
        # Calculate stats
        states = set()
        districts = set()
        total_water_potential = 0
        
        for doc in users_docs:
            data = doc.to_dict()
            if data.get('state'):
                states.add(data['state'])
            if data.get('district'):
                districts.add(data['district'])
            
            # Water potential calculation
            rainfall = data.get('avg_rainfall', 800)
            catchment = data.get('catchment_area', 0)
            total_water_potential += catchment * rainfall * 0.8 / 1000  # KL
        
        # Get cluster info
        try:
            initialize_clustering()
            cluster_info = get_cluster_info()
            n_clusters = len(cluster_info)
        except:
            n_clusters = max(1, len(users_docs) // 10)
        
        return {
            "total_droplets": len(users_docs),
            "total_clusters": n_clusters,
            "total_posts": len(posts_docs),
            "total_water_potential_kl": round(total_water_potential, 2),
            "active_states": list(states),
            "active_districts": list(districts)
        }
    except Exception as e:
        print(f"Error fetching stats: {e}")
        return {
            "total_droplets": 0,
            "total_clusters": 0,
            "total_posts": 0,
            "total_water_potential_kl": 0,
            "active_states": [],
            "active_districts": []
        }
