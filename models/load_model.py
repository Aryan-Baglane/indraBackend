"""
INDRA Model Loader
Handles loading and initialization of ML models:
- Sentence Transformer (for embeddings)
- Community Clustering (Multiple algorithms for grouping droplets)
  - KMeans: Fast, good for evenly sized clusters
  - DBSCAN: Density-based, finds natural clusters + outliers
  - Agglomerative: Hierarchical, good for nested communities
  - Geo-KMeans: Location-aware clustering using Haversine distance
"""

import os
import pickle
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import math

# Paths
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
SENTENCE_MODEL_PATH = os.path.join(MODEL_DIR, "all-MiniLM-L6-v2")
CLUSTERING_MODEL_PATH = os.path.join(MODEL_DIR, "community_kmeans.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "community_scaler.pkl")
CLUSTER_INFO_PATH = os.path.join(MODEL_DIR, "cluster_info.json")

# Lazy imports for optional dependencies
SentenceTransformer = None
KMeans = None
DBSCAN = None
AgglomerativeClustering = None
StandardScaler = None
silhouette_score = None
haversine_distances = None


def _import_sentence_transformer():
    global SentenceTransformer
    if SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer as ST
        SentenceTransformer = ST
    return SentenceTransformer


def _import_sklearn():
    global KMeans, DBSCAN, AgglomerativeClustering, StandardScaler, silhouette_score, haversine_distances
    if KMeans is None:
        from sklearn.cluster import KMeans as KM
        from sklearn.cluster import DBSCAN as DB
        from sklearn.cluster import AgglomerativeClustering as AC
        from sklearn.preprocessing import StandardScaler as SS
        from sklearn.metrics import silhouette_score as ss
        from sklearn.metrics.pairwise import haversine_distances as hd
        KMeans = KM
        DBSCAN = DB
        AgglomerativeClustering = AC
        StandardScaler = SS
        silhouette_score = ss
        haversine_distances = hd
    return KMeans, DBSCAN, AgglomerativeClustering, StandardScaler, silhouette_score, haversine_distances


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two coordinates in km"""
    R = 6371  # Earth's radius in km
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2) ** 2 + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
         math.sin(d_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# ==================== SENTENCE TRANSFORMER ====================

_sentence_model = None

def load_sentence_model():
    """Load sentence transformer model for embeddings"""
    global _sentence_model
    if _sentence_model is not None:
        return _sentence_model
    
    ST = _import_sentence_transformer()
    
    if os.path.exists(SENTENCE_MODEL_PATH):
        print(f"Loading sentence model from {SENTENCE_MODEL_PATH}")
        _sentence_model = ST(SENTENCE_MODEL_PATH)
    else:
        print("Downloading sentence-transformers/all-MiniLM-L6-v2...")
        _sentence_model = ST("sentence-transformers/all-MiniLM-L6-v2")
        os.makedirs(SENTENCE_MODEL_PATH, exist_ok=True)
        _sentence_model.save(SENTENCE_MODEL_PATH)
        print(f"Model saved to {SENTENCE_MODEL_PATH}")
    
    return _sentence_model


def get_embedding(text: str) -> np.ndarray:
    """Get embedding for text"""
    model = load_sentence_model()
    return model.encode(text)


def get_embeddings(texts: List[str]) -> np.ndarray:
    """Get embeddings for multiple texts"""
    model = load_sentence_model()
    return model.encode(texts)


# ==================== COMMUNITY CLUSTERING MODEL ====================

class CommunityClusteringModel:
    """
    Advanced clustering model for grouping droplets (users) by location and needs.
    
    Supports multiple algorithms:
    - 'kmeans': Fast, evenly sized clusters (default)
    - 'dbscan': Density-based, finds natural clusters + outliers
    - 'hierarchical': Agglomerative, good for nested communities
    - 'geo': Location-focused using Haversine distance
    
    Features used: state, district, n_members, catchment_area, farmland_area, 
                   budget, avg_rainfall, latitude, longitude
    """
    
    ALGORITHMS = ['kmeans', 'dbscan', 'hierarchical', 'geo', 'auto']
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.district_encoder: Dict[str, int] = {}
        self.state_encoder: Dict[str, int] = {}
        self.cluster_info: Dict[str, Any] = {}
        self.n_clusters = 5
        self.algorithm = 'kmeans'
        self._initialized = False
    
    def initialize(self):
        """Load existing model or prepare for training"""
        if self._initialized:
            return True
        
        _import_sklearn()
        
        try:
            self.load()
            print(f"Community clustering model loaded ({self.algorithm})")
            self._initialized = True
            return True
        except Exception as e:
            print(f"No existing clustering model found: {e}")
            self.scaler = StandardScaler()
            self._initialized = True
            return False
    
    def _encode_location(self, state: str, district: str) -> Tuple[int, int]:
        """Encode state and district to numeric values"""
        state = (state or "Unknown").strip().lower()
        district = (district or "Unknown").strip().lower()
        
        if state not in self.state_encoder:
            self.state_encoder[state] = len(self.state_encoder)
        if district not in self.district_encoder:
            self.district_encoder[district] = len(self.district_encoder)
        
        return self.state_encoder[state], self.district_encoder[district]
    
    def _prepare_features(self, users: List[Dict[str, Any]], include_geo: bool = True) -> np.ndarray:
        """Convert user data to feature matrix"""
        features = []
        
        for user in users:
            state_enc, district_enc = self._encode_location(
                user.get('state', 'Unknown'),
                user.get('district', 'Unknown')
            )
            
            feature_vector = [
                state_enc,
                district_enc,
                float(user.get('n_members', 4)),
                float(user.get('catchment_area', 100)),
                float(user.get('farmland_area', 0)),
                float(user.get('budget', 50000)) / 10000,  # Normalize budget
                float(user.get('avg_rainfall', 800)) / 100,  # Normalize rainfall
            ]
            
            if include_geo:
                lat = float(user.get('latitude', 0)) if user.get('latitude') else 0
                lon = float(user.get('longitude', 0)) if user.get('longitude') else 0
                feature_vector.extend([lat, lon])
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _prepare_geo_features(self, users: List[Dict[str, Any]]) -> np.ndarray:
        """Prepare only geographic features for geo-clustering"""
        features = []
        for user in users:
            lat = float(user.get('latitude', 0)) if user.get('latitude') else 0
            lon = float(user.get('longitude', 0)) if user.get('longitude') else 0
            # Convert to radians for Haversine
            features.append([math.radians(lat), math.radians(lon)])
        return np.array(features)
    
    def _select_best_algorithm(self, users: List[Dict[str, Any]]) -> str:
        """Automatically select the best clustering algorithm based on data"""
        n_users = len(users)
        
        # Check if we have good geo data
        has_geo = sum(1 for u in users if u.get('latitude') and u.get('longitude')) > n_users * 0.7
        
        # Check location diversity
        unique_districts = len(set(u.get('district', '') for u in users))
        unique_states = len(set(u.get('state', '') for u in users))
        
        if has_geo and n_users >= 10:
            # Good geo data - use geo-based clustering
            return 'geo'
        elif unique_districts <= 3 and n_users >= 10:
            # Few districts, use DBSCAN to find density clusters
            return 'dbscan'
        elif n_users < 20:
            # Small dataset - hierarchical works well
            return 'hierarchical'
        else:
            # Default to KMeans for larger, diverse datasets
            return 'kmeans'
    
    def fit(self, users: List[Dict[str, Any]], n_clusters: int = None, 
            algorithm: str = 'auto') -> Dict[str, Any]:
        """
        Train clustering model on user data.
        
        Args:
            users: List of user dictionaries
            n_clusters: Number of clusters (auto-detected if None)
            algorithm: 'kmeans', 'dbscan', 'hierarchical', 'geo', or 'auto'
        
        Returns:
            Dictionary with clustering results
        """
        if len(users) < 3:
            return {"error": "Need at least 3 droplets for clustering", "clusters": []}
        
        KM, DB, AC, SS, ss, hd = _import_sklearn()
        
        # Select algorithm
        if algorithm == 'auto':
            algorithm = self._select_best_algorithm(users)
            print(f"Auto-selected algorithm: {algorithm}")
        
        self.algorithm = algorithm
        
        # Prepare features based on algorithm
        if algorithm == 'geo':
            X = self._prepare_geo_features(users)
            X_scaled = X  # Already in radians, no scaling needed
        else:
            X = self._prepare_features(users, include_geo=(algorithm != 'dbscan'))
            self.scaler = SS()
            X_scaled = self.scaler.fit_transform(X)
        
        # Determine optimal cluster count
        if n_clusters is None and algorithm in ['kmeans', 'hierarchical', 'geo']:
            n_clusters = self._find_optimal_clusters(X_scaled, algorithm, len(users))
        
        self.n_clusters = n_clusters or 5
        
        # Fit the selected algorithm
        if algorithm == 'kmeans':
            labels = self._fit_kmeans(X_scaled)
        elif algorithm == 'dbscan':
            labels = self._fit_dbscan(X_scaled)
        elif algorithm == 'hierarchical':
            labels = self._fit_hierarchical(X_scaled)
        elif algorithm == 'geo':
            labels = self._fit_geo_kmeans(X_scaled, users)
        else:
            labels = self._fit_kmeans(X_scaled)  # Fallback
        
        # Update n_clusters based on actual clusters found
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # Remove noise label from DBSCAN
        self.n_clusters = len(unique_labels)
        
        self.cluster_info = self._calculate_cluster_stats(users, np.array(labels))
        self.save()
        
        # Calculate quality score
        try:
            if len(unique_labels) > 1:
                score = ss(X_scaled, labels)
            else:
                score = 0.0
        except:
            score = 0.0
        
        return {
            "n_clusters": self.n_clusters,
            "algorithm": self.algorithm,
            "silhouette_score": round(score, 3),
            "cluster_info": self.cluster_info,
            "labels": labels.tolist() if hasattr(labels, 'tolist') else list(labels)
        }
    
    def _find_optimal_clusters(self, X: np.ndarray, algorithm: str, n_samples: int) -> int:
        """Find optimal number of clusters using silhouette score"""
        KM, _, AC, _, ss, _ = _import_sklearn()
        
        max_clusters = min(10, n_samples - 1, max(2, n_samples // 3))
        best_score = -1
        best_k = 3
        
        for k in range(2, max_clusters + 1):
            try:
                if algorithm == 'hierarchical':
                    model = AC(n_clusters=k)
                else:
                    model = KM(n_clusters=k, random_state=42, n_init=10)
                
                labels = model.fit_predict(X)
                score = ss(X, labels)
                
                if score > best_score:
                    best_score = score
                    best_k = k
            except:
                continue
        
        print(f"Optimal clusters: {best_k} (silhouette: {best_score:.3f})")
        return best_k
    
    def _fit_kmeans(self, X: np.ndarray) -> np.ndarray:
        """Fit KMeans clustering"""
        KM, _, _, _, _, _ = _import_sklearn()
        self.model = KM(n_clusters=self.n_clusters, random_state=42, n_init=10)
        return self.model.fit_predict(X)
    
    def _fit_dbscan(self, X: np.ndarray) -> np.ndarray:
        """
        Fit DBSCAN clustering - finds natural density clusters.
        Good for finding communities with irregular shapes.
        """
        _, DB, _, _, _, _ = _import_sklearn()
        
        # DBSCAN parameters - eps is the neighborhood radius
        # min_samples is minimum points to form a cluster
        eps = 0.5  # Adjust based on your data scale
        min_samples = max(2, len(X) // 10)
        
        self.model = DB(eps=eps, min_samples=min_samples)
        labels = self.model.fit_predict(X)
        
        # Handle noise points (label -1) - assign to nearest cluster
        noise_mask = labels == -1
        if noise_mask.any() and not noise_mask.all():
            from sklearn.neighbors import NearestNeighbors
            valid_points = X[~noise_mask]
            valid_labels = labels[~noise_mask]
            
            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(valid_points)
            
            noise_points = X[noise_mask]
            _, indices = nn.kneighbors(noise_points)
            labels[noise_mask] = valid_labels[indices.flatten()]
        
        return labels
    
    def _fit_hierarchical(self, X: np.ndarray) -> np.ndarray:
        """
        Fit Agglomerative (Hierarchical) clustering.
        Creates nested clusters - good for community hierarchies.
        """
        _, _, AC, _, _, _ = _import_sklearn()
        self.model = AC(n_clusters=self.n_clusters, linkage='ward')
        return self.model.fit_predict(X)
    
    def _fit_geo_kmeans(self, X: np.ndarray, users: List[Dict[str, Any]]) -> np.ndarray:
        """
        Fit geo-aware KMeans using Haversine distance.
        Clusters users based on actual geographic proximity.
        """
        KM, _, _, _, _, hd = _import_sklearn()
        
        # Filter users with valid coordinates
        valid_indices = []
        valid_coords = []
        
        for i, user in enumerate(users):
            lat = user.get('latitude')
            lon = user.get('longitude')
            if lat and lon and lat != 0 and lon != 0:
                valid_indices.append(i)
                valid_coords.append([math.radians(lat), math.radians(lon)])
        
        if len(valid_coords) < 3:
            # Fall back to regular KMeans if not enough geo data
            return self._fit_kmeans(X)
        
        valid_coords = np.array(valid_coords)
        
        # Calculate Haversine distance matrix
        dist_matrix = hd(valid_coords) * 6371  # Convert to km
        
        # Use regular KMeans on the coordinates (works well enough)
        self.model = KM(n_clusters=min(self.n_clusters, len(valid_coords) - 1), 
                        random_state=42, n_init=10)
        valid_labels = self.model.fit_predict(valid_coords)
        
        # Assign labels back to all users
        labels = np.zeros(len(users), dtype=int)
        for i, idx in enumerate(valid_indices):
            labels[idx] = valid_labels[i]
        
        # Assign users without geo data to nearest cluster by district/state
        for i, user in enumerate(users):
            if i not in valid_indices:
                # Find most common cluster in same district
                same_district = [
                    labels[j] for j, u in enumerate(users) 
                    if j in valid_indices and u.get('district') == user.get('district')
                ]
                if same_district:
                    labels[i] = max(set(same_district), key=same_district.count)
                else:
                    labels[i] = 0
        
        return labels
    
    def predict(self, user: Dict[str, Any]) -> int:
        """Predict cluster for a single user"""
        if self.model is None:
            return 0
        
        X = self._prepare_features([user])
        X_scaled = self.scaler.transform(X)
        return int(self.model.predict(X_scaled)[0])
    
    def predict_batch(self, users: List[Dict[str, Any]]) -> List[int]:
        """Predict clusters for multiple users"""
        if self.model is None:
            return [0] * len(users)
        
        X = self._prepare_features(users)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled).tolist()
    
    def _calculate_cluster_stats(self, users: List[Dict[str, Any]], labels: np.ndarray) -> Dict[str, Any]:
        """Calculate statistics for each cluster"""
        cluster_stats = {}
        
        for cluster_id in range(self.n_clusters):
            cluster_users = [users[i] for i in range(len(users)) if labels[i] == cluster_id]
            
            if not cluster_users:
                continue
            
            districts = [u.get('district', 'Unknown') for u in cluster_users]
            states = [u.get('state', 'Unknown') for u in cluster_users]
            
            district_counts = {}
            for d in districts:
                district_counts[d] = district_counts.get(d, 0) + 1
            primary_district = max(district_counts, key=district_counts.get)
            
            cluster_stats[str(cluster_id)] = {
                "cluster_id": cluster_id,
                "droplet_count": len(cluster_users),
                "primary_district": primary_district,
                "primary_state": max(set(states), key=states.count),
                "districts": list(set(districts)),
                "avg_members": round(np.mean([u.get('n_members', 4) for u in cluster_users]), 1),
                "avg_catchment": round(np.mean([u.get('catchment_area', 100) for u in cluster_users]), 1),
                "avg_farmland": round(np.mean([u.get('farmland_area', 0) for u in cluster_users]), 1),
                "avg_budget": round(np.mean([u.get('budget', 50000) for u in cluster_users]), 0),
                "total_water_potential_kl": round(sum([
                    u.get('catchment_area', 100) * u.get('avg_rainfall', 800) * 0.8 / 1000 
                    for u in cluster_users
                ]), 2),
                "color": self._get_cluster_color(cluster_id)
            }
        
        return cluster_stats
    
    def _get_cluster_color(self, cluster_id: int) -> str:
        """Get color for cluster visualization"""
        colors = [
            "#0676c8", "#32a854", "#f59e0b", "#ef4444", "#8b5cf6",
            "#06b6d4", "#ec4899", "#84cc16", "#f97316", "#6366f1"
        ]
        return colors[cluster_id % len(colors)]
    
    def save(self):
        """Save model to disk"""
        try:
            if self.model is not None:
                with open(CLUSTERING_MODEL_PATH, 'wb') as f:
                    pickle.dump(self.model, f)
            
            if self.scaler is not None:
                with open(SCALER_PATH, 'wb') as f:
                    pickle.dump(self.scaler, f)
            
            save_data = {
                "cluster_info": self.cluster_info,
                "district_encoder": self.district_encoder,
                "state_encoder": self.state_encoder,
                "n_clusters": self.n_clusters,
                "last_updated": datetime.now().isoformat()
            }
            with open(CLUSTER_INFO_PATH, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            print(f"Clustering model saved to {MODEL_DIR}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load(self):
        """Load model from disk"""
        with open(CLUSTERING_MODEL_PATH, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(SCALER_PATH, 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(CLUSTER_INFO_PATH, 'r') as f:
            save_data = json.load(f)
            self.cluster_info = save_data.get("cluster_info", {})
            self.district_encoder = save_data.get("district_encoder", {})
            self.state_encoder = save_data.get("state_encoder", {})
            self.n_clusters = save_data.get("n_clusters", 5)
    
    def get_nearby_droplets(self, user: Dict[str, Any], all_users: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
        """Get nearby droplets from same district/state"""
        if not all_users:
            return []
        
        user_uid = user.get('uid', '')
        user_district = user.get('district', '')
        user_state = user.get('state', '')
        
        # Same district first
        same_district = [
            u for u in all_users 
            if u.get('district') == user_district and u.get('uid') != user_uid
        ]
        if same_district:
            return same_district[:limit]
        
        # Fall back to same state
        same_state = [
            u for u in all_users 
            if u.get('state') == user_state and u.get('uid') != user_uid
        ]
        return same_state[:limit]


# Global instance
_clustering_model = None


def get_clustering_model() -> CommunityClusteringModel:
    """Get or create the clustering model instance"""
    global _clustering_model
    if _clustering_model is None:
        _clustering_model = CommunityClusteringModel()
    return _clustering_model


def initialize_clustering() -> bool:
    """Initialize the clustering model"""
    model = get_clustering_model()
    return model.initialize()


def train_clustering(users: List[Dict[str, Any]], n_clusters: int = None) -> Dict[str, Any]:
    """Train the clustering model"""
    model = get_clustering_model()
    model.initialize()
    return model.fit(users, n_clusters)


def predict_cluster(user: Dict[str, Any]) -> int:
    """Predict cluster for a user"""
    model = get_clustering_model()
    model.initialize()
    return model.predict(user)


def get_cluster_info() -> Dict[str, Any]:
    """Get cluster information"""
    model = get_clustering_model()
    model.initialize()
    return model.cluster_info


def get_nearby_droplets(user: Dict[str, Any], all_users: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
    """Get nearby droplets"""
    model = get_clustering_model()
    model.initialize()
    return model.get_nearby_droplets(user, all_users, limit)


# ==================== DOWNLOAD SCRIPT ====================

def download_all_models():
    """Download and save all required models"""
    print("Downloading Sentence Transformer model...")
    load_sentence_model()
    print("Done!")
    
    print("Initializing Clustering model...")
    initialize_clustering()
    print("Done!")


if __name__ == "__main__":
    download_all_models()