# INDRA Backend API Documentation

> **Initiative for Drainage and Rainwater Acquisition**  
> Production-ready FastAPI backend with AI-powered water conservation features

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- pip or conda

### Installation

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\Activate.ps1  # Windows PowerShell

pip install -r requirements.txt
```

### Environment Setup

1. Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

2. Fill in your API keys in `.env`:
```env
OPENROUTER_API_KEY=your_key_here
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_key
```

### Run Server

```bash
uvicorn main:app --reload --port 8000
```

Server runs at: `http://localhost:8000`  
API Docs at: `http://localhost:8000/docs`

---

## 📁 Project Structure

```
backend/
├── main.py              # FastAPI app entry point
├── config.py            # Centralized configuration (env vars)
├── requirements.txt     # Python dependencies
├── render.yaml          # Render deployment config
├── .env.example         # Environment template
│
├── # Core Modules
├── assessment.py        # RWH assessment & cost prediction
├── vendor.py            # Vendor search (web scraping)
├── gis_utils.py         # GIS data manager (rainfall, groundwater)
│
├── # AI Chatbots (RAG + LLM)
├── chatbot_standard.py  # Urban/Standard chatbot
├── chatbot_rural.py     # Rural/Gramin chatbot
│
├── # Gramin Features
├── water_management.py  # Water distribution AI
├── crop_suggestion.py   # Smart cropping AI
├── community.py         # Social dashboard & clustering
│
├── # Auth & Users
├── user_auth.py         # Firebase auth & user profiles
│
├── # Data & Models
├── data/
│   └── INDRA_Processed_Data.csv  # GIS rainfall data (165K records)
└── models/
    └── all-MiniLM-L6-v2/         # Local embedding model
```

---

## 🔌 API Endpoints Reference

### Base URL
- **Local**: `http://localhost:8000`
- **Production**: `https://your-render-url.onrender.com`

---

## 1. Health & Status

### `GET /`
Root endpoint - API info

**Response:**
```json
{
  "message": "INDRA API - Rainwater Harvesting Platform",
  "version": "1.0.0",
  "status": "operational"
}
```

### `GET /health`
Health check for monitoring

**Response:**
```json
{"status": "healthy", "service": "INDRA Backend"}
```

---

## 2. Assessment (Urban RWH)

### `POST /api/assessment/analyze`
Analyze RWH potential and get cost estimates

**Request Body:**
```json
{
  "name": "Rahul Sharma",
  "email": "rahul@example.com",
  "phone": "9876543210",
  "state": "Maharashtra",
  "district": "Mumbai",
  "pincode": "400001",
  "n_members": 4,
  "catchment_area": 100,
  "roof_type": "Flat",
  "roof_material": "RCC",
  "budget": 25000
}
```

**Response:**
```json
{
  "user_info": {
    "name": "Rahul Sharma",
    "location": "Mumbai, Maharashtra - 400001"
  },
  "gis_data": {
    "rainfall": {
      "monsoon": 2200,
      "total_annual": 2400
    },
    "groundwater": {
      "extraction_percentage": 45.2
    }
  },
  "rwh_potential": {
    "harvestable_liters": 216000,
    "monthly_potential": 18000,
    "days_supply": 150
  },
  "cost_analysis": {
    "estimated_cost": 22500,
    "budget_status": "Within Budget",
    "roi_years": 2.5
  },
  "recommendations": [
    "Install rooftop collection system",
    "Add first-flush diverter"
  ]
}
```

**Data Flow:**
1. Frontend sends user profile data (from Firebase or form)
2. Backend looks up GIS data using pincode
3. Cost predictor calculates based on: budget, roof type, roof material, household size, rainfall zone
4. Returns comprehensive analysis

---

## 3. Vendor Search

### `GET /api/vendors/search`
Search for RWH vendors and service providers

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| location | string | Yes | City/area name |
| search_type | string | No | `all`, `stores`, `mechanics`, `components`, `online`, `services` |
| lat | float | No | User latitude |
| lon | float | No | User longitude |

**Example:**
```
GET /api/vendors/search?location=Mumbai&search_type=all
```

**Response:**
```json
{
  "stores": [
    {
      "name": "Rainwater Solutions Mumbai",
      "category": "store",
      "location": "Andheri West, Mumbai",
      "contact": "+91-22-2634-5678",
      "email": "info@rainwatersolutions.in",
      "website": "https://rainwatersolutions.in",
      "rating": 4.5,
      "price_range": "₹5,000 - ₹50,000",
      "description": "Complete RWH systems for residential and commercial"
    }
  ],
  "mechanics": [...],
  "components": [...],
  "online_stores": [...],
  "service_providers": [...]
}
```

### `GET /api/vendors/diy-guide`
Get DIY installation guide

**Response:**
```json
{
  "title": "Complete DIY RWH Installation Guide",
  "difficulty": "Moderate",
  "estimated_time": "2-3 days",
  "estimated_cost": "₹5,000 - ₹15,000",
  "steps": [
    "Survey your roof area and calculate potential",
    "Install gutters along roof edges",
    "Connect downpipes to collection point",
    "Install first-flush diverter",
    "Set up storage tank with filter"
  ],
  "materials_needed": [
    "PVC pipes (4 inch)",
    "Gutters",
    "Storage tank (1000-5000L)",
    "Filter mesh",
    "First-flush diverter"
  ]
}
```

---

## 4. Chatbots (AI-Powered)

### `POST /api/chatbot/standard`
Urban/Standard mode chatbot - focuses on RWH systems, costs, installation

**Request:**
```json
{
  "message": "How much does a basic RWH system cost?"
}
```

**Response:**
```json
{
  "response": "A basic RWH system for a 100 sqm roof costs ₹15,000-25,000. This includes gutters, pipes, filter, and a 1000L tank. Installation adds ₹3,000-5,000. ROI is typically 2-3 years through water bill savings."
}
```

### `POST /api/chatbot/rural`
Rural/Gramin mode chatbot - focuses on farming, irrigation, water management (Hindi-friendly)

**Request:**
```json
{
  "message": "Mere khet mein paani ki kami hai, kya karun?"
}
```

**Response:**
```json
{
  "response": "Aap check dam ya farm pond bana sakte hain. Drip irrigation se 40% paani bachega. Kam paani wali fasal jaise bajra, jowar ugayein. Mulching se bhi paani bachta hai."
}
```

**AI Architecture:**
- **RAG**: Retrieves relevant context from Qdrant vector database
- **LLM**: OpenRouter API (nvidia/nemotron model)
- **Embeddings**: Local all-MiniLM-L6-v2 model

---

## 5. Water Management (Gramin)

### `POST /api/gramin/water-management/predict`
AI-powered water distribution for rural communities

**Request:**
```json
{
  "location": "Nashik, Maharashtra",
  "pincode": "422001",
  "season": "summer",
  "crop_type": "Onion",
  "cattle_count": 5,
  "household_members": 6,
  "farm_size_acres": 2.5
}
```

**Response:**
```json
{
  "distribution": {
    "irrigation_buckets": 45,
    "cattle_buckets": 8,
    "drinking_buckets": 4,
    "irrigation_pct": 78.9,
    "cattle_pct": 14.0,
    "drinking_pct": 7.0
  },
  "recommendations": [
    "Use drip irrigation for onions - saves 40% water",
    "Water cattle in morning and evening only"
  ],
  "ai_insights": "Based on Nashik's 700mm annual rainfall and your farm size, prioritize irrigation during morning hours. Consider rainwater harvesting to supplement summer water needs.",
  "water_status": "Semi-Critical",
  "gis_summary": "Nashik District: 700mm annual rainfall, 78% groundwater extraction"
}
```

### `GET /api/gramin/water-management/tips`
Get water conservation tips

**Query Parameters:**
| Parameter | Type | Required |
|-----------|------|----------|
| season | string | Yes |
| crop_type | string | No |

**Example:**
```
GET /api/gramin/water-management/tips?season=summer&crop_type=wheat
```

---

## 6. Smart Cropping (Gramin)

### `POST /api/gramin/crop-suggestions`
AI-powered crop recommendations based on location and conditions

**Request:**
```json
{
  "location": "Pune",
  "pincode": "411001",
  "soil_type": "Black Cotton",
  "season": "Kharif",
  "water_availability": "Medium",
  "farm_size_acres": 3.0
}
```

**Response:**
```json
{
  "recommendations": [
    {
      "crop_name": "Soybean",
      "water_requirement_liters": 350000,
      "estimated_market_price_per_kg": 45,
      "yield_per_acre_kg": 1200,
      "total_profit_estimate": 54000,
      "price_per_liter_ratio": 0.154,
      "environmental_impact_score": 8,
      "soil_health_impact": "Positive",
      "farmer_ease_score": 8,
      "rank": 1,
      "justification": "Nitrogen fixing improves soil health"
    },
    {
      "crop_name": "Cotton",
      "water_requirement_liters": 500000,
      "estimated_market_price_per_kg": 65,
      "yield_per_acre_kg": 400,
      "total_profit_estimate": 26000,
      "price_per_liter_ratio": 0.052,
      "environmental_impact_score": 5,
      "soil_health_impact": "Neutral",
      "farmer_ease_score": 6,
      "rank": 2,
      "justification": "Good for black soil but high water use"
    }
  ],
  "season_context": "Kharif season (Jun-Sep) benefits from monsoon rains",
  "water_context": "Pune receives 900mm annual rainfall",
  "general_advice": "Consider intercropping to maximize land use"
}
```

---

## 7. User Authentication

### `POST /api/users/profile`
Create or update user profile

**Headers:**
```
Authorization: Bearer <firebase_id_token>
```

**Request:**
```json
{
  "uid": "firebase_user_id",
  "email": "user@example.com",
  "name": "Rahul Sharma",
  "state": "Maharashtra",
  "district": "Mumbai",
  "pincode": "400001",
  "n_members": 4,
  "catchment_area": 100,
  "farmland_area": 0,
  "roof_type": "Flat",
  "roof_material": "RCC",
  "budget": 25000,
  "mode": "urban"
}
```

### `GET /api/users/profile/{uid}`
Get user profile

### `GET /api/users/leaderboard`
Get community leaderboard (top droplets)

---

## 8. Community (Social Dashboard)

### `GET /api/community/feed`
Get community feed posts

### `POST /api/community/posts`
Create a new post

### `GET /api/community/clusters`
Get community clusters (grouped by location/water-table)

---

## 9. News & Stats

### `GET /api/news/water`
Get water conservation news articles

### `GET /api/stats/water`
Get India water statistics

### `GET /api/blogs`
Get curated blog posts

---

## 🔧 Configuration Reference

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENROUTER_API_KEY` | Yes | - | OpenRouter API key for LLM |
| `QDRANT_URL` | Yes | - | Qdrant cloud cluster URL |
| `QDRANT_API_KEY` | Yes | - | Qdrant API key |
| `LLM_MODEL` | No | `nvidia/nemotron-nano-12b-v2-vl:free` | LLM model name |
| `LLM_TEMPERATURE` | No | `0.3` | LLM temperature |
| `RAG_COLLECTION_NAME` | No | `standrd_rag` | Qdrant collection name |
| `GIS_DATA_PATH` | No | `./data/INDRA_Processed_Data.csv` | GIS data file |
| `CORS_ORIGINS` | No | `http://localhost:5173` | Allowed origins (comma-separated) |
| `DEBUG_MODE` | No | `false` | Enable debug logging |

---

## 🚀 Deployment (Render)

### Option 1: Blueprint Deploy
1. Push code to GitHub
2. Go to Render Dashboard → New → Blueprint
3. Connect your repo
4. Render reads `render.yaml` automatically
5. Set environment variables in Render dashboard

### Option 2: Manual Deploy
1. Create new Web Service on Render
2. Connect GitHub repo
3. Settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Add environment variables
5. Deploy

### Environment Variables on Render
Set these in the Render dashboard (Environment tab):
- `OPENROUTER_API_KEY`
- `QDRANT_URL`
- `QDRANT_API_KEY`
- `CORS_ORIGINS` (your frontend URLs)

---

## 📊 Data Flow Diagrams

### Assessment Flow
```
Frontend                    Backend                     External
   │                           │                           │
   │  POST /assessment/analyze │                           │
   │ ─────────────────────────>│                           │
   │  {pincode, roof_type...}  │                           │
   │                           │                           │
   │                           │  Lookup GIS Data          │
   │                           │ ─────────────────────────>│
   │                           │  rainfall, groundwater    │ GIS CSV
   │                           │ <─────────────────────────│
   │                           │                           │
   │                           │  Calculate RWH Potential  │
   │                           │  (runoff coefficient,     │
   │                           │   household multiplier)   │
   │                           │                           │
   │                           │  Predict Cost             │
   │                           │  (budget tier, material,  │
   │                           │   rainfall zone)          │
   │                           │                           │
   │  Response                 │                           │
   │ <─────────────────────────│                           │
   │  {gis_data, potential,    │                           │
   │   cost, recommendations}  │                           │
```

### Chatbot Flow (RAG + LLM)
```
Frontend                    Backend                     External
   │                           │                           │
   │  POST /chatbot/standard   │                           │
   │ ─────────────────────────>│                           │
   │  {message: "..."}         │                           │
   │                           │                           │
   │                           │  1. Embed query           │
   │                           │  (local MiniLM model)     │
   │                           │                           │
   │                           │  2. Vector search         │
   │                           │ ─────────────────────────>│ Qdrant
   │                           │  relevant docs (k=2)      │
   │                           │ <─────────────────────────│
   │                           │                           │
   │                           │  3. Build prompt          │
   │                           │  (context + question)     │
   │                           │                           │
   │                           │  4. LLM inference         │
   │                           │ ─────────────────────────>│ OpenRouter
   │                           │  response                 │
   │                           │ <─────────────────────────│
   │                           │                           │
   │  Response                 │                           │
   │ <─────────────────────────│                           │
   │  {response: "..."}        │                           │
```

### Crop Suggestion Flow
```
Frontend                    Backend                     External
   │                           │                           │
   │  POST /crop-suggestions   │                           │
   │ ─────────────────────────>│                           │
   │  {location, soil, season} │                           │
   │                           │                           │
   │                           │  1. GIS Lookup            │
   │                           │  (rainfall, groundwater)  │
   │                           │                           │
   │                           │  2. RAG retrieval         │
   │                           │ ─────────────────────────>│ Qdrant
   │                           │  crop knowledge           │
   │                           │ <─────────────────────────│
   │                           │                           │
   │                           │  3. LLM generation        │
   │                           │ ─────────────────────────>│ OpenRouter
   │                           │  5 ranked crops (JSON)    │
   │                           │ <─────────────────────────│
   │                           │                           │
   │                           │  4. Parse & validate      │
   │                           │                           │
   │  Response                 │                           │
   │ <─────────────────────────│                           │
   │  {recommendations: [...]} │                           │
```

---

## 🔒 Security Notes

1. **Never commit `.env` file** - Use `.env.example` as template
2. **Firebase Admin SDK** - Keep `firebase-service-account.json` secure
3. **CORS** - Set specific origins in production, not `*`
4. **API Keys** - Use Render's environment variables, not hardcoded

---

## 🐛 Troubleshooting

### "GIS data not found"
- Check `GIS_DATA_PATH` points to correct CSV file
- Ensure `data/INDRA_Processed_Data.csv` exists

### "Qdrant collection not found"
- Verify `QDRANT_URL` and `QDRANT_API_KEY`
- Check collection name matches `RAG_COLLECTION_NAME`

### "LLM request failed"
- Verify `OPENROUTER_API_KEY` is valid
- Check rate limits on free tier

### "Firebase not initialized"
- Ensure `firebase-service-account.json` exists
- Or set `FIREBASE_SERVICE_ACCOUNT_PATH` env var

---

## 📞 Support

For issues or questions:
- Create GitHub issue
- Email: team@indra.app

---

*Built with ❤️ for water conservation in India*
