from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator
import uvicorn
import os
import sys
import redis
import json
import hashlib
import joblib
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.logger import get_logger

logger = get_logger("API")

# --- SETTING ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'titanic_pipeline.pkl')
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")

# --- GLOBAL VARIABLES (RAM) ---
ml_models = {}


# --- LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. STARTUP
    try:
        logger.info("Loading model into memory... 🧠")
        ml_models["titanic"] = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully! ✅")
    except Exception as e:
        logger.error(f"Critical Error: Model could not be loaded: {e}")
        ml_models["titanic"] = None

    # Redis Connection
    try:
        pool = redis.ConnectionPool(host=REDIS_HOST, port=6379, db=0, decode_responses=True)
        app.state.redis = redis.Redis(connection_pool=pool)
        app.state.redis.ping()
        logger.info("Redis connection successful! 🚀")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        app.state.redis = None

    yield

    # 2. SHUTDOWN
    ml_models.clear()
    logger.info("Clean up complete. Shutting down...")


# Launch the application with lifespan
app = FastAPI(
    title="Titanic Survival Prediction API",
    version="1.0.0",
    lifespan=lifespan
)

Instrumentator().instrument(app).expose(app)


class PassengerData(BaseModel):
    PassengerId: int
    Name: str
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Ticket: str
    Fare: float
    Cabin: str | None = None
    Embarked: str


@app.post("/predict")
def predict_survival(passenger: PassengerData, request: Request):
    try:
        # 1. Create Cache Key
        data_dict = passenger.dict()
        data_str = json.dumps(data_dict, sort_keys=True)
        cache_key = hashlib.sha256(data_str.encode()).hexdigest()

        # 2. Redis Control (Accessed via State)
        r = request.app.state.redis
        if r:
            cached = r.get(cache_key)
            if cached:
                logger.info(f"Cache HIT! ⚡: {passenger.Name}")
                return json.loads(cached)

        # 3. Model Prediction (from RAM)
        model = ml_models.get("titanic")
        if not model:
            raise HTTPException(status_code=500, detail="Model not loaded")

        logger.info(f"Cache MISS. Computing... 🧮: {passenger.Name}")

        # Turn into DataFrame
        df = pd.DataFrame([data_dict])
        prediction = model.predict(df)[0]

        # int64 JSON cannot be serialized, convert it to int.
        result = int(prediction)

        response_payload = {
            "passenger_name": passenger.Name,
            "prediction": result,
            "source": "model"
        }

        # 4. Write to Redis
        if r:
            cache_to_save = response_payload.copy()
            cache_to_save["source"] = "cache"
            r.setex(cache_key, 3600, json.dumps(cache_to_save))

        return response_payload

    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)