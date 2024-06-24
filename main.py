from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel, Field
from surprise import Dataset, Reader
import pandas as pd

# Load the saved model
model = joblib.load('algo.joblib')

# Load the data
data = pd.read_csv("DATA1.csv", usecols=['UserId', 'ProductId', 'Rating'])

# Prepare the dataset for the surprise library
reader = Reader(rating_scale=(data['Rating'].min(), data['Rating'].max()))
dataset = Dataset.load_from_df(data[['UserId', 'ProductId', 'Rating']], reader)
trainset = dataset.build_full_trainset()

# Define FastAPI application
app = FastAPI()

class Item(BaseModel):
    UserId: int = Field(..., gt=0, description="User ID must be greater than 0")
    ProductId: int = Field(..., gt=0, description="Product ID must be greater than 0")
    Rating: int = Field(..., gt=0, le=5, description="Rating must be between 0 and 5")

class UserRequest(BaseModel):
    user_id: int = Field(..., gt=0, description="User ID must be greater than 0")
    rating: int = Field(..., gt=0, le=5, description="Rating must be between 0 and 5")

@app.get("/")
def root():
    return {"message": "Welcome to the Recommender System API!"}

@app.post("/recommend")
def get_recommendations(item: Item):
    user_id = item.UserId
    product_id = item.ProductId
    rating = item.Rating

    try:
        # Add the new rating to the dataset
        data.loc[len(data)] = [user_id, product_id, rating]

        # Rebuild the dataset for the surprise library
        dataset = Dataset.load_from_df(data[['UserId', 'ProductId', 'Rating']], reader)
        trainset = dataset.build_full_trainset()

        # Convert user_id to internal format
        inner_user_id = trainset.to_inner_uid(user_id)

        # Get list of all items
        all_items = trainset.all_items()
        all_item_ids = [trainset.to_raw_iid(item) for item in all_items]

        # Check unrated items
        user_ratings = trainset.ur[inner_user_id]
        rated_items = {item for (item, rating) in user_ratings}
        unrated_items = [item for item in all_item_ids if trainset.to_inner_iid(item) not in rated_items]

        # Predict ratings for unrated items
        predictions = []
        for item in unrated_items:
            pred = model.predict(user_id, item)
            predictions.append((item, pred.est))

        # Sort predictions and select top 5 items
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_n_recommendations = predictions[:2]

        # Build response
        recommended_products = [item for item, rating in top_n_recommendations]
        return {"recommendations": recommended_products}

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
