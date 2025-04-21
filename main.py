from fastapi import FastAPI, Request
from pydantic import BaseModel, Field, confloat
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
import joblib
import pandas as pd
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY

# Initialize API with description
app = FastAPI(
    title="Recruitment SVM Classifier API",
    description="This API predicts whether a candidate is suitable for hiring based on experience and technical test score.",
    version="1.0.0"
)

# Load model and scaler
model = joblib.load("model_files/model.pkl")
scaler = joblib.load("model_files/scaler.pkl")

# Input schema with validation, description, and example
class Applicant(BaseModel):
    experience_years: confloat(ge=0.0, le=10.0) = Field(
        ...,
        description="Total years of software development experience (must be between 0.0 and 10.0)",
        example=3.5
    )
    technical_score: confloat(ge=0.0, le=100.0) = Field(
        ...,
        description="Score obtained from the technical test (must be between 0.0 and 100.0)",
        example=85.0
    )

# Prediction endpoint
@app.post("/predict")
def predict(applicant: Applicant):
    exp = applicant.experience_years
    score = applicant.technical_score

    if exp < 2.0 and score < 60.0:
        return {
            "prediction": "Rejected (Rule-based)",
            "input": {"experience_years": exp, "technical_score": score}
        }

    input_df = pd.DataFrame([[exp, score]], columns=["experience_years", "technical_score"])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    return {
        "prediction": "Accepted" if prediction == 0 else "Rejected",
        "input": {"experience_years": exp, "technical_score": score}
    }

# Custom validation error handler for clearer user feedback
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "message": "Validation Error: Please enter valid input values.",
            "requirements": {
                "experience_years": "Must be a number between 0.0 and 10.0",
                "technical_score": "Must be a number between 0.0 and 100.0"
            },
            "details": jsonable_encoder(exc.errors())
        }
    )
