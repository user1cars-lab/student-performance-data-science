from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import FEATURES
from src.predict import predict

app = FastAPI(title="Student Performance Prediction API", version="1.0.0")


class StudentInput(BaseModel):
    school: str = "GP"
    sex: str = "F"
    age: int = Field(17, ge=15, le=22)
    address: str = "U"
    famsize: str = "GT3"
    pstatus: str = "A"
    medu: int = Field(2, ge=0, le=4)
    fedu: int = Field(2, ge=0, le=4)
    mjob: str = "other"
    fjob: str = "other"
    reason: str = "course"
    guardian: str = "mother"
    traveltime: int = Field(1, ge=1, le=4)
    studytime: int = Field(2, ge=1, le=4)
    failures: int = Field(0, ge=0, le=4)
    schoolsup: str = "no"
    famsup: str = "yes"
    paid: str = "no"
    activities: str = "no"
    nursery: str = "yes"
    higher: str = "yes"
    internet: str = "yes"
    romantic: str = "no"
    famrel: int = Field(4, ge=1, le=5)
    freetime: int = Field(3, ge=1, le=5)
    goout: int = Field(3, ge=1, le=5)
    dalc: int = Field(1, ge=1, le=5)
    walc: int = Field(1, ge=1, le=5)
    health: int = Field(5, ge=1, le=5)
    absences: int = Field(2, ge=0, le=93)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict_grade(student: StudentInput) -> dict[str, float | str]:
    try:
        values = student.model_dump()
        grade, level = predict(values)
        return {"predicted_grade": round(grade, 3), "performance_level": level}
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
