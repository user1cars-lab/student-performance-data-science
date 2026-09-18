from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint():
    response = client.post("/predict", json={})
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["predicted_grade"] <= 20
    assert body["performance_level"] in {"Low", "Medium", "High"}


def test_predict_endpoint_rejects_invalid_range():
    response = client.post("/predict", json={"age": 40})
    assert response.status_code == 422
