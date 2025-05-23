from fastapi.testclient import TestClient
from .main import app, app_state

client = TestClient(app)

def example_history():
    # 12개 step, [[혈당, 인슐린]] (mg/dL, U/h), 가장 간단한 정상 예시
    return [
        [110.5, 0.15], [112.1, 0.18], [113.4, 0.10], [115.2, 0.20],
        [118.3, 0.14], [120.0, 0.22], [121.8, 0.30], [123.4, 0.24],
        [122.7, 0.18], [124.9, 0.20], [125.0, 0.25], [126.1, 0.27]
    ]

def test_env_step_success():
    payload = {"history": example_history(), "hour": 10}
    response = client.post("/env_step", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    # 행동값이 범위 내인지 체크 (0.0 ~ 5.0 U/h)
    assert "insulin_action_U_per_h" in data
    assert 0.0 <= data["insulin_action_U_per_h"] <= 5.0

def test_env_step_failure_shape():
    # history의 shape가 이상할 때(11개)
    wrong_history = example_history()[:-1]
    payload = {"history": wrong_history, "hour": 0}
    response = client.post("/env_step", json=payload)
    assert response.status_code == 500
    assert "history 형상 오류" in response.json()["detail"]

def test_env_step_nan():
    nan_history = example_history()
    nan_history[2][0] = float("nan")
    payload = {"history": nan_history, "hour": 0}
    response = client.post("/env_step", json=payload)
    assert response.status_code == 500
    assert "입력 NaN/Inf" in response.json()["detail"]


def test_env_step_counter_ignored_hour():
    app_state["api_call_counter"] = 0
    payload1 = {"history": example_history(), "hour": 5}
    resp1 = client.post("/env_step", json=payload1)
    assert resp1.status_code == 200
    assert app_state["api_call_counter"] == 1

    payload2 = {"history": example_history(), "hour": 99}
    resp2 = client.post("/env_step", json=payload2)
    assert resp2.status_code == 200
    # Counter should increment regardless of provided hour
    assert app_state["api_call_counter"] == 2
