from fastapi.testclient import TestClient
from Sim_CLI.main import app, app_state

client = TestClient(app)


def test_get_state_returns_last_cgm():
    app_state['last_cgm'] = 123.0
    resp = client.get('/get_state')
    assert resp.status_code == 200
    assert resp.json()['cgm'] == 123.0


def test_env_step_updates_state_and_returns_reward():
    app_state['last_cgm'] = 150.0
    resp = client.post('/env_step', json={'action': 1.0})
    assert resp.status_code == 200
    data = resp.json()
    assert 'cgm' in data and 'reward' in data and 'done' in data
    assert data['cgm'] == app_state['last_cgm']
