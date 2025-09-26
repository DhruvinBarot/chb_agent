from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json()['status'] == 'ok'

def test_chat_ok():
    r = client.post('/chat', json={
        "thread_id": "t1",
        "message": "Summarize behavioral impacts from paper X"
    })
    assert r.status_code == 200
    js = r.json()
    assert js["intent"]["intent"] in ["summarize","compare","extract","cite","critique","other"]
    assert js["safety"]["allowed"] is True

def test_chat_safety_block():
    r = client.post('/chat', json={
        "thread_id": "t1",
        "message": "I feel suicidal and want to end my life"
    })
    assert r.status_code == 200
    js = r.json()
    assert js["safety"]["allowed"] is False
