from fastapi.testclient import TestClient
from diart_server.app import create_app


def test_live_page_served():
    client = TestClient(create_app())
    r = client.get("/demo/live")
    assert r.status_code == 200 and "diarize/stream" in r.text
