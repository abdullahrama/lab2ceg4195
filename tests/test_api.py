"""
tests/test_api.py — CEG4195 Lab 2
-----------------------------------
pytest tests for all Flask API endpoints.

Design: a session-scoped autouse fixture creates a stub (untrained) UNet
checkpoint and sets MODEL_PATH in os.environ BEFORE lab2 is imported.
This lets CI run all tests without a real trained model or a GPU.

Run locally:
    pytest tests/ -v

Run in CI:
    The ci.yml workflow sets MODEL_PATH directly in the env block of the
    test step, using a stub .pth created by an inline Python script.
"""

import io
import os
import pytest
import numpy as np
import torch
import segmentation_models_pytorch as smp
from PIL import Image


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def stub_model(tmp_path_factory):
    """
    Creates a freshly initialised (untrained) UNet checkpoint and points
    MODEL_PATH at it.  Must run before lab2 is imported anywhere, so it is
    session-scoped and autouse=True.
    """
    model_dir  = tmp_path_factory.mktemp("models")
    model_path = str(model_dir / "unet_resnet34.pth")

    m = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    torch.save(m.state_dict(), model_path)

    os.environ["MODEL_PATH"]    = model_path
    os.environ["SEG_THRESHOLD"] = "0.5"
    os.environ["PORT"]          = "5000"

    return model_path


@pytest.fixture(scope="session")
def client(stub_model):
    """
    Imports lab2 only after MODEL_PATH is set, then returns a Flask test client.
    """
    import lab2  # noqa: PLC0415 — intentional late import
    lab2.app.config["TESTING"] = True
    with lab2.app.test_client() as c:
        yield c


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_png_bytes(width: int = 64, height: int = 64) -> io.BytesIO:
    """Returns a BytesIO containing a random RGB PNG image."""
    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_home_returns_200(client):
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "message" in data
    assert "endpoints" in data


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert "model" in data
    assert "threshold" in data
    assert "device" in data


def test_segment_valid_image(client):
    img_buf = _make_png_bytes()
    resp = client.post(
        "/segment",
        data={"file": (img_buf, "test.png", "image/png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert "mask_png_base64" in data
    assert "building_fraction" in data
    assert 0.0 <= data["building_fraction"] <= 1.0
    assert "threshold_used" in data
    assert "input_size" in data
    assert len(data["input_size"]) == 2


def test_segment_missing_file_returns_400(client):
    resp = client.post(
        "/segment",
        data={},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_segment_empty_filename_returns_400(client):
    resp = client.post(
        "/segment",
        data={"file": (io.BytesIO(b""), "", "image/png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    assert "error" in resp.get_json()
