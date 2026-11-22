"""Frame encoding for the model (Lumine: one 1280x720 JPEG per 200 ms, sent as base64)."""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

DEFAULT_SIZE = (1280, 720)


def encode(frame, size=DEFAULT_SIZE, quality=90):
    """RGB uint8 (H, W, 3) -> (jpeg bytes, data URL). size=None keeps the native 854x480."""
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("expected an RGB uint8 frame")
    image = Image.fromarray(frame, "RGB")
    if size is not None and image.size != tuple(size):
        image = image.resize(tuple(size), Image.BICUBIC)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=int(quality), optimize=False)
    data = buf.getvalue()
    return data, "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")
