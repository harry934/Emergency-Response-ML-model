"""Image preprocessing for model inference."""
from __future__ import annotations

import io
import numpy as np
from PIL import Image

TARGET_SIZE: tuple[int, int] = (224, 224)


def preprocess_image(
    source: str | bytes | io.IOBase,
    target_size: tuple[int, int] = TARGET_SIZE,
) -> np.ndarray:
    """Load, resize, normalise and batch-expand an image.

    Parameters
    ----------
    source:
        A file path string, raw bytes, or any file-like object accepted by
        ``PIL.Image.open``.
    target_size:
        ``(width, height)`` to resize to before normalising.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(1, height, width, 3)`` with pixel values in
        ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``source`` is ``None`` or the image cannot be decoded.
    """
    if source is None:
        raise ValueError("source must not be None")

    try:
        if isinstance(source, Image.Image):
            img: Image.Image = source
        else:
            if isinstance(source, (bytes, bytearray)):
                source = io.BytesIO(source)
            img = Image.open(source)
    except Exception as exc:
        raise ValueError(f"Could not open image: {exc}") from exc

    img = img.convert("RGB")
    img = img.resize(target_size, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)
