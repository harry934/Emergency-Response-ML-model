"""Model loading and inference."""
from __future__ import annotations

import numpy as np

LABELS: list[str] = ["Accident", "HeavyTraffic", "NormalRoadActivity"]


def load_model(model_path: str):
    """Load a Keras model from *model_path*.

    Returns the loaded model, or raises the underlying TensorFlow/Keras
    exception on failure — callers should catch and handle gracefully.
    """
    import tensorflow as tf  # deferred so the module is importable without TF

    return tf.keras.models.load_model(model_path)


def predict_image(
    img_array: np.ndarray,
    model,
    threshold: float = 0.0,
) -> tuple[str, np.ndarray]:
    """Run inference on a pre-processed image array.

    Parameters
    ----------
    img_array:
        Float32 array of shape ``(1, 224, 224, 3)`` as produced by
        :func:`interface.core.preprocessor.preprocess_image`.
    model:
        Loaded Keras model.  Pass ``None`` to activate fallback mode (random
        normalised probabilities — useful for UI demos without a model file).
    threshold:
        Minimum confidence required for a definitive prediction.  If the
        highest probability is below *threshold*, the label ``"Uncertain"``
        is returned.  ``0.0`` disables the check.

    Returns
    -------
    tuple[str, np.ndarray]
        ``(label, probs)`` where *label* is one of the ``LABELS`` list (or
        ``"Uncertain"``) and *probs* is a length-3 float array that sums to
        approximately ``1.0``.
    """
    if model is not None:
        probs: np.ndarray = np.array(model.predict(img_array, verbose=0)[0], dtype=np.float32)
    else:
        raw = np.random.rand(3).astype(np.float32)
        probs = raw / raw.sum()

    if threshold > 0.0 and float(np.max(probs)) < threshold:
        return "Uncertain", probs

    label = LABELS[int(np.argmax(probs))]
    return label, probs
