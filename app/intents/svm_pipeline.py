"""Offline SVM Intent Classifier Pipeline using scikit-learn.

Provides deterministic, offline machine-learning intent classification
using TF-IDF vectorization and a Linear Support Vector Machine (LinearSVC).
Operates completely locally without cloud APIs, network access, or heavy LLMs.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .models import IntentType
from .normalizer import normalize_text, strip_polite_fillers

logger = logging.getLogger(__name__)


@dataclass
class SVMPrediction:
    """Represents the output of an SVM intent inference step."""
    intent: IntentType
    confidence: float
    top_margin: float
    margin_gap: float
    is_confident: bool
    scores: Dict[str, float] = field(default_factory=dict)
    raw_text: str = ""
    normalized_text: str = ""


class SVMIntentPipeline:
    """Production offline SVM pipeline for in-vehicle voice assistant intent recognition."""

    DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "intent_svm.joblib"
    DEFAULT_META_PATH = Path(__file__).resolve().parent.parent / "models" / "intent_svm_meta.json"

    def __init__(
        self,
        model_path: Optional[Path] = None,
        confidence_threshold: float = 0.12,
        margin_gap_threshold: float = 0.10,
    ):
        """Initializes the SVM pipeline.

        Args:
            model_path: Path to the serialized .joblib model file.
            confidence_threshold: Softmax threshold for acceptable certainty.
            margin_gap_threshold: Minimum margin difference between top and runner-up class.
        """
        self.model_path = Path(model_path) if model_path else self.DEFAULT_MODEL_PATH
        self.meta_path = self.model_path.with_suffix(".json") if self.model_path.name.endswith(".joblib") else self.DEFAULT_META_PATH
        self.confidence_threshold = confidence_threshold
        self.margin_gap_threshold = margin_gap_threshold

        self.pipeline: Optional[Pipeline] = None
        self.classes: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self._is_loaded: bool = False

        if self.model_path.exists():
            self.load(self.model_path)

    @property
    def is_loaded(self) -> bool:
        """Returns True if the model is successfully loaded and ready for inference."""
        return self._is_loaded and self.pipeline is not None

    def load(self, path: Optional[Path] = None) -> bool:
        """Loads the serialized model artifact from disk without network access."""
        target_path = Path(path) if path else self.model_path
        if not target_path.exists():
            logger.warning(f"SVM model artifact not found at {target_path}. Falling back to deterministic rules.")
            self._is_loaded = False
            return False

        try:
            loaded_obj = joblib.load(target_path)
            if not isinstance(loaded_obj, Pipeline):
                logger.error(f"Invalid model artifact at {target_path}: expected scikit-learn Pipeline.")
                self._is_loaded = False
                return False

            self.pipeline = loaded_obj
            self.model_path = target_path

            # Extract classes
            clf = self.pipeline.named_steps.get("clf")
            if clf is not None and hasattr(clf, "classes_"):
                self.classes = list(clf.classes_)
            else:
                self.classes = []

            # Load metadata if present
            meta_file = target_path.with_name(target_path.stem + "_meta.json")
            if meta_file.exists():
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        self.metadata = json.load(f)
                except Exception as meta_err:
                    logger.warning(f"Could not read metadata file {meta_file}: {meta_err}")

            self._is_loaded = True
            logger.info(f"Loaded offline SVM intent model from {target_path} ({len(self.classes)} classes).")
            return True

        except Exception as e:
            logger.error(f"Failed to load SVM model artifact from {target_path}: {e}")
            self._is_loaded = False
            self.pipeline = None
            return False

    def save(self, target_path: Optional[Path] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Saves the trained pipeline and metadata record locally."""
        if self.pipeline is None:
            raise ValueError("Cannot save an uninitialized or untrained pipeline.")

        out_path = Path(target_path) if target_path else self.model_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.pipeline, out_path, compress=3)
        self.model_path = out_path

        meta_file = out_path.with_name(out_path.stem + "_meta.json")
        meta_to_save = dict(metadata or {})
        meta_to_save.setdefault("classes", self.classes)
        meta_to_save.setdefault("model_file", str(out_path.name))

        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta_to_save, f, indent=2)

        self.metadata = meta_to_save
        self._is_loaded = True
        logger.info(f"Saved SVM model to {out_path} and metadata to {meta_file}.")

    def preprocess(self, text: str) -> str:
        """Preprocesses text for vectorization while preserving domain keywords."""
        if not text:
            return ""
        norm = normalize_text(text)
        cleaned = strip_polite_fillers(norm)
        return cleaned

    def predict(self, text: str) -> SVMPrediction:
        """Classifies an utterance using the loaded SVM pipeline.

        Returns:
            SVMPrediction with predicted intent, confidence score, decision margin, and gap.
        """
        if not self.is_loaded or self.pipeline is None:
            raise RuntimeError("SVM pipeline is not loaded. Call load() or train the model first.")

        cleaned = self.preprocess(text)
        if not cleaned:
            return SVMPrediction(
                intent=IntentType.UNSUPPORTED,
                confidence=0.0,
                top_margin=0.0,
                margin_gap=0.0,
                is_confident=False,
                scores={},
                raw_text=text,
                normalized_text="",
            )

        # Compute decision function margins across all classes
        clf: LinearSVC = self.pipeline.named_steps["clf"]
        decision_scores = self.pipeline.decision_function([cleaned])

        if decision_scores.ndim == 1:
            # Binary classification fallback
            margins = np.array([-decision_scores[0], decision_scores[0]])
        else:
            margins = decision_scores[0]

        classes = list(clf.classes_)
        best_idx = int(np.argmax(margins))
        best_class_str = classes[best_idx]
        top_margin = float(margins[best_idx])

        # Runner-up margin and gap
        sorted_margins = np.sort(margins)
        runner_up_margin = float(sorted_margins[-2]) if len(sorted_margins) > 1 else top_margin
        margin_gap = float(top_margin - runner_up_margin)

        # Softmax pseudo-probability
        exp_margins = np.exp(margins - np.max(margins))
        probs = exp_margins / np.sum(exp_margins)
        confidence = float(probs[best_idx])

        scores_dict = {cls_name: float(m) for cls_name, m in zip(classes, margins)}

        # Out-of-distribution / Ambiguity check:
        # In a Linear SVM, a class prediction is only confident if it lies on the positive
        # side of the separating hyperplane (top_margin > 0.0), with sufficient margin gap.
        is_confident = (
            (confidence >= self.confidence_threshold)
            and (top_margin > 0.0)
            and (margin_gap >= self.margin_gap_threshold)
        )

        predicted_intent = getattr(IntentType, best_class_str, IntentType.UNSUPPORTED)
        if not is_confident and predicted_intent != IntentType.UNSUPPORTED:
            # Low confidence predictions on uncertain inputs are marked UNSUPPORTED for safety
            predicted_intent = IntentType.UNSUPPORTED

        return SVMPrediction(
            intent=predicted_intent,
            confidence=confidence,
            top_margin=top_margin,
            margin_gap=margin_gap,
            is_confident=is_confident,
            scores=scores_dict,
            raw_text=text,
            normalized_text=cleaned,
        )


def build_svm_pipeline(
    ngram_range: Tuple[int, int] = (1, 2),
    char_ngram_range: Tuple[int, int] = (3, 5),
    c_param: float = 1.5,
    random_state: int = 42,
) -> Pipeline:
    """Constructs a reproducible scikit-learn Pipeline with word and character n-gram features."""
    from sklearn.pipeline import FeatureUnion

    features = FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                ngram_range=ngram_range,
                sublinear_tf=True,
                min_df=1,
                token_pattern=r"(?u)\b\w+\b",
                strip_accents="unicode",
            ),
        ),
        (
            "char",
            TfidfVectorizer(
                ngram_range=char_ngram_range,
                analyzer="char_wb",
                sublinear_tf=True,
                min_df=1,
                strip_accents="unicode",
            ),
        ),
    ])

    classifier = LinearSVC(
        C=c_param,
        class_weight="balanced",
        random_state=random_state,
        max_iter=3000,
    )
    return Pipeline([("feats", features), ("clf", classifier)])

