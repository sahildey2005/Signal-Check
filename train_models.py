"""Train compact headline models from the supplied CSV datasets.

Uses only Python's standard library. The resulting JSON contains token weights,
not copies of the source rows. Inputs can be overridden from the command line.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DEFAULT_CLICKBAIT = DATA_DIR / "clickbait_data.csv"
DEFAULT_FAKE = DATA_DIR / "Fake.csv"
DEFAULT_TRUE = DATA_DIR / "True.csv"
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
MAX_FEATURES = 30000
MIN_DOCUMENT_FREQUENCY = 2
ALPHA = 1.0
SEED = 42


def clean_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").replace("’", "'")
    return " ".join(value.split())


def normalize_title(value: str) -> str:
    return " ".join(TOKEN_RE.findall(clean_text(value).lower()))


def features(value: str) -> list[str]:
    raw = clean_text(value)
    words = TOKEN_RE.findall(raw.lower())
    unigram_features = [f"u:{word}" for word in words if len(word) > 1]
    bigram_features = [f"b:{left}_{right}" for left, right in zip(words, words[1:])]
    style_features: list[str] = []
    exclamations = raw.count("!")
    questions = raw.count("?")
    if exclamations:
        style_features.append("style:exclamation")
    if exclamations >= 2:
        style_features.append("style:multiple_exclamations")
    if questions:
        style_features.append("style:question")
    if questions >= 2:
        style_features.append("style:multiple_questions")
    uppercase_words = re.findall(r"\b[A-Z]{3,}\b", raw)
    if len(uppercase_words) >= 2 and len(uppercase_words) / max(len(words), 1) >= 0.2:
        style_features.append("style:all_caps")
    if re.search(r"\d", raw):
        style_features.append("style:number")
    if re.search(r"[\"“”]", raw):
        style_features.append("style:quote")
    if "..." in raw or "…" in raw:
        style_features.append("style:ellipsis")
    if len(words) <= 6:
        style_features.append("style:short")
    elif len(words) >= 14:
        style_features.append("style:long")
    return unigram_features + bigram_features + style_features


def read_clickbait(path: Path) -> tuple[list[tuple[str, int]], dict[str, int]]:
    title_labels: dict[str, set[int]] = defaultdict(set)
    original_titles: dict[str, str] = {}
    rows = 0
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"headline", "clickbait"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path.name} must contain headline and clickbait columns")
        for row in reader:
            rows += 1
            title = clean_text(row.get("headline", ""))
            normalized = normalize_title(title)
            try:
                label = int((row.get("clickbait") or "").strip())
            except ValueError:
                continue
            if normalized and label in (0, 1):
                title_labels[normalized].add(label)
                original_titles.setdefault(normalized, title)
    conflicts = {title for title, labels in title_labels.items() if len(labels) > 1}
    records = [(original_titles[title], next(iter(labels))) for title, labels in title_labels.items() if title not in conflicts]
    return records, {
        "source_rows": rows,
        "usable_unique_headlines": len(records),
        "conflicting_headlines_removed": len(conflicts),
    }


def read_fake_real(fake_path: Path, true_path: Path) -> tuple[list[tuple[str, int]], dict[str, int]]:
    by_title: dict[str, set[int]] = defaultdict(set)
    original_titles: dict[str, str] = {}
    source_rows = {"fake": 0, "true": 0}
    for label, path, name in ((1, fake_path, "fake"), (0, true_path, "true")):
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            if "title" not in (reader.fieldnames or []):
                raise ValueError(f"{path.name} must contain a title column")
            for row in reader:
                source_rows[name] += 1
                title = clean_text(row.get("title", ""))
                normalized = normalize_title(title)
                if normalized:
                    by_title[normalized].add(label)
                    original_titles.setdefault(normalized, title)

    conflicts = {title for title, labels in by_title.items() if len(labels) > 1}
    records: list[tuple[str, int]] = []
    for normalized, labels in by_title.items():
        if normalized in conflicts:
            continue
        records.append((original_titles[normalized], next(iter(labels))))
    return records, {
        "source_rows": source_rows,
        "usable_unique_headlines": len(records),
        "conflicting_headlines_removed": len(conflicts),
    }


def split_records(records: list[tuple[str, int]], seed: int = SEED) -> tuple[list[tuple[str, int]], list[tuple[str, int]], list[tuple[str, int]]]:
    groups: dict[int, list[str]] = defaultdict(list)
    for text, label in records:
        groups[label].append(text)
    rng = random.Random(seed)
    train: list[tuple[str, int]] = []
    calibration: list[tuple[str, int]] = []
    validation: list[tuple[str, int]] = []
    for label, texts in sorted(groups.items()):
        rng.shuffle(texts)
        train_at = max(1, int(len(texts) * 0.7))
        calibration_at = max(train_at + 1, int(len(texts) * 0.85))
        calibration_at = min(calibration_at, len(texts) - 1)
        train.extend((text, label) for text in texts[:train_at])
        calibration.extend((text, label) for text in texts[train_at:calibration_at])
        validation.extend((text, label) for text in texts[calibration_at:])
    rng.shuffle(train)
    rng.shuffle(calibration)
    rng.shuffle(validation)
    return train, calibration, validation


def train_nb(records: list[tuple[str, int]], positive_label: int) -> dict[str, object]:
    document_frequency: Counter[str] = Counter()
    for text, _ in records:
        document_frequency.update(set(features(text)))

    eligible = [token for token, count in document_frequency.items() if count >= MIN_DOCUMENT_FREQUENCY]
    vocabulary = sorted(eligible, key=lambda token: (-document_frequency[token], token))[:MAX_FEATURES]
    vocabulary_set = set(vocabulary)
    token_counts: dict[int, Counter[str]] = {0: Counter(), 1: Counter()}
    document_counts = Counter()

    for text, label in records:
        document_counts[label] += 1
        token_counts[label].update(token for token in features(text) if token in vocabulary_set)

    vocabulary_size = max(1, len(vocabulary))
    totals = {label: sum(token_counts[label].values()) for label in (0, 1)}
    positive_docs = document_counts[positive_label]
    negative_docs = document_counts[1 - positive_label]
    bias = math.log((positive_docs + ALPHA) / (negative_docs + ALPHA))
    weights = {}
    denominator_positive = totals[positive_label] + ALPHA * vocabulary_size
    denominator_negative = totals[1 - positive_label] + ALPHA * vocabulary_size
    for token in vocabulary:
        positive_probability = (token_counts[positive_label][token] + ALPHA) / denominator_positive
        negative_probability = (token_counts[1 - positive_label][token] + ALPHA) / denominator_negative
        weights[token] = round(math.log(positive_probability / negative_probability), 6)

    return {
        "bias": round(bias, 6),
        "positive_label": positive_label,
        "vocabulary_size": len(vocabulary),
        "weights": weights,
    }


def predict_score(model: dict[str, object], text: str) -> float:
    weights = model["weights"]
    assert isinstance(weights, dict)
    score = float(model["bias"])
    for token, count in Counter(features(text)).items():
        score += float(weights.get(token, 0.0)) * count
    return score


def sigmoid(value: float) -> float:
    value = max(-30.0, min(30.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def fit_calibration(model: dict[str, object], records: list[tuple[str, int]], positive_label: int) -> dict[str, float]:
    scores = [predict_score(model, text) for text, _ in records]
    scale = sorted(abs(score) for score in scores)[len(scores) // 2] if scores else 1.0
    scale = max(scale, 1.0)
    normalized = [(max(-8.0, min(8.0, score / scale)), int(label == positive_label)) for score, (_, label) in zip(scores, records)]
    slope = 1.0
    intercept = 0.0
    learning_rate = 0.08
    for _ in range(3000):
        grad_slope = 0.0
        grad_intercept = 0.0
        for value, target in normalized:
            error = sigmoid(slope * value + intercept) - target
            grad_slope += error * value
            grad_intercept += error
        count = max(len(normalized), 1)
        grad_slope = grad_slope / count + 0.01 * (slope - 1.0)
        grad_intercept /= count
        slope -= learning_rate * grad_slope
        intercept -= learning_rate * grad_intercept
    return {
        "scale": round(scale, 6),
        "slope": round(slope, 6),
        "intercept": round(intercept, 6),
    }


def estimate_probability(model: dict[str, object], score: float) -> float:
    calibration = model.get("calibration", {})
    assert isinstance(calibration, dict)
    scale = max(float(calibration.get("scale", 1.0)), 1.0)
    slope = float(calibration.get("slope", 1.0))
    intercept = float(calibration.get("intercept", 0.0))
    normalized = max(-8.0, min(8.0, score / scale))
    return sigmoid(slope * normalized + intercept)


def metrics(model: dict[str, object], records: list[tuple[str, int]], positive_label: int) -> dict[str, float | int]:
    tp = tn = fp = fn = 0
    for text, label in records:
        predicted_positive = estimate_probability(model, predict_score(model, text)) >= 0.5
        actual_positive = label == positive_label
        if predicted_positive and actual_positive:
            tp += 1
        elif predicted_positive and not actual_positive:
            fp += 1
        elif not predicted_positive and actual_positive:
            fn += 1
        else:
            tn += 1
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "holdout_rows": total,
        "accuracy": round((tp + tn) / total, 4) if total else 0.0,
        "precision_positive_class": round(precision, 4),
        "recall_positive_class": round(recall, 4),
        "f1_positive_class": round(f1, 4),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def build_model(records: list[tuple[str, int]], positive_label: int, positive_name: str, negative_name: str) -> tuple[dict[str, object], dict[str, object]]:
    train, calibration_records, validation = split_records(records)
    model = train_nb(train, positive_label)
    model["positive_name"] = positive_name
    model["negative_name"] = negative_name
    model["calibration"] = fit_calibration(model, calibration_records, positive_label)
    report = {
        "training_rows": len(train),
        "calibration_rows": len(calibration_records),
        **metrics(model, validation, positive_label),
    }
    return model, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Signal Check headline models.")
    parser.add_argument("--clickbait", type=Path, default=DEFAULT_CLICKBAIT)
    parser.add_argument("--fake", type=Path, default=DEFAULT_FAKE)
    parser.add_argument("--true", type=Path, default=DEFAULT_TRUE)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("models.json"))
    args = parser.parse_args()

    for path in (args.clickbait, args.fake, args.true):
        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {path}")

    clickbait_records, clickbait_source = read_clickbait(args.clickbait)
    fake_real_records, fake_real_source = read_fake_real(args.fake, args.true)
    clickbait_model, clickbait_report = build_model(clickbait_records, 1, "clickbait", "not clickbait")
    fake_model, fake_report = build_model(fake_real_records, 1, "fake-labeled", "true-labeled")

    output = {
        "format_version": 1,
        "model_type": "multinomial-naive-bayes-headline-demo",
        "clickbait": clickbait_model,
        "fake_real": fake_model,
        "report": {
            "split": "70% train / 15% calibration / 15% holdout; deduplicated headlines; seed 42",
            "clickbait_source": clickbait_source,
            "clickbait_holdout": clickbait_report,
            "fake_real_source": fake_real_source,
            "fake_real_holdout": fake_report,
            "notes": [
                "The fake/real labels come from the supplied filenames and are not independent fact-checks.",
                "Models learn patterns in these datasets and may reflect their age, topics, publishers, and labeling choices.",
                "Predictions apply to headline text only; they do not verify the article or claim.",
                "Naive Bayes percentages are model estimates, not calibrated real-world probabilities.",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "clickbait": clickbait_report, "fake_real": fake_report}, indent=2))


if __name__ == "__main__":
    main()
