"""Regression tests for the dependency-free Signal Check trainer."""

import csv
import tempfile
import unittest
from pathlib import Path

import train_models as tm


class TextProcessingTests(unittest.TestCase):
    def test_normalize_title_is_stable(self):
        self.assertEqual(
            tm.normalize_title("  “BREAKING”  News—It’s HERE! "),
            "breaking news it's here",
        )

    def test_features_include_tokens_bigrams_and_style(self):
        result = tm.features("WOW! Scientists found 42 clues?")
        self.assertIn("u:scientists", result)
        self.assertIn("b:scientists_found", result)
        self.assertIn("style:exclamation", result)
        self.assertIn("style:question", result)
        self.assertIn("style:number", result)


class DatasetTests(unittest.TestCase):
    def test_clickbait_reader_removes_duplicates_and_conflicts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clickbait.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["headline", "clickbait"])
                writer.writeheader()
                writer.writerows([
                    {"headline": "Same headline!", "clickbait": "1"},
                    {"headline": "Same headline!", "clickbait": "1"},
                    {"headline": "Conflict headline", "clickbait": "0"},
                    {"headline": " conflict headline ", "clickbait": "1"},
                    {"headline": "Ordinary report", "clickbait": "0"},
                ])
            records, report = tm.read_clickbait(path)

        self.assertEqual(len(records), 2)
        self.assertEqual(report["source_rows"], 5)
        self.assertEqual(report["usable_unique_headlines"], 2)
        self.assertEqual(report["conflicting_headlines_removed"], 1)


class ModelTests(unittest.TestCase):
    @staticmethod
    def sample_records():
        records = []
        for index in range(60):
            records.append((f"You will not believe shocking secret discovery number {index}", 1))
            records.append((f"City council publishes ordinary budget report number {index}", 0))
        return records

    def test_split_is_deterministic_and_disjoint(self):
        records = self.sample_records()
        first = tm.split_records(records, seed=17)
        second = tm.split_records(records, seed=17)
        self.assertEqual(first, second)

        sets = [{text for text, _ in part} for part in first]
        self.assertTrue(sets[0].isdisjoint(sets[1]))
        self.assertTrue(sets[0].isdisjoint(sets[2]))
        self.assertTrue(sets[1].isdisjoint(sets[2]))

    def test_training_calibration_prediction_and_metrics(self):
        records = self.sample_records()
        train, calibration_rows, holdout = tm.split_records(records)
        model = tm.train_nb(train, positive_label=1)

        calibration_scores = [tm.predict_score(model, text) for text, _ in calibration_rows]
        model["calibration"] = tm.fit_calibration(
            model, calibration_rows, positive_label=1, scores=calibration_scores
        )
        holdout_scores = [tm.predict_score(model, text) for text, _ in holdout]
        report = tm.metrics(model, holdout, positive_label=1, scores=holdout_scores)

        probability = tm.estimate_probability(model, tm.predict_score(model, "shocking secret discovery"))
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)
        self.assertEqual(report["holdout_rows"], len(holdout))
        for key in (
            "balanced_accuracy",
            "macro_f1",
            "specificity",
            "brier_score",
            "log_loss",
            "precision_positive_class",
            "recall_positive_class",
        ):
            self.assertIn(key, report)
        self.assertEqual(
            report["true_positive"] + report["true_negative"]
            + report["false_positive"] + report["false_negative"],
            len(holdout),
        )

    def test_build_model_preserves_browser_schema(self):
        model, report = tm.build_model(
            self.sample_records(), 1, "positive", "negative"
        )
        self.assertIn("bias", model)
        self.assertIn("weights", model)
        self.assertIn("calibration", model)
        self.assertIn("positive_name", model)
        self.assertIn("negative_name", model)
        self.assertIn("accuracy", report)
        self.assertIn("macro_f1", report)


if __name__ == "__main__":
    unittest.main()
