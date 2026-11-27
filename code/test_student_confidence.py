"""Regression checks for five-way confidence and paired label review."""

import unittest

import torch

from student_confidence import label_confidence, reference_review, summarize


class StudentConfidenceTest(unittest.TestCase):
    def test_five_way_confidence_conditions_on_letters_even_when_another_token_wins(self):
        logits = torch.tensor([20.0, 1.0, -2.0, 3.0, 1.0, -1.0])
        ids = [1, 2, 3, 4, 5]
        result = label_confidence(logits, ids)
        self.assertEqual(result["five_way_label"], "C")
        self.assertEqual(result["logit_C"], 3.0)
        self.assertAlmostEqual(result["confidence"], logits[ids].softmax(0)[2].item())
        self.assertAlmostEqual(sum(result[f"p_{label}"] for label in "ABCDE"), 1.0)
        self.assertAlmostEqual(result["label_mass"], logits.softmax(0)[ids].sum().item())
        self.assertLess(result["label_mass"], 0.001)

    def test_review_requires_both_archived_predictions_and_five_way_choices_to_match_teacher(self):
        first = [
            {"complaint_id": str(i), "reference_label": "B", "teacher_label": "A",
             "strict_label": strict, "five_way_label": "A", "confidence": confidence, "label_mass": 0.5}
            for i, strict, confidence in ((1, "A", 0.93), (2, "A", 1.0), (3, "", 0.98))
        ]
        second = [{**row, "strict_label": "A", "confidence": 0.99} for row in first]
        review = reference_review(first, second)
        self.assertEqual([row["complaint_id"] for row in review], ["2", "1"])
        self.assertEqual([row["min_confidence"] for row in review], [0.99, 0.93])

    def test_summary_keeps_invalid_and_reference_incorrect_distinct(self):
        cases = [
            {"reference_label": "A", "strict_label": "", "five_way_label": "A",
             "archived_first_token_choice": "A", "confidence": 1.0},
            {"reference_label": "A", "strict_label": "B", "five_way_label": "B",
             "archived_first_token_choice": "B", "confidence": 0.6},
        ]
        summary = summarize(cases)
        self.assertEqual(summary["five_way_cfpb_accuracy"], 0.5)
        self.assertEqual(summary["archived_strict_invalid"], 1)
        self.assertEqual(summary["archived_strict_cfpb_accuracy"], 0.0)
        self.assertEqual(summary["confidence_bins"][-1]["n"], 1)
        self.assertEqual(summary["confidence_bins"][2]["n"], 1)
