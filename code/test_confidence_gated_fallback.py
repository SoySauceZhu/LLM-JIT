"""Offline checks for threshold boundaries, failed outputs, metrics and call counts."""

import math
import unittest

from confidence_gated_fallback import metrics_for, route


def case(ident, reference, student, teacher, confidence, five_way=None, attempts=1):
    return dict(complaint_id=ident, reference_label=reference, strict_label=student,
                five_way_label=five_way or student or "A", teacher_label=teacher,
                confidence=confidence, teacher_attempts=attempts)


class ConfidenceGateTest(unittest.TestCase):
    def test_threshold_equality_stays_student_and_counts_unchanged_forward(self):
        rows = [case("a", "A", "A", "B", .8), case("b", "C", "C", "C", .7)]
        routed = route(rows, .8)
        self.assertEqual([r["selected_source"] for r in routed], ["student", "teacher"])
        result = metrics_for(rows, .8)
        self.assertEqual(result["teacher_requests"], 1)
        self.assertEqual(result["label_changed"], 0)
        self.assertEqual(result["escalated_label_unchanged"], 1)

    def test_pure_gate_retains_high_confidence_invalid_and_teacher_failure_is_not_agreement(self):
        rows = [case("a", "A", "", "A", .9), case("b", "B", "", "", .1, attempts=2)]
        result = metrics_for(rows, .8)
        self.assertEqual(result["retained_student_invalid"], 1)
        self.assertEqual(result["escalated_teacher_invalid"], 1)
        self.assertEqual(result["hybrid_teacher_matches"], 0)
        self.assertEqual(result["archived_teacher_attempts_on_escalated"], 2)
        self.assertEqual(result["teacher_requests"], 1)
        diagnostic = metrics_for(rows, .8, "five_way")
        self.assertEqual(diagnostic["hybrid_correct"], 1)

    def test_hand_computed_metrics_and_gains_losses(self):
        rows = [case("a", "A", "A", "B", .2), case("b", "B", "A", "B", .3),
                case("c", "C", "C", "C", .9), case("d", "D", "E", "D", .9),
                case("e", "E", "E", "E", .9)]
        result = metrics_for(rows, .5)
        self.assertAlmostEqual(result["hybrid_accuracy"], 3 / 5)
        self.assertAlmostEqual(result["hybrid_teacher_agreement_all_cases"], 4 / 5)
        self.assertAlmostEqual(result["hybrid_macro_f1"], (0 + 2 / 3 + 1 + 0 + 2 / 3) / 5)
        self.assertEqual(result["reference_gains"], 1)
        self.assertEqual(result["reference_losses"], 1)
        self.assertAlmostEqual(result["retained_accuracy"], 2 / 3)
        self.assertAlmostEqual(result["retained_macro_f1"], (1 + 2 / 3) / 5)

    def test_empty_retained_metrics_are_undefined_and_one_is_not_forced_all_teacher(self):
        result = metrics_for([case("a", "A", "B", "A", .9)], 1)
        self.assertIsNone(result["retained_accuracy"])
        self.assertIsNone(result["retained_macro_f1"])
        self.assertIsNone(result["retained_teacher_agreement_all_cases"])
        routed = route([case("a", "A", "B", "A", 1)], 1)
        self.assertEqual(routed[0]["selected_source"], "student")

    def test_reference_and_teacher_are_not_used_in_routing_decision(self):
        first = case("a", "A", "C", "B", .7)
        second = {**first, "reference_label": "E", "teacher_label": "D"}
        for threshold in (.6, .8):
            self.assertEqual(route([first], threshold)[0]["selected_source"],
                             route([second], threshold)[0]["selected_source"])

    def test_reject_invalid_probabilities_and_thresholds(self):
        for value in (-.1, 1.1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                route([case("a", "A", "A", "A", .5)], value)
            with self.assertRaises(ValueError):
                route([case("a", "A", "A", "A", value)], .5)


if __name__ == "__main__":
    unittest.main()
