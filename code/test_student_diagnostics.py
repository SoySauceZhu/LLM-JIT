"""Offline evidence that diagnostics preserve decoding and survive resume/export."""

import json
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from mortgage_prompt import load_prompt
from run_store import attempts_by_case, initialize
from student_baseline import collect_student, export_results, load_student, predict_once, student_config
from student_diagnostics import inference_runtime
from test_baselines import fixture_cases, generated


def inference_fixture(empty=True):
    labels = list(range(10, 15))
    vocabulary = {1: "<eos>", 2: "<bos>", 16: "<end_of_turn>",
                  **dict(zip(labels, "ABCDE"))}
    tokenizer = Mock(bos_token_id=2, eos_token_id=1, pad_token_id=0)
    tokenizer.apply_chat_template.return_value = {
        "input_ids": torch.tensor([[2, 3]]), "attention_mask": torch.tensor([[1, 1]])}
    tokenizer.convert_ids_to_tokens.side_effect = lambda i: (
        [vocabulary.get(t, str(t)) for t in i] if isinstance(i, list) else vocabulary.get(i, str(i)))
    tokenizer.decode.side_effect = lambda ids, **_: "".join(vocabulary.get(i, str(i)) for i in ids)
    tokenizer.get_chat_template.return_value = "native chat template"
    scores = torch.full((1, 20), -20.)
    scores[0, 10], scores[0, 16] = (4., 5.) if empty else (5., 4.)
    output = SimpleNamespace(sequences=torch.tensor([[2, 3, *([16] if empty else [10, 16])]]),
                             scores=(scores,))
    model = SimpleNamespace(device=torch.device("cpu"), training=False,
                            config=SimpleNamespace(max_position_embeddings=32, _attn_implementation="sdpa"),
                            parameters=lambda: iter([torch.nn.Parameter(torch.zeros(1))]),
                            generate=Mock(return_value=output))
    from transformers import GenerationConfig
    settings = GenerationConfig(max_new_tokens=16, do_sample=False, eos_token_id=[1, 16])
    return model, tokenizer, settings, labels


class TestStudentDiagnostics(unittest.TestCase):
    def test_actual_stop_token_is_preserved_without_changing_generation(self):
        for empty in (True, False):
            with self.subTest(empty=empty):
                model, tokenizer, settings, labels = inference_fixture(empty)
                baseline = predict_once(model, tokenizer, settings, labels, [])
                runtime = inference_runtime(model, tokenizer, settings)
                observed = predict_once(model, tokenizer, settings, labels, [],
                                        diagnostics=True, runtime=runtime)
                diagnostic = observed.pop("diagnostics")
                self.assertEqual(baseline, observed)
                self.assertEqual(model.generate.call_count, 2)  # No extra diagnostic forward pass.
                self.assertEqual(diagnostic["generated_token_ids"], [16] if empty else [10, 16])
                self.assertEqual(diagnostic["removed_stop_token_id"], 16)
                self.assertEqual(diagnostic["stop_reason"], "eos_token")
                self.assertEqual(diagnostic["first_step"]["selected"]["token_id"], 16 if empty else 10)
                self.assertEqual(diagnostic["first_step"]["selected"]["rank"], 1)
                self.assertEqual(diagnostic["first_step"]["best_label_minus_best_stop_score"], -1 if empty else 1)
                self.assertEqual(diagnostic["prompt"]["bos_token_count"], 1)
                self.assertEqual(runtime["attention_implementation"], "sdpa")
                self.assertEqual(runtime["generation_config"]["eos_token_id"], [1, 16])
                json.dumps(diagnostic, allow_nan=False)

    def test_nonfinite_scores_are_reported_as_valid_json(self):
        model, tokenizer, settings, labels = inference_fixture()
        model.generate.return_value.scores[0][0, 7:10] = torch.tensor([float("nan"), float("inf"), -float("inf")])
        result = predict_once(model, tokenizer, settings, labels, [], diagnostics=True)
        step = result["diagnostics"]["first_step"]
        self.assertEqual(step["nonfinite_counts"], {"nan": 1, "positive_inf": 1, "negative_inf": 1})
        self.assertIsNone(step["selected"]["probability"])
        self.assertIsNone(step["selected"]["rank"])
        json.dumps(result, allow_nan=False)

    def test_each_retry_is_committed_and_reexported_without_regeneration(self):
        model, tokenizer, settings, labels = inference_fixture()
        empty = predict_once(model, tokenizer, settings, labels, [], diagnostics=True)
        cases = fixture_cases()[:2]
        with TemporaryDirectory() as tmp:
            output = Path(tmp)
            db = sqlite3.connect(output / "run.sqlite")
            self.addCleanup(db.close)
            config = student_config(cases, "validation", load_prompt("v3"), diagnostics=True)
            initialize(db, config)
            with self.assertRaises(KeyboardInterrupt):
                collect_student(db, cases, Mock(side_effect=[empty, KeyboardInterrupt]))
            self.assertEqual(json.loads(attempts_by_case(db)["1"][0]["diagnostics"]), empty["diagnostics"])
            export_results(db, output, output / "missing_teacher")
            self.assertEqual(len(json.loads((output / "diagnostics.json").read_text())["attempts"]), 1)
            accepted = {**generated("A"), "diagnostics": {"version": 1, "generated_token_ids": [10, 16]}}
            collect_student(db, cases, Mock(side_effect=[empty, accepted]))
            metrics = export_results(db, output, output / "missing_teacher")
            records = json.loads((output / "diagnostics.json").read_text())["attempts"]
            self.assertEqual([(r["complaint_id"], r["attempt_number"]) for r in records],
                             [("1", 1), ("1", 2), ("2", 1)])
            self.assertEqual(metrics["diagnostics"]["attempts_recorded"], 3)
            self.assertEqual(metrics["diagnostics"]["generation_attempts"], 3)
            no_more_inference = Mock(side_effect=AssertionError("completed run"))
            collect_student(db, cases, no_more_inference)
            no_more_inference.assert_not_called()
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                initialize(db, student_config(cases, "validation", load_prompt("v3")))

    def test_legacy_schema_three_runs_resume_without_adding_diagnostics(self):
        with sqlite3.connect(":memory:") as db:
            cases = fixture_cases()[:1]
            config = student_config(cases, "validation", load_prompt("v3"))
            self.assertNotIn("diagnostics", config)
            self.assertNotIn("attn_implementation", config)
            initialize(db, config)
            db.execute("ALTER TABLE attempts DROP COLUMN diagnostics")
            db.commit()
            initialize(db, config)
            collect_student(db, cases, lambda _: generated("A"))
            self.assertNotIn("diagnostics", attempts_by_case(db)["1"][0])

    def test_explicit_attention_is_applied_to_model_and_frozen_in_config(self):
        model, tokenizer, _, labels = inference_fixture()
        model.to = Mock(return_value=SimpleNamespace(eval=lambda: model))
        model.generation_config = SimpleNamespace(eos_token_id=[1, 16])
        tokenizer.encode.side_effect = lambda label, **_: [labels["ABCDE".index(label)]]
        tokenizer.get_vocab.return_value = {"<end_of_turn>": 16}
        tokenizer.convert_tokens_to_ids.return_value = 16
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
                patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model) as load:
            load_student("gemma3-270m-it", attn_implementation="eager")
        self.assertEqual(load.call_args.kwargs["attn_implementation"], "eager")
        config = student_config(fixture_cases(), "validation", load_prompt("v3"), attn_implementation="eager")
        self.assertEqual(config["attn_implementation"], "eager")


if __name__ == "__main__":
    unittest.main()
