"""Offline acceptance tests for three partitions, full collection and resume."""

import csv
import json
import sqlite3
import subprocess
import sys
import unittest
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from openai import APITimeoutError

from mortgage_data import (
    EXPERIMENT_SPLIT_COUNTS, ROOT, audit_rows, load_cohort,
    load_experiment, manifest_cases, prepare_cases, select_rows, sha256,
)
from mortgage_prompt import build_messages, load_prompt
from run_store import attempts_by_case, initialize, read_config
from student_baseline import (
    collect_student, export_results as export_student, load_teacher_labels, predict_once, student_config,
    student_output,
    generation_stop_ids,
    load_student, STUDENT_MODELS,
)
from teacher_baseline import export_results as export_teacher, initialize as initialize_teacher
from teacher_collection import collect
from test_teacher_collection import reply


def fixture_cases():
    return [{"complaint_id": str(i), "text": text, "reference_label": "A",
             "split": "validation",
             "sequence_no": i, "received_at_utc": f"2024-02-0{i}T00:00:00.000Z",
             "arrival_index": i, "input_sha256": sha256(text),
             "student_tokens_before_clip": 1, "clipped": False}
            for i, text in enumerate("ABCDE", 1)]


def generated(raw, choice="C"):
    return {"raw_reply": raw, "first_token_choice": choice,
            "prompt_tokens": 12, "completion_tokens": 2}


class TestDataAndCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_cohort()

    def test_frozen_cohort_has_exactly_three_splits_and_preserves_pilot_inputs(self):
        self.assertEqual(audit_rows(self.rows, reference_snapshot=True)["split_counts"],
                         dict(history=9159, validation=1013, test=2024))
        archived = read_config(ROOT / "output/gemma_dev_n300_s42_v1")["cases"]
        by_id = {row["complaint_id"]: row for row in self.rows}
        self.assertTrue(all(by_id[case["complaint_id"]]["split"] == "history" for case in archived))
        self.assertEqual(len(select_rows(self.rows, "history")), 9159)
        self.assertEqual(len(select_rows(self.rows, "validation")), 1013)

    def test_formal_data_covers_all_rows_with_disjoint_chronological_partitions(self):
        formal = load_experiment()
        self.assertEqual(len(formal), 12196)
        self.assertEqual(formal, self.rows)
        self.assertEqual({r["split"] for r in formal}, set(EXPERIMENT_SPLIT_COUNTS))
        partitions = []
        for split, count in EXPERIMENT_SPLIT_COUNTS.items():
            part = select_rows(formal, split)
            self.assertEqual(len(part), count)
            self.assertEqual(part, select_rows(self.rows, split))
            partitions.append(part)
        for key in ("complaint_id", "text_sha256"):
            identifiers = [set(row[key] for row in part) for part in partitions]
            self.assertEqual(sum(map(len, identifiers)), len(set.union(*identifiers)))
            self.assertEqual(set.union(*identifiers), {row[key] for row in formal})
        self.assertLess(partitions[0][-1]["received_at_utc"], partitions[1][0]["received_at_utc"])
        self.assertLess(partitions[1][-1]["received_at_utc"], partitions[2][0]["received_at_utc"])

    def test_full_split_selection_keeps_arrival_order_without_sampling(self):
        expected = select_rows(self.rows, "history")
        self.assertEqual(select_rows(list(reversed(self.rows)), "history"), expected)
        self.assertEqual(select_rows(list(reversed(self.rows))), self.rows)

    def test_unknown_or_unavailable_split_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown split"):
            select_rows(self.rows, "typo")
        with self.assertRaisesRegex(ValueError, "Unknown split"):
            select_rows(load_experiment(), "development")
        with self.assertRaisesRegex(ValueError, "No rows available"):
            select_rows(select_rows(self.rows, "history"), "validation")

    def test_audit_rejects_duplicates_changed_text_and_reordered_stream(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            audit_rows([self.rows[0], self.rows[0]])
        duplicate_text = {**self.rows[0], "complaint_id": "99999999", "sequence_no": "12000",
                          "received_at_utc": "2024-02-01T00:00:00.000Z", "split": "history"}
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            audit_rows([self.rows[0], duplicate_text])
        with self.assertRaisesRegex(ValueError, "changed narrative"):
            audit_rows([{**self.rows[0], "text": "changed"}])
        with self.assertRaisesRegex(ValueError, "sequence order"):
            audit_rows(self.rows[:2][::-1])
        with self.assertRaisesRegex(ValueError, "temporal split"):
            audit_rows([{**self.rows[0], "split": "validation"}])

    def test_same_head_tail_preparation_and_prompt_for_both_models(self):
        tokenizer = Mock()
        tokenizer.encode.return_value = SimpleNamespace(ids=list(range(3001)))
        tokenizer.decode.return_value = "identical clipped text"
        case = prepare_cases([self.rows[0]], tokenizer)[0]
        tokenizer.decode.assert_called_once_with(list(range(2000)) + list(range(2001, 3001)),
                                                skip_special_tokens=False)
        self.assertTrue(case["clipped"])
        self.assertEqual(case["student_tokens_before_clip"], 3001)
        self.assertEqual(case["input_sha256"], sha256("identical clipped text"))
        self.assertEqual(case["split"], "history")
        self.assertEqual(manifest_cases([case])[0]["input_sha256"], case["input_sha256"])
        for version in ("v1", "v2", "v3", "v4"):
            messages = build_messages(case["text"], load_prompt(version))
            self.assertIn(case["text"], messages[-1]["content"])
        tokenizer.encode.return_value = SimpleNamespace(ids=list(range(3000)))
        self.assertFalse(prepare_cases([self.rows[0]], tokenizer)[0]["clipped"])

    def test_previews_do_not_prepare_tokenizer_or_infer(self):
        import student_baseline
        import teacher_baseline
        for module, split in ((teacher_baseline, None), (teacher_baseline, "history"),
                              (teacher_baseline, "validation"), (teacher_baseline, "test"),
                              (student_baseline, "history"), (student_baseline, "validation"),
                              (student_baseline, "test")):
            flags = [] if split is None else ["--split", split]
            with self.subTest(module=module.__name__, split=split), TemporaryDirectory() as tmp, \
                    patch.object(module, "prepare_cases") as prepare, \
                    patch.object(module, "load_experiment", return_value=self.rows) as formal, \
                    patch("sys.argv", [module.__name__, *flags, "--output", str(Path(tmp) / "run")]):
                module.main()
                prepare.assert_not_called()
                formal.assert_called_once_with()
                self.assertFalse((Path(tmp) / "run").exists())

    def test_cli_rejects_removed_split_sampling_and_invalid_concurrency(self):
        for script, flags in (("teacher_baseline.py", ["--concurrency", "0"]),
                              ("teacher_baseline.py", ["--size", "10"]),
                              ("student_baseline.py", ["--seed", "42"]),
                              ("teacher_baseline.py", ["--split", "development"]),
                              ("student_baseline.py", ["--split", "development"])):
            result = subprocess.run([sys.executable, str(ROOT / "code" / script), *flags],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stderr)


class TestStudentCUDA(unittest.TestCase):
    def test_cuda_preview_needs_no_gpu_or_model_download(self):
        import student_baseline

        rows = [{**case, "split": "validation"} for case in fixture_cases()]
        with TemporaryDirectory() as tmp, \
                patch.object(student_baseline, "load_experiment", return_value=rows), \
                patch.object(student_baseline, "prepare_cases") as prepare, \
                patch.object(student_baseline, "load_student") as load, \
                patch("sys.argv", ["student_baseline", "--device", "cuda", "--output", tmp]):
            student_baseline.main()
            prepare.assert_not_called()
            load.assert_not_called()
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_unavailable_cuda_fails_before_loading_weights_or_tokenizer(self):
        for name in STUDENT_MODELS:
            with self.subTest(model=name), patch("torch.cuda.is_available", return_value=False), \
                    patch("transformers.AutoTokenizer.from_pretrained") as tokenizer, \
                    patch("transformers.AutoModelForCausalLM.from_pretrained") as weights:
                with self.assertRaisesRegex(ValueError, "CUDA GPU is unavailable.*--device cpu"):
                    load_student(name, "cuda")
                tokenizer.assert_not_called()
                weights.assert_not_called()

    def test_cuda_loads_each_student_in_fp32_and_freezes_device(self):
        import torch

        for name, profile in STUDENT_MODELS.items():
            with self.subTest(model=name):
                tokenizer = Mock(eos_token_id=1, pad_token_id=0)
                tokenizer.encode.side_effect = lambda letter, **_: [ord(letter)]
                tokenizer.get_vocab.return_value = {"<end_of_turn>": 106}
                tokenizer.convert_tokens_to_ids.return_value = 106
                model = Mock()
                model.to.return_value = model
                model.eval.return_value = model
                model.generation_config = SimpleNamespace(eos_token_id=1)
                with patch("torch.cuda.is_available", return_value=True), \
                        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
                        patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model) as weights:
                    loaded, _, _, _ = load_student(name, "cuda")
                self.assertIs(loaded, model)
                self.assertEqual(weights.call_args.args, (profile["model"],))
                self.assertIs(weights.call_args.kwargs["dtype"], torch.float32)
                self.assertEqual(weights.call_args.kwargs["revision"], profile["revision"])
                model.to.assert_called_once_with("cuda")
                model.eval.assert_called_once_with()
                config = student_config(fixture_cases(), "validation", load_prompt("v3"), name, "cuda")
                self.assertEqual(config["device"], "cuda")
                self.assertEqual(config["dtype"], "torch.float32")
                with closing(sqlite3.connect(":memory:")) as db:
                    initialize(db, config)
                    initialize(db, config)  # Same-device resume is allowed.
                    with self.assertRaisesRegex(ValueError, "configuration changed"):
                        initialize(db, {**config, "device": "cpu"})

    def test_predict_moves_all_inputs_to_the_cuda_model_device(self):
        import torch

        input_ids = Mock()
        input_ids.to.return_value = torch.tensor([[4, 5]])
        attention_mask = Mock()
        attention_mask.to.return_value = torch.tensor([[1, 1]])
        tokenizer = Mock(eos_token_id=9)
        tokenizer.apply_chat_template.return_value = {"input_ids": input_ids, "attention_mask": attention_mask}
        tokenizer.decode.return_value = "C"
        model = SimpleNamespace(
            device=torch.device("cuda:0"), config=SimpleNamespace(max_position_embeddings=32),
            generate=Mock(return_value=SimpleNamespace(
                sequences=torch.tensor([[4, 5, 2, 9]]), scores=(torch.tensor([[0., 0., 10., 0., 0.]]),))))
        result = predict_once(model, tokenizer, SimpleNamespace(max_new_tokens=16), list(range(5)), [])
        input_ids.to.assert_called_once_with(model.device)
        attention_mask.to.assert_called_once_with(model.device)
        self.assertIs(model.generate.call_args.kwargs["input_ids"], input_ids.to.return_value)
        self.assertIs(model.generate.call_args.kwargs["attention_mask"], attention_mask.to.return_value)
        self.assertEqual(result["raw_reply"], "C")
        self.assertEqual(result["first_token_choice"], "C")


class TestFullTeacherCollection(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.output = Path(tmp.name)
        self.path = self.output / "run.sqlite"
        self.db = sqlite3.connect(self.path)
        self.addCleanup(self.db.close)
        self.cases = fixture_cases()
        for case, split in zip(self.cases, ("history", "history", "validation", "test", "test")):
            case["split"] = split
        self.config = initialize_teacher(self.db, self.cases)

    def test_resume_keeps_raw_attempts_without_training_tables_or_exports(self):
        def before_restart(client, text):
            if text in "AB":
                return reply(text)
            raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            collect(self.db, self.path, None, self.cases, 1, predict=before_restart)
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0], 2)
        with closing(sqlite3.connect(self.path)) as resumed:
            initialize_teacher(resumed, self.cases)

            def after_restart(client, text):
                if text == "C":
                    raise APITimeoutError(request=httpx.Request("POST", "https://example.invalid"))
                return reply(text)

            collect(resumed, self.path, None, self.cases, 1, predict=after_restart)
            self.assertEqual(resumed.execute("SELECT COUNT(*) FROM attempts").fetchone()[0], 6)
            self.assertEqual(dict(resumed.execute("SELECT status, COUNT(*) FROM attempts GROUP BY status")),
                             {"ok": 4, "timeout": 2})
            predictor = Mock(side_effect=AssertionError("must not repeat finished cases"))
            collect(resumed, self.path, None, self.cases, 1, predict=predictor)
            predictor.assert_not_called()
            self.assertEqual({row[0] for row in resumed.execute("SELECT name FROM sqlite_master WHERE type='table'")},
                             {"run", "attempts"})
            summary = export_teacher(resumed, self.cases, self.output)
        self.assertEqual(summary["n_terminal"], 5)
        self.assertEqual(summary["usage"]["attempts_missing_usage"], 6)
        self.assertIsNone(summary["usage"]["cost_usd"])
        self.assertNotIn("milestones", summary)
        self.assertNotIn("history_successes", summary)
        self.assertFalse((self.output / "history_labels.csv").exists())
        self.assertFalse((self.output / "prefixes").exists())
        with (self.output / "predictions.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row["split"] for row in rows], [case["split"] for case in self.cases])
        self.assertEqual([row["complaint_id"] for row in rows], [case["complaint_id"] for case in self.cases])
        self.assertEqual(rows[2]["status"], "timeout")
        self.assertTrue(summary["complete"])

    def test_partition_filter_can_collect_concurrently_without_prefix_logic(self):
        collect(self.db, self.path, None, self.cases, 2, predict=lambda *_: reply("A"))
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0], 5)
        with self.assertRaisesRegex(ValueError, "positive"):
            collect(self.db, self.path, None, self.cases, 0, predict=Mock())

    def test_config_changes_refused_and_empty_exports_are_provisional(self):
        summary = export_teacher(self.db, self.cases, self.output)
        self.assertEqual(summary["n_pending"], 5)
        self.assertEqual(summary["n"], 0)
        self.assertIsNone(summary["accuracy"])
        self.assertIsNone(self.config["split"])
        self.assertEqual(self.config["schema_version"], 3)
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            initialize_teacher(self.db, self.cases, load_prompt("v4"))
        self.assertEqual(read_config(self.output), self.config)


class TestStudentAndPairing(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.output = Path(tmp.name) / "student"
        self.teacher = Path(tmp.name) / "teacher"
        self.output.mkdir()
        self.teacher.mkdir()
        self.cases = fixture_cases()
        self.config = student_config(self.cases, "validation", load_prompt("v3"))
        self.db = sqlite3.connect(self.output / "run.sqlite")
        self.addCleanup(self.db.close)
        initialize(self.db, self.config)

    def teacher_collect(self, cases=None, prompt=None):
        with closing(sqlite3.connect(self.teacher / "run.sqlite")) as db:
            initialize_teacher(db, cases or self.cases, prompt or load_prompt("v3"), split="validation")
            collect(db, self.teacher / "run.sqlite", None, cases or self.cases, 1,
                    predict=lambda _, text: reply("A"))

    def test_restart_between_retries_preserves_attempt_budget_and_raw_replies(self):
        count = 0

        def interrupted(case):
            nonlocal count
            count += 1
            if count == 2:
                raise KeyboardInterrupt()
            return generated("Label: C", "D")

        with self.assertRaises(KeyboardInterrupt):
            collect_student(self.db, self.cases, interrupted)
        metrics = export_student(self.db, self.output, self.teacher)
        self.assertEqual(metrics["n_terminal"], 0)
        self.assertEqual(metrics["usage"]["attempts"], 1)
        self.assertIsNone(metrics["strict_generation"]["accuracy"])
        predictor = Mock(return_value=generated(" A "))
        collect_student(self.db, self.cases, predictor)
        self.assertEqual(predictor.call_count, 5)  # One remaining retry, four untouched cases.
        self.assertEqual(len(attempts_by_case(self.db)["1"]), 2)
        metrics = export_student(self.db, self.output, self.teacher)
        self.assertTrue(metrics["complete"])
        self.assertEqual(metrics["strict_generation"]["accuracy"], 1.0)
        self.assertIsNone(metrics["strict_generation"]["teacher_agreement_all_cases"])
        self.assertEqual(metrics["usage"]["known_completion_tokens"], 12)
        self.assertEqual(self.db.execute("SELECT raw_reply FROM attempts ORDER BY id LIMIT 1").fetchone()[0], "Label: C")
        predictor.reset_mock()
        collect_student(self.db, self.cases, predictor)
        predictor.assert_not_called()

    def test_invalid_terminal_outputs_are_errors_and_diagnostics_stay_separate(self):
        collect_student(self.db, self.cases, lambda _: generated("C because...", "A"))
        metrics = export_student(self.db, self.output, self.teacher)
        self.assertEqual(metrics["usage"]["attempts"], 10)
        self.assertEqual(metrics["strict_generation"]["accuracy"], 0)
        self.assertEqual(metrics["strict_generation"]["valid_rate"], 0)
        self.assertEqual(metrics["first_token_choice"]["accuracy"], 1)
        self.assertEqual(set(metrics["strict_generation"]["per_class"]), set("ABCDE"))
        self.assertEqual(metrics["strict_generation"]["confusion_rows_reference_columns_prediction"]["A"]["FAIL"], 5)

    def test_teacher_can_finish_later_and_export_without_repeating_student(self):
        collect_student(self.db, self.cases, lambda _: generated("A"))
        export_student(self.db, self.output, self.teacher)
        self.teacher_collect()
        before = self.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
        metrics = export_student(self.db, self.output, self.teacher, require_teacher=True)
        self.assertEqual(metrics["pairing_status"], "complete")
        self.assertEqual(metrics["strict_generation"]["teacher_agreement_all_cases"], 1)
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0], before)
        # Re-export via CLI uses the frozen manifest and does not load model weights.
        result = subprocess.run([sys.executable, str(ROOT / "code/student_baseline.py"),
                                 "--export-only", "--output", str(self.output),
                                 "--teacher-run", str(self.teacher)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_incomplete_teacher_and_student_do_not_publish_full_agreement(self):
        with closing(sqlite3.connect(self.teacher / "run.sqlite")) as db:
            initialize_teacher(db, self.cases, split="validation")
        self.assertEqual(load_teacher_labels(self.teacher, self.config)[1], "incomplete")
        collect_student(self.db, self.cases[:1], lambda _: generated("A"))
        self.teacher_collect()
        metrics = export_student(self.db, self.output, self.teacher)
        self.assertTrue(metrics["provisional"])
        self.assertIsNone(metrics["strict_generation"]["teacher_agreement_all_cases"])
        self.assertEqual(metrics["n_pending"], 4)

    def test_all_cohort_teacher_pairs_selected_partition_despite_other_pending_cases(self):
        history = {**self.cases[0], "complaint_id": "history-only", "split": "history"}
        test = {**self.cases[-1], "complaint_id": "test-only", "split": "test"}
        full_cases = [history, *self.cases, test]
        with closing(sqlite3.connect(self.teacher / "run.sqlite")) as db:
            initialize_teacher(db, full_cases)
            # Only validation is complete; the unrelated partitions remain pending.
            collect(db, self.teacher / "run.sqlite", None, self.cases, 1,
                    predict=lambda *_: reply("A"))
        before = (self.teacher / "run.sqlite").read_bytes()
        labels, status = load_teacher_labels(self.teacher, self.config, required=True)
        self.assertEqual(status, "complete")
        self.assertEqual(labels, {case["complaint_id"]: "A" for case in self.cases})
        self.assertEqual((self.teacher / "run.sqlite").read_bytes(), before)
        changed = {**self.config, "cases": [dict(case) for case in self.config["cases"]]}
        changed["cases"][0]["input_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "inputs"):
            load_teacher_labels(self.teacher, changed)
        with closing(sqlite3.connect(self.teacher / "run.sqlite")) as db:
            db.execute("DELETE FROM attempts WHERE complaint_id = ?", (self.cases[0]["complaint_id"],))
            db.commit()
        self.assertEqual(load_teacher_labels(self.teacher, self.config)[1], "incomplete")

    def test_all_cohort_teacher_rejects_cross_partition_mismatches(self):
        full_cases = [dict(case) for case in self.cases]
        full_cases[0]["split"] = "history"
        with closing(sqlite3.connect(self.teacher / "run.sqlite")) as db:
            initialize_teacher(db, full_cases)
        with self.assertRaisesRegex(ValueError, "inputs"):
            load_teacher_labels(self.teacher, self.config)

    def test_pairing_rejects_prompt_order_and_input_changes(self):
        self.teacher_collect()
        for key, value in (("cases", list(reversed(self.config["cases"]))),
                           ("system_prompt", "changed prompt"),
                           ("final_instruction", "changed final reminder")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                load_teacher_labels(self.teacher, {**self.config, key: value})
        altered = [dict(c) for c in self.config["cases"]]
        altered[0]["input_sha256"] = "changed hash"
        with self.assertRaisesRegex(ValueError, "inputs"):
            load_teacher_labels(self.teacher, {**self.config, "cases": altered})
        with self.assertRaisesRegex(ValueError, "not found"):
            load_teacher_labels(self.output / "missing", self.config, required=True)

    def test_archived_pilot_pairing_is_read_only(self):
        for version in ("v1", "v2", "v3", "v4"):
            run = ROOT / "output" / f"gemma_dev_n300_s42_{version}"
            archived = read_config(run)
            config = {**archived, "backend": "student"}
            before = (run / "run.sqlite").read_bytes()
            labels, status = load_teacher_labels(run, config, required=True)
            self.assertEqual(status, "complete")
            self.assertEqual(len(labels), 300)
            self.assertEqual((run / "run.sqlite").read_bytes(), before)

    def test_local_runtime_error_pauses_without_consuming_retry_budget(self):
        with self.assertRaisesRegex(RuntimeError, "fixture interruption"):
            collect_student(self.db, self.cases, Mock(side_effect=RuntimeError("fixture interruption")))
        metrics = export_student(self.db, self.output, self.teacher)
        self.assertEqual(metrics["n_terminal"], 0)
        collect_student(self.db, self.cases, lambda _: generated("A"))
        self.assertEqual(len(attempts_by_case(self.db)["1"]), 2)  # Paused plus accepted.

    def test_export_only_teacher_cli(self):
        self.teacher_collect()
        result = subprocess.run([sys.executable, str(ROOT / "code/teacher_baseline.py"),
                                 "--export-only", "--output", str(self.teacher)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads((self.teacher / "metrics.json").read_text())["complete"])

    def test_predict_once_decodes_only_generated_tokens_and_checks_context(self):
        import torch

        tokenizer = Mock(eos_token_id=9)
        tokenizer.apply_chat_template.return_value = {"input_ids": torch.tensor([[4, 5, 6]])}
        tokenizer.decode.return_value = "C"
        model = Mock()
        model.config.max_position_embeddings = 32
        model.generate.return_value = SimpleNamespace(
            sequences=torch.tensor([[4, 5, 6, 2, 9]]),
            scores=(torch.tensor([[0., 0., 10., 0., 0.]]),))
        settings = SimpleNamespace(max_new_tokens=16)
        result = predict_once(model, tokenizer, settings, list(range(5)), [{"role": "user", "content": "test"}])
        self.assertEqual(result, generated("C", "C") | {"prompt_tokens": 3})
        tokenizer.decode.assert_called_once_with([2], skip_special_tokens=False,
                                                clean_up_tokenization_spaces=False)
        model.config.max_position_embeddings = 18
        with self.assertRaisesRegex(ValueError, "context window"):
            predict_once(model, tokenizer, settings, list(range(5)), [])

    def test_qwen_profile_keeps_shared_clipping_and_teacher_pairing(self):
        qwen = student_config(self.cases, "validation", load_prompt("v3"), "qwen3.5-0.8b")
        self.assertEqual(qwen["model"], "Qwen/Qwen3.5-0.8B")
        self.assertEqual(qwen["chat_template_kwargs"], {"enable_thinking": False})
        self.assertNotEqual(qwen["model_revision"], qwen["revision"])
        self.assertEqual(qwen["cases"], self.config["cases"])
        self.assertEqual(qwen["tokenizer"], self.config["tokenizer"])
        self.teacher_collect()
        self.assertEqual(load_teacher_labels(self.teacher, qwen)[1], "complete")
        self.assertNotEqual(student_output("smollm2-135m", "validation", "v3"),
                            student_output("qwen3.5-0.8b", "validation", "v3"))
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            initialize(self.db, qwen)

    def test_qwen_template_options_and_multiple_end_tokens(self):
        import torch

        tokenizer = Mock(eos_token_id=9)
        tokenizer.apply_chat_template.return_value = {"input_ids": torch.tensor([[4]])}
        tokenizer.decode.return_value = "A"
        model = SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=32)),
                                generate=Mock(return_value=SimpleNamespace(
                                    sequences=torch.tensor([[4, 0, 8]]),
                                    scores=(torch.tensor([[10., 0., 0., 0., 0.]]),))))
        result = predict_once(model, tokenizer, SimpleNamespace(max_new_tokens=16, eos_token_id=[8, 9]),
                              list(range(5)), [], {"enable_thinking": False})
        self.assertEqual(result["raw_reply"], "A")
        self.assertFalse(tokenizer.apply_chat_template.call_args.kwargs["enable_thinking"])
        tokenizer.decode.assert_called_once_with([0], skip_special_tokens=False,
                                                clean_up_tokenization_spaces=False)
        self.assertEqual(generation_stop_ids(248044, 248046), [248046, 248044])
        self.assertEqual(generation_stop_ids([8, 9], 9), [9, 8])
        self.assertEqual(generation_stop_ids(None, 9), [9])

    def test_gemma_profile_preserves_inputs_and_has_an_independent_ledger(self):
        gemma = student_config(self.cases, "validation", load_prompt("v3"), "gemma3-270m-it")
        self.assertEqual(gemma["model"], "google/gemma-3-270m-it")
        self.assertEqual(gemma["model_revision"], "ac82b4e820549b854eebf28ce6dedaf9fdfa17b3")
        self.assertEqual(gemma["chat_template_kwargs"], {})
        self.assertEqual(gemma["generation_stop_tokens"], ["<end_of_turn>"])
        self.assertEqual(gemma["cases"], self.config["cases"])
        self.assertEqual(gemma["tokenizer"], self.config["tokenizer"])
        self.teacher_collect()
        self.assertEqual(load_teacher_labels(self.teacher, gemma)[1], "complete")
        paths = {student_output(name, "validation", "v3") for name in STUDENT_MODELS}
        self.assertEqual(len(paths), 3)
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            initialize(self.db, gemma)

    def test_gemma_loader_stops_on_native_end_of_turn_and_decodes_a_letter(self):
        import torch

        tokenizer = Mock(eos_token_id=1, pad_token_id=0)
        tokenizer.encode.side_effect = lambda letter, **_: [ord(letter) - ord("A") + 10]
        tokenizer.get_vocab.return_value = {"<end_of_turn>": 106}
        tokenizer.convert_tokens_to_ids.return_value = 106
        tokenizer.apply_chat_template.return_value = {"input_ids": torch.tensor([[2]])}
        tokenizer.decode.return_value = "A"
        model = Mock(device=torch.device("cpu"))
        model.to.return_value.eval.return_value = model
        model.generation_config = SimpleNamespace(eos_token_id=1)
        model.config = SimpleNamespace(max_position_embeddings=32768)
        logits = torch.zeros((1, 15))
        logits[0, 10] = 10
        model.generate.return_value = SimpleNamespace(sequences=torch.tensor([[2, 10, 106]]),
                                                       scores=(logits,))
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer) as loader, \
                patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model) as weights:
            loaded, native, settings, label_ids = load_student("gemma3-270m-it")
        self.assertIs(loaded, model)
        loader.assert_called_once_with("google/gemma-3-270m-it",
                                       revision=STUDENT_MODELS["gemma3-270m-it"]["revision"],
                                       cache_dir=ROOT / ".cache/huggingface")
        self.assertEqual(weights.call_args.kwargs["revision"], loader.call_args.kwargs["revision"])
        self.assertEqual(settings.eos_token_id, [1, 106])
        self.assertEqual(settings.max_new_tokens, 16)
        messages = build_messages("A payment was applied incorrectly.", load_prompt("v3"))
        result = predict_once(loaded, native, settings, label_ids, messages)
        self.assertEqual(result["raw_reply"], "A")
        self.assertEqual(result["first_token_choice"], "A")
        self.assertEqual(tokenizer.apply_chat_template.call_args.args[0], messages)
        tokenizer.decode.assert_called_once_with([10], skip_special_tokens=False,
                                                clean_up_tokenization_spaces=False)

    def test_gemma_gated_access_error_explains_local_authentication(self):
        with patch("transformers.AutoTokenizer.from_pretrained", side_effect=OSError("gated repo")), \
                patch("transformers.AutoModelForCausalLM.from_pretrained") as weights:
            with self.assertRaisesRegex(RuntimeError, "HF_TOKEN.*hf auth login"):
                load_student("gemma3-270m-it")
            weights.assert_not_called()

    def test_archived_ledger_cannot_be_migrated_by_a_new_run(self):
        with closing(sqlite3.connect(":memory:")) as db:
            db.execute("CREATE TABLE run (id INTEGER PRIMARY KEY, config TEXT NOT NULL)")
            db.execute("INSERT INTO run VALUES (1, ?)", (json.dumps({"split": "development"}),))
            db.commit()
            with self.assertRaisesRegex(ValueError, "archived pilot"):
                initialize(db, self.config)
            self.assertEqual(db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall(),
                             [("run",)])

    def test_failed_teacher_is_included_in_agreement_denominator(self):
        with closing(sqlite3.connect(self.teacher / "run.sqlite")) as db:
            initialize_teacher(db, self.cases, split="validation")
            collect(db, self.teacher / "run.sqlite", None, self.cases, 1,
                    predict=lambda _, text: reply("invalid" if text == "A" else "A"))
        collect_student(self.db, self.cases, lambda _: generated("A"))
        metrics = export_student(self.db, self.output, self.teacher)
        self.assertEqual(metrics["strict_generation"]["teacher_agreement_all_cases"], 4 / 5)


class TestTeacherEntrypoint(unittest.TestCase):
    def test_interrupted_run_exports_and_complete_resume_needs_no_credentials(self):
        import teacher_baseline

        cases = fixture_cases()
        for case, split in zip(cases, ("history", "history", "validation", "test", "test")):
            case["split"] = split
        rows = [dict(case) for case in cases]
        calls = []

        def predict(client, text, prompt):
            calls.append(text)
            if text == "B":
                raise KeyboardInterrupt()
            return reply("A")

        with TemporaryDirectory() as tmp, \
                patch.object(teacher_baseline, "load_experiment", return_value=rows), \
                patch.object(teacher_baseline, "prepare_cases", return_value=cases), \
                patch.object(teacher_baseline, "make_client") as client, \
                patch("sys.argv", ["teacher_baseline", "--run", "--output", tmp]), \
                patch.object(teacher_baseline, "request_label", side_effect=predict):
            with self.assertRaises(KeyboardInterrupt):
                teacher_baseline.main()
            metrics = json.loads((Path(tmp) / "metrics.json").read_text())
            self.assertEqual(metrics["n_terminal"], 1)
            self.assertEqual(metrics["n_pending"], 4)
            with patch.object(teacher_baseline, "request_label", return_value=reply("A")):
                teacher_baseline.main()
            self.assertTrue(json.loads((Path(tmp) / "metrics.json").read_text())["complete"])
            stored = read_config(tmp)
            self.assertIsNone(stored["split"])
            self.assertEqual({case["split"] for case in stored["cases"]}, {"history", "validation", "test"})
            client.reset_mock()
            teacher_baseline.main()
            client.assert_not_called()

    def test_schema_two_student_can_export_without_changing_its_ledger(self):
        source = ROOT / "output/gemma3_270m_dev_n300_s42_v3_cuda/run.sqlite"
        original = source.read_bytes()
        with TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "run.sqlite"
            ledger.write_bytes(original)
            stored = read_config(tmp)
            self.assertEqual(stored["schema_version"], 2)
            result = subprocess.run(
                [sys.executable, str(ROOT / "code/student_baseline.py"), "--export-only", "--output", tmp,
                 "--teacher-run", str(ROOT / "output/gemma_dev_n300_s42_v3")],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(ledger.read_bytes(), original)
            self.assertEqual(json.loads((Path(tmp) / "config.json").read_text()), stored)
            self.assertTrue(json.loads((Path(tmp) / "metrics.json").read_text())["complete"])


if __name__ == "__main__":
    unittest.main()
