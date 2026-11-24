"""Offline checks for Teacher selection, request settings and model isolation."""

import json
import sqlite3
import subprocess
import sys
import unittest
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from mortgage_prompt import ROOT, build_messages, load_prompt
from run_store import initialize, make_config
from student_baseline import DEFAULT_STUDENT_MODEL, collect_student, student_config
from teacher_baseline import initialize as initialize_teacher
from teacher_client import (
    BASE_URL, DECODING, DEFAULT_TEACHER_MODEL, REQUEST_TIMEOUT_SECONDS, SDK_MAX_RETRIES,
    TEACHER_MODELS, request_label, teacher_output,
)
from teacher_collection import collect
from test_baselines import fixture_cases, generated
from test_teacher_collection import reply


class TestTeacherProfiles(unittest.TestCase):
    def test_requests_use_exact_models_and_only_deepseek_disables_thinking(self):
        client = Mock()
        prompt = load_prompt("v3")
        request_label(client, "A mortgage complaint.", prompt)
        default_request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(default_request["model"], "deepseek-ai/DeepSeek-V4.1-Flash-1017")
        self.assertEqual(default_request["extra_body"], {"enable_thinking": False})
        self.assertEqual(default_request["temperature"], 0)
        self.assertEqual(default_request["max_tokens"], 16)
        self.assertEqual(default_request["messages"], build_messages("A mortgage complaint.", prompt))
        request_label(client, "A mortgage complaint.", prompt, teacher_model="gemma4-31b-it")
        gemma_request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(gemma_request["model"], "google/gemma-4-31B-it")
        self.assertNotIn("extra_body", gemma_request)
        self.assertEqual(gemma_request["messages"], default_request["messages"])

    def test_frozen_config_records_deepseek_settings_and_refuses_model_changes(self):
        with closing(sqlite3.connect(":memory:")) as db:
            stored = initialize_teacher(db, fixture_cases())
            self.assertEqual(stored["model"], TEACHER_MODELS[DEFAULT_TEACHER_MODEL]["model"])
            self.assertEqual(stored["request_extra_body"], {"enable_thinking": False})
            initialize_teacher(db, fixture_cases())
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                initialize_teacher(db, fixture_cases(), teacher_model="gemma4-31b-it")

    def test_optional_gemma_can_resume_preexisting_schema_three_configuration(self):
        cases, prompt = fixture_cases(), load_prompt("v3")
        legacy = make_config(cases, None, prompt, backend="teacher", model="google/gemma-4-31B-it",
                             decoding=DECODING, base_url=BASE_URL, max_attempts=2,
                             request_timeout_seconds=REQUEST_TIMEOUT_SECONDS, sdk_max_retries=SDK_MAX_RETRIES)
        with closing(sqlite3.connect(":memory:")) as db:
            initialize(db, legacy)
            stored = initialize_teacher(db, cases, prompt, teacher_model="gemma4-31b-it")
        self.assertEqual(stored, legacy)

    def test_collector_without_override_uses_ledger_teacher_and_prompt(self):
        for name in TEACHER_MODELS:
            with self.subTest(teacher=name), TemporaryDirectory() as tmp:
                ledger = Path(tmp) / "run.sqlite"
                cases = fixture_cases()[:1]
                with closing(sqlite3.connect(ledger)) as db:
                    config = initialize_teacher(db, cases, load_prompt("v4"), teacher_model=name)
                    with patch("teacher_collection.request_label", return_value=reply("A")) as request:
                        collect(db, ledger, None, cases, 1)
                    self.assertEqual(request.call_args.kwargs["teacher_model"], name)
                    self.assertEqual(request.call_args.kwargs["prompt"], config)

    def test_cli_previews_show_defaults_and_isolated_teacher_destinations(self):
        paths = {teacher_output(name, None, "v3") for name in TEACHER_MODELS}
        self.assertEqual(len(paths), 2)
        self.assertNotIn(ROOT / "output/baseline_v3/teacher_all", paths)
        for flags, expected in (([], DEFAULT_TEACHER_MODEL),
                                (["--teacher-model", "gemma4-31b-it"], "gemma4-31b-it")):
            with self.subTest(teacher=expected):
                result = subprocess.run([sys.executable, str(ROOT / "code/teacher_baseline.py"), *flags],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"model={expected}", result.stdout)
                self.assertIn(str(teacher_output(expected, None, "v3")), result.stdout)
                self.assertIn("Preview only", result.stdout)
        result = subprocess.run([sys.executable, str(ROOT / "code/student_baseline.py")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("model=gemma3-270m-it", result.stdout)
        self.assertEqual(DEFAULT_STUDENT_MODEL, "gemma3-270m-it")
        self.assertEqual(student_config(fixture_cases(), "validation", load_prompt("v3"))["model"],
                         "google/gemma-3-270m-it")

    def test_smoke_propagates_selected_teacher_without_changing_prompt(self):
        import teacher_smoke

        for name in TEACHER_MODELS:
            case = {**fixture_cases()[0], "split": "history"}
            with self.subTest(teacher=name), \
                    patch.object(teacher_smoke, "load_experiment", return_value=[case]), \
                    patch.object(teacher_smoke, "prepare_cases", return_value=[case]), \
                    patch.object(teacher_smoke, "make_client"), \
                    patch.object(teacher_smoke, "request_label", return_value=reply("A")) as request, \
                    patch("sys.argv", ["teacher_smoke", "--teacher-model", name]):
                teacher_smoke.main()
                self.assertEqual(request.call_args.kwargs["teacher_model"], name)
                self.assertEqual(request.call_args.args[2], load_prompt("v3"))


class TestStudentTeacherSelection(unittest.TestCase):
    def test_reexport_selects_each_teacher_without_rerunning_or_changing_student(self):
        import student_baseline

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = fixture_cases()
            paths = {}
            for name, label in ((DEFAULT_TEACHER_MODEL, "A"), ("gemma4-31b-it", "B")):
                path = root / name
                path.mkdir()
                paths[name] = path
                with closing(sqlite3.connect(path / "run.sqlite")) as db:
                    initialize_teacher(db, cases, teacher_model=name)
                    collect(db, path / "run.sqlite", None, cases, 1,
                            predict=lambda *_, label=label: reply(label))
            student = root / "student"
            student.mkdir()
            ledger = student / "run.sqlite"
            with closing(sqlite3.connect(ledger)) as db:
                initialize(db, student_config(cases, "validation", load_prompt("v3")))
                collect_student(db, cases, lambda _: generated("A"))
            original = ledger.read_bytes()
            selections = [([], DEFAULT_TEACHER_MODEL),
                          (["--teacher-model", "gemma4-31b-it"], "gemma4-31b-it"),
                          (["--teacher-run", str(paths["gemma4-31b-it"])], "gemma4-31b-it")]
            for flags, expected in selections:
                with self.subTest(flags=flags), \
                        patch.object(student_baseline, "teacher_output", side_effect=lambda model, *_: paths[model]), \
                        patch.object(student_baseline, "load_student") as load, \
                        patch("sys.argv", ["student_baseline", "--export-only", "--output", str(student), *flags]):
                    student_baseline.main()
                    load.assert_not_called()
                    metrics = json.loads((student / "metrics.json").read_text())
                    self.assertEqual(metrics["teacher_model"], TEACHER_MODELS[expected]["model"])
                    self.assertEqual(metrics["strict_generation"]["teacher_agreement_all_cases"],
                                     1 if expected == DEFAULT_TEACHER_MODEL else 0)
                    self.assertEqual(ledger.read_bytes(), original)
            with patch("sys.argv", ["student_baseline", "--export-only", "--output", str(student),
                                    "--teacher-model", DEFAULT_TEACHER_MODEL,
                                    "--teacher-run", str(paths["gemma4-31b-it"])]):
                with self.assertRaisesRegex(ValueError, "Teacher model differs"):
                    student_baseline.main()
