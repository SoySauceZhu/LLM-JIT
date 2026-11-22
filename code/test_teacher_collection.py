"""Offline regression tests for out-of-order responses and concurrent resume."""

import csv
import sqlite3
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier, Event
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from openai import RateLimitError

from baseline_gemma import export_results, initialize
from teacher_collection import collect, single_runner


def reply(label):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=label))],
                           id="fixture", model="fixture", usage=None)


class TestTeacherCollection(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.output = Path(tmp.name)
        self.path = self.output / "run.sqlite"
        self.db = sqlite3.connect(self.path)
        self.addCleanup(self.db.close)
        self.cases = [dict(complaint_id=str(i), text=label, reference_label=label,
                           clipped=False, student_tokens_before_clip=1)
                      for i, label in enumerate("ABCDE")]
        initialize(self.db, self.cases, 42)

    def test_overlap_keeps_order_and_retry_budget(self):
        third_started = Event()
        calls = Counter()

        def predict(client, text):
            calls[text] += 1
            if text == "A":
                # With two workers, B must finish before C starts.
                self.assertTrue(third_started.wait(5))
            if text == "C":
                third_started.set()
                return reply("invalid")
            return reply(text)

        # B has one committed invalid response from a previous serial run.
        with self.db:
            self.db.execute("""INSERT INTO attempts (complaint_id, status, seconds)
                VALUES ('1', 'invalid', 0)""")
        with patch("teacher_collection.request_label", side_effect=predict):
            collect(self.db, self.path, None, self.cases, 2)
            collect(self.db, self.path, None, self.cases, 4)
        self.assertEqual(calls, dict(A=1, B=1, C=2, D=1, E=1))
        accepted = [row[0] for row in self.db.execute(
            "SELECT label FROM attempts WHERE status='ok' ORDER BY id")]
        self.assertLess(accepted.index("B"), accepted.index("A"))
        export_results(self.db, self.cases, self.output)
        with (self.output / "predictions.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([r["teacher_label"] for r in rows], ["A", "B", "", "D", "E"])
        self.assertEqual([r["attempts"] for r in rows], ["1", "2", "2", "1", "1"])

    def test_rate_limit_drains_active_request_before_resume(self):
        started = Barrier(2)
        stopped = Event()

        def predict(client, text):
            self.assertIn(text, "AB")
            started.wait(timeout=5)
            if text == "A":
                request = httpx.Request("POST", "https://example.invalid")
                raise RateLimitError("fixture limit", body=None, response=httpx.Response(
                    429, request=request, headers={"Retry-After": "10"}))
            self.assertTrue(stopped.wait(5))
            return reply(text)

        with patch("teacher_collection.Event", return_value=stopped), \
                patch("teacher_collection.request_label", side_effect=predict):
            with self.assertRaisesRegex(SystemExit, "Retry-After: 10"):
                collect(self.db, self.path, None, self.cases, 2)
        self.assertEqual(dict(self.db.execute("SELECT complaint_id, status FROM attempts")),
                         {"0": "paused", "1": "ok"})
        with patch("teacher_collection.request_label", side_effect=lambda _, text: reply(text)) as api:
            collect(self.db, self.path, None, self.cases, 1)
        self.assertEqual([call.args[1] for call in api.call_args_list], list("ACDE"))
        self.assertEqual(dict(self.db.execute(
            "SELECT concurrency, COUNT(*) FROM attempts GROUP BY concurrency")), {1: 4, 2: 2})

    def test_only_one_collector_per_run(self):
        with single_runner(self.output):
            with self.assertRaisesRegex(SystemExit, "Another baseline runner"):
                with single_runner(self.output):
                    self.fail("acquired a second lock")
        with single_runner(self.output):
            self.assertTrue(self.path.exists())
