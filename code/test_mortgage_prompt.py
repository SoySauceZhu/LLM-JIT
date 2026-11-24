import json
import sqlite3
import unittest
from contextlib import closing
from unittest.mock import Mock

from teacher_baseline import initialize
from mortgage_prompt import DEFAULT_VERSION, LABELS, SUPPORTED_VERSIONS, build_messages, load_prompt
from teacher_client import request_label


class TestMortgagePrompt(unittest.TestCase):
    def test_v3_is_the_shared_default_and_archived_versions_remain_available(self):
        self.assertEqual(DEFAULT_VERSION, "v3")
        self.assertEqual(SUPPORTED_VERSIONS, ("v1", "v2", "v3", "v4"))
        for version in ("v2", "v3", "v4"):
            self.assertEqual(load_prompt(version)["prompt_file"], f"mortgage_{version}.toml")

    def test_served_few_shot_messages_can_be_replayed_from_the_ledger(self):
        client = Mock()
        text = "The amount on my escrow statement is wrong."
        request_label(client, text)
        sent = client.chat.completions.create.call_args.kwargs["messages"]
        with closing(sqlite3.connect(":memory:")) as db:
            initialize(db, [], 42)
            stored = json.loads(db.execute("SELECT config FROM run").fetchone()[0])
        self.assertEqual(sent, build_messages(text, stored))
        self.assertEqual([m["role"] for m in sent], ["system"] + ["user", "assistant"] * 5 + ["user"])
        self.assertCountEqual([m["content"] for m in sent if m["role"] == "assistant"], LABELS)
        self.assertIn(text, sent[-1]["content"])
        self.assertTrue(sent[-1]["content"].split(text, 1)[1].strip())

    def test_archived_zero_shot_messages_still_replay(self):
        v1 = load_prompt("v1")
        self.assertEqual(set(v1), {"prompt_file", "system_prompt", "user_template"})
        self.assertEqual(build_messages("example", v1), [
            {"role": "system", "content": v1["system_prompt"]},
            {"role": "user", "content": "<complaint>\nexample\n</complaint>"},
        ])
