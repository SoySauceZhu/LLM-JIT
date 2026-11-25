"""Regression checks for completion boundaries, token weights and frozen base weights."""

import csv
import unittest
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from torch.nn.functional import cross_entropy
from transformers import Gemma3ForCausalLM, Gemma3TextConfig

from mortgage_prompt import load_prompt
from student_lora import Example, add_lora, collate, completion_loss, encode_answer, read_training_rows


def tiny_gemma():
    return Gemma3ForCausalLM(Gemma3TextConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=32, layer_types=["full_attention"], pad_token_id=0,
    ))


class TemplateFixture:
    def __init__(self, answer=None, prefix=None):
        self.answer = [14, 7, 9] if answer is None else answer
        self.prefix = [2, 5, 6] if prefix is None else prefix

    def apply_chat_template(self, messages, *, add_generation_prompt):
        if add_generation_prompt:
            return {"input_ids": [2, 5, 6]}
        return {"input_ids": self.prefix + self.answer}

    def decode(self, ids, **kwargs):
        return "".join({14: "D", 7: "<end_of_turn>", 9: "\n", 10: "A", 1: "<eos>"}[i] for i in ids)


class StudentLoRATest(unittest.TestCase):
    def test_template_checks_label_prefix_and_native_terminator(self):
        case = {"complaint_id": "123", "text": "A hardship complaint"}
        prompt = load_prompt("v3")
        encoded = encode_answer(TemplateFixture(), prompt, case, "D", 7, 8)
        self.assertEqual(encoded.input_ids, [2, 5, 6, 14, 7])
        self.assertEqual(encoded.audit()["first_supervised_index"], 3)
        for fixture in (TemplateFixture(answer=[14, 1, 9]), TemplateFixture(answer=[10, 7, 9]),
                        TemplateFixture(answer=[7]), TemplateFixture(answer=[14, 7, 10]),
                        TemplateFixture(prefix=[2, 5, 8])):
            with self.subTest(answer=fixture.answer, prefix=fixture.prefix), self.assertRaises(ValueError):
                encode_answer(fixture, prompt, case, "D", 7, 8)

    def test_load_uses_teacher_labels_and_rejects_held_out_ids(self):
        rows = [dict(complaint_id="1", split="history", text="first", reference_label="A"),
                dict(complaint_id="2", split="history", text="second", reference_label="A"),
                dict(complaint_id="3", split="test", text="held-out", reference_label="A")]
        with TemporaryDirectory() as temp, patch("student_lora.load_cohort", return_value=rows):
            output = Path(temp) / "train.csv"
            with output.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=[*rows[0], "teacher_label"])
                writer.writeheader()
                writer.writerow({**rows[0], "teacher_label": "D"})
                writer.writerow({**rows[1], "teacher_label": ""})
            self.assertEqual(read_training_rows(output), ([(rows[0], "D")], 1))
            with output.open("a", newline="") as file:
                csv.DictWriter(file, fieldnames=[*rows[0], "teacher_label"]).writerow(
                    {**rows[2], "split": "history", "teacher_label": "A"},
                )
            with self.assertRaisesRegex(ValueError, "frozen history input"):
                read_training_rows(output)

    def test_tail_loss_matches_native_full_loss_and_weights_microbatches(self):
        model = tiny_gemma().eval()
        examples = [Example("1", "A", [2, 3, 4, 10, 7]), Example("2", "D", [2, 4, 14, 7])]
        padded = collate(examples, 0, "cpu")
        self.assertEqual(padded["labels"].tolist(), [[-100, -100, -100, 10, 7], [-100, -100, -100, 14, 7]])
        self.assertEqual(padded["attention_mask"].tolist(), [[1, 1, 1, 1, 1], [0, 1, 1, 1, 1]])
        with torch.no_grad():
            full = model(**padded, use_cache=False)
            explicit = cross_entropy(full.logits[:, -3:-1].reshape(-1, 32), torch.tensor([10, 7, 14, 7]))
            tail = completion_loss(model, padded, 4)
            accumulated = sum(completion_loss(model, collate([e], 0, "cpu"), 4) for e in examples)
        torch.testing.assert_close(full.loss, explicit)
        torch.testing.assert_close(tail, explicit)
        torch.testing.assert_close(accumulated, explicit)

    @unittest.skipUnless(find_spec("peft"), "requires the lora extra")
    def test_optimizer_updates_only_lora(self):
        model, params = add_lora(tiny_gemma(), rank=2, alpha=4, dropout=0)
        before = {name: param.detach().clone() for name, param in model.named_parameters()}
        batch = collate([Example("1", "D", [2, 4, 14, 7])], 0, "cpu")
        loss = completion_loss(model, batch, 2)
        loss.backward()
        torch.optim.AdamW(params, lr=1e-2, weight_decay=0).step()
        changed = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                self.assertIsNone(param.grad)
                torch.testing.assert_close(param, before[name], rtol=0, atol=0)
            elif not torch.equal(param, before[name]):
                changed.append(name)
        self.assertTrue(changed)
        self.assertTrue(all(".lora_" in name for name in changed))


if __name__ == "__main__":
    unittest.main()
