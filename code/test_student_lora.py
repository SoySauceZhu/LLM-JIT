"""Regression checks for completion boundaries, token weights and frozen base weights."""

import csv
import unittest
from copy import deepcopy
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import torch
from torch.nn.functional import cross_entropy
from transformers import Gemma3ForCausalLM, Gemma3TextConfig, Qwen3Config, Qwen3ForCausalLM

from mortgage_prompt import load_prompt
from student_baseline import STUDENT_MODELS, adapter_identity, load_student
from student_lora import (
    Example, add_lora, collate, completion_loss, encode_answer, prepare_examples, read_training_rows, save_run,
)


def tiny_gemma():
    return Gemma3ForCausalLM(Gemma3TextConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=32, layer_types=["full_attention"], pad_token_id=0,
    ))


def tiny_qwen():
    return Qwen3ForCausalLM(Qwen3Config(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=32, pad_token_id=0, bos_token_id=2, eos_token_id=7,
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

    def test_qwen_training_uses_the_non_thinking_generation_prefix(self):
        case = {"complaint_id": "123", "text": "A hardship complaint", "clipped": False}
        tokenizer = Mock(pad_token_id=0)
        tokenizer.get_vocab.return_value = {"<|im_end|>": 7}
        tokenizer.convert_tokens_to_ids.return_value = 7
        prefix = [2, 5, 6, 8]  # Includes the template's empty thinking block.
        tokenizer.apply_chat_template.side_effect = [{"input_ids": prefix}, {"input_ids": [*prefix, 14, 7, 9]}]
        tokenizer.decode.side_effect = TemplateFixture().decode
        with patch("student_lora.prepare_cases", return_value=[case]):
            examples, stop, clipped = prepare_examples([(case, "D")], tokenizer, 32, "qwen3-0.6b")
        tokenizer.convert_tokens_to_ids.assert_called_once_with("<|im_end|>")
        self.assertEqual((stop, clipped), (7, 0))
        self.assertEqual(examples[0].input_ids, [*prefix, 14, 7])
        self.assertEqual(collate(examples, 0, "cpu")["labels"].tolist(), [[-100] * len(prefix) + [14, 7]])
        for call, generate in zip(tokenizer.apply_chat_template.call_args_list, (True, False), strict=True):
            self.assertEqual(call.kwargs, {"add_generation_prompt": generate, "enable_thinking": False})

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
        examples = [Example("1", "A", [2, 3, 4, 10, 7]), Example("2", "D", [2, 4, 14, 7])]
        padded = collate(examples, 0, "cpu")
        self.assertEqual(padded["labels"].tolist(), [[-100, -100, -100, 10, 7], [-100, -100, -100, 14, 7]])
        self.assertEqual(padded["attention_mask"].tolist(), [[1, 1, 1, 1, 1], [0, 1, 1, 1, 1]])
        for factory in (tiny_gemma, tiny_qwen):
            with self.subTest(model=factory.__name__), torch.no_grad():
                model = factory().eval()
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

    @unittest.skipUnless(find_spec("peft"), "requires the lora extra")
    def test_qwen_adapter_round_trip_uses_the_selected_base_model(self):
        student = "qwen3-0.6b"
        fresh = tiny_qwen()
        fresh.name_or_path = STUDENT_MODELS[student]["model"]
        lora, params = add_lora(deepcopy(fresh), 2, 4, 0, student_model=student)
        example = Example("1", "D", [2, 4, 14, 7])
        batch = collate([example], 0, "cpu")
        completion_loss(lora, batch, 2).backward()
        torch.optim.AdamW(params, lr=1e-2).step()
        tokenizer = Mock(eos_token_id=7, pad_token_id=0)
        tokenizer.encode.side_effect = lambda label, **_: [10 + "ABCDE".index(label)]
        tokenizer.get_vocab.return_value = {"<|im_end|>": 7}
        tokenizer.convert_tokens_to_ids.return_value = 7
        with TemporaryDirectory() as temp:
            output = Path(temp) / "adapter"
            save_run(output, lora, [example], {}, student_model=student)
            with self.assertRaisesRegex(ValueError, "pinned Student"):
                adapter_identity(output, "gemma3-270m-it")
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
                    patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=fresh):
                restored, _, settings, _ = load_student(student, adapter=output)
        self.assertIn(7, settings.eos_token_id)
        self.assertFalse(any(p.requires_grad for p in restored.parameters()))
        with torch.no_grad():
            torch.testing.assert_close(restored(**batch).logits, lora.eval()(**batch).logits)


if __name__ == "__main__":
    unittest.main()
