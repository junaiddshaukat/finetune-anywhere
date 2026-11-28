"""Behavior tests using temporary data, never fictional public showcase entries."""

import json
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import yaml

from common import ROOT, read_rows
from convert_to_chat import convert
from memory_calc import estimate
from validate_dataset import validate
from validate_showcase import is_late, validate as validate_entry
from build_showcase import build


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def chat(prompt="Test input", answer="Test answer"):
    return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]}


def test_memory_formula():
    weights, kv, overhead = estimate(1.7, 4, 4096)
    assert weights == pytest.approx(0.935)
    assert kv == pytest.approx(2 * 28 * 4 * 128 * 4096 * 2 / 1e9)
    assert overhead == 1.5


@pytest.mark.parametrize("recipe", sorted((ROOT / "recipes").glob("*/data")))
def test_recipe_splits(recipe):
    seen = set()
    manifest = json.loads((recipe / "manifest.json").read_text())
    assert manifest["license"] == "MIT"
    for split, count in manifest["splits"].items():
        errors, lengths, prompts = validate(recipe / f"{split}.jsonl")
        assert not errors
        assert len(lengths) == count
        assert not seen & prompts
        seen |= prompts


@pytest.mark.parametrize(
    "row,expected",
    [
        ({"messages": []}, "messages must"),
        (
            {"messages": [{"role": "assistant", "content": "a"}, {"role": "user", "content": "b"}]},
            "expected role",
        ),
        (chat(answer=" "), "nonempty"),
        ({"messages": [{"role": "user", "content": "a"}, 5]}, "expected role"),
    ],
)
def test_dataset_errors(tmp_path, row, expected):
    path = tmp_path / "bad.jsonl"
    write_jsonl(path, [row])
    assert any(expected in e for e in validate(path)[0])


def test_duplicate_and_invalid_json(tmp_path):
    path = tmp_path / "data.jsonl"
    write_jsonl(path, [chat(), chat()])
    assert any("duplicate" in e for e in validate(path)[0])
    path.write_text("{}\nnot json\n")
    assert ":2:" in validate(path)[0][0]
    path.write_text("")
    with pytest.raises(ValueError, match="no examples"):
        read_rows(path)


def test_cross_split_leakage_cli(tmp_path):
    a, b = tmp_path / "train.jsonl", tmp_path / "eval.jsonl"
    write_jsonl(a, [chat()])
    write_jsonl(b, [chat(answer="Different target")])
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/validate_dataset.py"), str(a), str(b)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "leakage" in result.stderr


@pytest.mark.parametrize(
    "kind,content",
    [
        ("csv", 'question,answer\n"a,b","c,d"\n'),
        ("json", json.dumps([{"question": "a,b", "answer": "c,d"}])),
        ("text", "Q: a,b\nA: c,d"),
    ],
)
def test_converter(tmp_path, kind, content):
    path = tmp_path / "input"
    path.write_text(content)
    assert convert(path, kind) == [chat("a,b", "c,d")]


def test_converter_rejects_unpaired_notes(tmp_path):
    path = tmp_path / "notes"
    path.write_text("An unlabelled source document")
    with pytest.raises(ValueError, match="Q:"):
        convert(path, "text")


def manifest(event="community", slug="test-entry"):
    return {
        "project_name": "Temporary unit-test fixture",
        "slug": slug,
        "event": event,
        "members": [{"github": "test-user"}],
        "one_liner": "Test table escaping | safe",
        "base_model": "unit-test/base",
        "training_tool": "trl",
        "training_platform": "other",
        "method": "lora",
        "dataset": {"description": "Fixture data", "size": 10},
        "model_link": "https://huggingface.co/test-user/test-model",
        "quantization": "Q4_K_M",
        "runtime": "llama.cpp",
        "hardware_tested": "Test fixture only",
        "demo_video": "https://example.org/demo",
        "before_after": {
            "prompt": "Fixture prompt",
            "base_output": "Fixture base",
            "finetuned_output": "Fixture tuned",
        },
    }


def make_entry(root, event="community", slug="test-entry"):
    folder = root / "showcase" / event / slug
    folder.mkdir(parents=True)
    (folder.parent / "README.md").write_text("# Event\n\n<!-- SHOWCASE:START -->\n\n<!-- SHOWCASE:END -->\n")
    (folder / "model.yaml").write_text(yaml.safe_dump(manifest(event, slug)))
    (folder / "README.md").write_text("Temporary fixture for validation. " * 8)
    return folder


def changed(folder, root):
    return [str((folder / name).relative_to(root)) for name in ["model.yaml", "README.md"]]


def test_valid_entry(tmp_path):
    folder = make_entry(tmp_path)
    result = validate_entry(tmp_path, changed(folder, tmp_path), "2026-11-20T12:00:00Z")
    assert result["passed"], result


def test_broken_entry_readable_errors(tmp_path):
    folder = make_entry(tmp_path)
    data = manifest()
    data["slug"] = "wrong"
    data["model_link"] = "http://example.org/model"
    data["runtime"] = "imaginary"
    (folder / "model.yaml").write_text(yaml.safe_dump(data))
    result = validate_entry(tmp_path, changed(folder, tmp_path) + ["README.md"], "2026-11-20T12:00:00Z")
    assert not result["passed"]
    assert any("slug must" in e for e in result["errors"])
    assert any("model_link" in e for e in result["errors"])
    assert any("runtime" in e for e in result["errors"])
    assert any("change only" in e for e in result["errors"])


def test_exactly_one_folder(tmp_path):
    first = make_entry(tmp_path)
    second = make_entry(tmp_path, slug="another-entry")
    result = validate_entry(
        tmp_path, changed(first, tmp_path) + changed(second, tmp_path), "2026-11-20T12:00:00Z"
    )
    assert not result["passed"]
    assert "exactly one" in " ".join(result["errors"])


def test_file_and_template_rejection(tmp_path):
    folder = make_entry(tmp_path)
    (folder / "model.gguf").write_bytes(b"not weights")
    (folder / "large.txt").write_bytes(b"x" * 5_000_001)
    (folder / "README.md").write_text((ROOT / "showcase/_template/README.md").read_text())
    result = validate_entry(tmp_path, changed(folder, tmp_path), "2026-11-20T12:00:00Z")
    assert not result["passed"]
    assert any("5 MB" in e for e in result["errors"])
    assert any("weights" in e for e in result["errors"])
    assert any("completed README" in e for e in result["errors"])


def test_symlink_rejected(tmp_path):
    folder = make_entry(tmp_path)
    (folder / "external").symlink_to(tmp_path / "showcase/community/README.md")
    result = validate_entry(tmp_path, changed(folder, tmp_path), "2026-11-20T12:00:00Z")
    assert not result["passed"]
    assert any("symlink" in e for e in result["errors"])


def test_deadline_and_hackathon_video(tmp_path):
    assert not is_late("2026-11-28T18:59:59Z")
    assert is_late("2026-11-28T19:00:00Z")
    folder = make_entry(tmp_path, event="qwen-local-hackathon-2026")
    result = validate_entry(tmp_path, changed(folder, tmp_path), "2026-11-29T00:00:00Z")
    assert result["passed"] and result["late"]
    data = manifest("qwen-local-hackathon-2026")
    data["demo_video"] = ""
    (folder / "model.yaml").write_text(yaml.safe_dump(data))
    assert not validate_entry(tmp_path, changed(folder, tmp_path), "2026-11-20T00:00:00Z")["passed"]


def test_schema_optional_urls(tmp_path):
    folder = make_entry(tmp_path)
    data = manifest()
    data["repo_link"] = "javascript:alert(1)"
    (folder / "model.yaml").write_text(yaml.safe_dump(data))
    assert not validate_entry(tmp_path, changed(folder, tmp_path), "2026-11-20T00:00:00Z")["passed"]


def test_build_tables_and_idempotence(tmp_path):
    (tmp_path / "schema").mkdir()
    shutil.copy(ROOT / "schema/model.schema.json", tmp_path / "schema/model.schema.json")
    marker = "# Showcase\n\n<!-- SHOWCASE:START -->\n\n<!-- SHOWCASE:END -->\n"
    (tmp_path / "README.md").write_text(marker)
    (tmp_path / "showcase").mkdir()
    (tmp_path / "showcase/README.md").write_text(marker)
    for i in range(12):
        make_entry(tmp_path, slug=f"entry-{i:02}")
    build(tmp_path)
    root_text = (tmp_path / "README.md").read_text()
    assert root_text.count("[Temporary unit-test fixture]") == 10
    assert "&#124;" in root_text
    assert (tmp_path / "showcase/README.md").read_text().count("[Temporary unit-test fixture]") == 12
    assert (tmp_path / "showcase/community/README.md").read_text().count(
        "[Temporary unit-test fixture]"
    ) == 12
    build(tmp_path)
    assert (tmp_path / "README.md").read_text() == root_text


def test_compare_actual_http_and_no_reference_leakage(tmp_path):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(data)
            body = json.dumps({"choices": [{"message": {"content": "mock | response\nline"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    source = tmp_path / "eval.jsonl"
    write_jsonl(source, [chat("Held-out prompt", "SECRET REFERENCE")])
    url = f"http://127.0.0.1:{server.server_port}/v1"
    output = tmp_path / "compare.md"
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/compare.py"),
                "--eval",
                str(source),
                "--base-url",
                url,
                "--tuned-url",
                url,
                "--output",
                str(output),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert len(requests) == 2
        assert all("SECRET REFERENCE" not in json.dumps(r) for r in requests)
        assert "mock &#124; response<br>line" in output.read_text()
        assert output.with_suffix(".json").exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_tokenizer_requests_ids_not_mapping(tmp_path):
    class Tokenizer:
        def apply_chat_template(self, messages, *, tokenize, return_dict, enable_thinking):
            assert tokenize is True
            assert return_dict is False
            assert enable_thinking is False
            return list(range(73))

    path = tmp_path / "tokens.jsonl"
    write_jsonl(path, [chat()])
    errors, lengths, _ = validate(path, Tokenizer(), max_tokens=72)
    assert lengths == [73]
    assert "73 tokens exceeds 72" in errors[0]


def test_run_locally_entry_without_fine_tune(tmp_path):
    folder = make_entry(tmp_path, event="qwen-local-hackathon-2026")
    data = yaml.safe_load((ROOT / "showcase/_template/model.yaml").read_text())
    data.update(
        project_name="Local assistant fixture",
        slug="test-entry",
        event="qwen-local-hackathon-2026",
        members=[{"name": "Test", "github": "test-user"}],
        one_liner="Runs a Qwen model locally",
        base_model="unit-test/base",
        runtime="ollama",
        hardware_tested="Test fixture only",
        demo_video="https://example.org/demo",
    )
    (folder / "model.yaml").write_text(yaml.safe_dump(data))
    assert validate_entry(tmp_path, changed(folder, tmp_path), "2026-11-20T00:00:00Z")["passed"]
    data["method"] = "qlora"
    (folder / "model.yaml").write_text(yaml.safe_dump(data))
    result = validate_entry(tmp_path, changed(folder, tmp_path), "2026-11-20T00:00:00Z")
    assert not result["passed"]
