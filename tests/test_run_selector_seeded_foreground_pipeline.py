from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_selector_seeded_foreground_pipeline as pipeline  # noqa: E402


def _args(tmp_path: Path):
    return pipeline.create_parser().parse_args(
        [
            "--scale500-run-dir",
            str(tmp_path / "scale500"),
            "--selection-jsonl",
            str(tmp_path / "results.jsonl"),
            "--selection-case-input-root",
            str(tmp_path / "cases"),
            "--output-root",
            str(tmp_path / "out"),
            "--run-id",
            "selector_seeded_v1",
            "--case",
            "anon_a",
            "--overwrite",
        ]
    )


def test_export_command_uses_strict_verifier_inputs(tmp_path: Path) -> None:
    args = _args(tmp_path)

    cmd = pipeline.build_export_command(args)

    assert "export_scale500_detector_to_auto_context.py" in " ".join(cmd)
    assert cmd[cmd.index("--selection-policy") + 1] == "verifier"
    assert cmd[cmd.index("--selection-jsonl") + 1] == str(tmp_path / "results.jsonl")
    assert cmd[cmd.index("--selection-case-input-root") + 1] == str(tmp_path / "cases")
    assert cmd[cmd.index("--case") + 1] == "anon_a"
    assert "--overwrite" in cmd


def test_auto_context_command_fills_holes_but_skips_close_by_default(tmp_path: Path) -> None:
    args = _args(tmp_path)

    cmd = pipeline.build_auto_context_command(args)

    assert "--resume" in cmd
    assert "--skip-stage2" in cmd
    assert "--stage7-skip-fill-holes" not in cmd
    assert "--stage7-skip-close" in cmd
    assert cmd[cmd.index("--stage7-max-hole-size") + 1] == "1"
    assert cmd[cmd.index("--stage6-icl-k") + 1] == "0"
    assert cmd[cmd.index("--stage6-max-workers") + 1] == "16"
    assert cmd[cmd.index("--stage6-model") + 1] == "Qwen/Qwen3-VL-8B-Instruct-FP8"


def test_auto_context_command_propagates_custom_stage7_max_hole_size(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.stage7_max_hole_size = 4

    cmd = pipeline.build_auto_context_command(args)

    assert cmd[cmd.index("--stage7-max-hole-size") + 1] == "4"


def test_stage7_fill_holes_opt_in_removes_skip_flag(tmp_path: Path) -> None:
    args = pipeline.create_parser().parse_args(
        [
            "--scale500-run-dir",
            str(tmp_path / "scale500"),
            "--selection-jsonl",
            str(tmp_path / "results.jsonl"),
            "--output-root",
            str(tmp_path / "out"),
            "--run-id",
            "selector_seeded_v1",
            "--stage7-fill-holes",
        ]
    )

    cmd = pipeline.build_auto_context_command(args)

    assert "--stage7-skip-fill-holes" not in cmd


def test_stage7_skip_fill_holes_opt_out_adds_skip_flag(tmp_path: Path) -> None:
    args = pipeline.create_parser().parse_args(
        [
            "--scale500-run-dir",
            str(tmp_path / "scale500"),
            "--selection-jsonl",
            str(tmp_path / "results.jsonl"),
            "--output-root",
            str(tmp_path / "out"),
            "--run-id",
            "selector_seeded_v1",
            "--stage7-skip-fill-holes",
        ]
    )

    cmd = pipeline.build_auto_context_command(args)

    assert "--stage7-skip-fill-holes" in cmd


def test_stage7_close_opt_in_removes_skip_close_flag(tmp_path: Path) -> None:
    args = pipeline.create_parser().parse_args(
        [
            "--scale500-run-dir",
            str(tmp_path / "scale500"),
            "--selection-jsonl",
            str(tmp_path / "results.jsonl"),
            "--output-root",
            str(tmp_path / "out"),
            "--run-id",
            "selector_seeded_v1",
            "--stage7-close",
        ]
    )

    cmd = pipeline.build_auto_context_command(args)

    assert "--stage7-skip-close" not in cmd


def test_zshrc_openrouter_loader_handles_export(tmp_path: Path) -> None:
    zshrc = tmp_path / ".zshrc"
    zshrc.write_text(
        "\n".join(
            [
                "# ignored",
                "export OTHER_KEY=abc",
                "export OPENROUTER_API_KEY='sk-test value'",
            ]
        ),
        encoding="utf-8",
    )

    assert pipeline.parse_openrouter_key_from_zshrc(zshrc) == "sk-test value"


def test_write_plan_records_no_secret_value(tmp_path: Path, monkeypatch) -> None:
    args = _args(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-secret")
    env = pipeline.build_env(args)
    export_cmd = pipeline.build_export_command(args)
    auto_cmd = pipeline.build_auto_context_command(args)

    paths = pipeline.write_plan(args=args, export_cmd=export_cmd, auto_context_cmd=auto_cmd, env=env)

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    plan_text = Path(paths["json"]).read_text(encoding="utf-8")
    shell_text = Path(paths["shell"]).read_text(encoding="utf-8")
    assert payload["openrouter_key_available"] is True
    assert payload["stage7"]["max_hole_size"] == 1
    assert "sk-secret" not in plan_text
    assert "sk-secret" not in shell_text
    assert "OPENROUTER_API_KEY is missing" in shell_text
    assert "parse_openrouter_key_from_zshrc" in shell_text


def test_export_dry_run_cannot_execute_auto_context(tmp_path: Path) -> None:
    args = [
        "--scale500-run-dir",
        str(tmp_path / "scale500"),
        "--selection-jsonl",
        str(tmp_path / "results.jsonl"),
        "--output-root",
        str(tmp_path / "out"),
        "--run-id",
        "selector_seeded_v1",
        "--export-dry-run",
        "--execute-auto-context",
    ]

    old_argv = sys.argv
    try:
        sys.argv = ["run_selector_seeded_foreground_pipeline.py", *args]
        try:
            pipeline.main()
        except SystemExit as exc:
            assert "--export-dry-run cannot be combined" in str(exc)
        else:
            raise AssertionError("Expected incompatible flags to raise SystemExit")
    finally:
        sys.argv = old_argv


class _FakeResponse:
    status = 200

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_vllm_ready_requires_expected_model(monkeypatch) -> None:
    def fake_urlopen(url, timeout):
        return _FakeResponse({"data": [{"id": "Qwen/Qwen3-VL-8B-Instruct-FP8"}]})

    monkeypatch.setattr(pipeline.urllib.request, "urlopen", fake_urlopen)

    pipeline.check_vllm_ready(
        "http://127.0.0.1:8000/v1",
        expected_model="Qwen/Qwen3-VL-8B-Instruct-FP8",
    )


def test_vllm_ready_rejects_wrong_model(monkeypatch) -> None:
    def fake_urlopen(url, timeout):
        return _FakeResponse({"data": [{"id": "some-other-model"}]})

    monkeypatch.setattr(pipeline.urllib.request, "urlopen", fake_urlopen)

    try:
        pipeline.check_vllm_ready(
            "http://127.0.0.1:8000/v1",
            expected_model="Qwen/Qwen3-VL-8B-Instruct-FP8",
        )
    except RuntimeError as exc:
        assert "does not serve expected model" in str(exc)
    else:
        raise AssertionError("Expected wrong served model to fail readiness")


def test_vllm_preflight_only_needed_when_stage6_can_run(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.max_stage = 3
    assert pipeline.will_run_stage6(args) is False

    args.max_stage = 6
    assert pipeline.will_run_stage6(args) is True

    args.max_stage = None
    assert pipeline.will_run_stage6(args) is True
