"""Unit tests for the workflow runner's local and cloud orchestration logic.

External boundaries (subprocesses, object stores, and bundle downloads) are
mocked.  These tests exercise the runner's command, environment, staging,
result, and failure contracts without requiring workflow engines or cloud
credentials.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import tools.workflow_runner.run as runner


def test_json_env_returns_default_for_invalid_or_non_object(monkeypatch):
    monkeypatch.setenv("INPUTS_JSON", "not-json")
    assert runner._load_json_env("INPUTS_JSON", {"fallback": True}) == {"fallback": True}

    monkeypatch.setenv("INPUTS_JSON", "[1, 2]")
    assert runner._load_json_env("INPUTS_JSON", {"fallback": True}) == {"fallback": True}


def test_download_local_file_and_file_uri(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("payload")
    destination = tmp_path / "nested" / "copy.txt"

    runner._download_uri_to_path(str(source), destination)
    assert destination.read_text() == "payload"

    second = tmp_path / "second.txt"
    runner._download_uri_to_path(source.as_uri(), second)
    assert second.read_text() == "payload"


def test_download_rejects_missing_and_unsupported_uris(tmp_path):
    with pytest.raises(RuntimeError, match="Local path not found"):
        runner._download_uri_to_path(str(tmp_path / "missing"), tmp_path / "out")
    with pytest.raises(RuntimeError, match="Unsupported download URI scheme"):
        runner._download_uri_to_path("https://example.test/input", tmp_path / "out")
    with pytest.raises(RuntimeError, match="Bad S3 URI"):
        runner._download_uri_to_path("s3:///missing-bucket", tmp_path / "out")


def test_download_s3_uses_bucket_key_and_destination(tmp_path, monkeypatch):
    client = MagicMock()
    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=lambda name: client))

    destination = tmp_path / "cache" / "input.tgz"
    runner._download_uri_to_path("s3://bucket/path/input.tgz", destination)

    client.download_file.assert_called_once_with(
        "bucket", "path/input.tgz", str(destination),
    )


def test_extract_tgz_and_run_command_boundaries(tmp_path, monkeypatch):
    check_call = MagicMock()
    run = MagicMock(return_value=SimpleNamespace(returncode=4))
    monkeypatch.setattr(runner.subprocess, "check_call", check_call)
    monkeypatch.setattr(runner.subprocess, "run", run)

    runner._extract_tgz(tmp_path / "bundle.tgz", tmp_path / "extract")
    assert (tmp_path / "extract").is_dir()
    check_call.assert_called_once_with([
        "tar", "-xzf", str(tmp_path / "bundle.tgz"), "-C", str(tmp_path / "extract"),
    ])
    assert runner._run_command("echo hi", tmp_path, {"A": "1"}) == 4
    assert runner._run_command(["echo", "hi"], tmp_path, {"A": "1"}) == 4
    assert run.call_args_list[0].kwargs["shell"] is True
    assert run.call_args_list[1].kwargs.get("shell") is None


def test_upload_rejects_bad_s3_and_unknown_schemes(monkeypatch):
    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=lambda name: MagicMock()))
    with pytest.raises(RuntimeError, match="Bad S3 URI"):
        runner._upload_uri("s3://bucket", b"data")
    with pytest.raises(RuntimeError, match="Unsupported RESULT_URI scheme"):
        runner._upload_uri("https://example.test/results", b"data")


def test_s3_put_file_uses_upload_file(tmp_path, monkeypatch):
    client = MagicMock()
    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=lambda name: client))
    local = tmp_path / "input.txt"
    local.write_text("input")

    runner._s3_put_file("bucket", "path/input.txt", local)
    client.upload_file.assert_called_once_with(str(local), "bucket", "path/input.txt")


def test_upload_local_and_s3_boundaries(tmp_path, monkeypatch):
    local = tmp_path / "results.json"
    runner._upload_uri(str(local), b"local")
    assert local.read_bytes() == b"local"

    client = MagicMock()
    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=lambda name: client))
    runner._upload_uri("s3://bucket/path/results.json", b"cloud", "text/plain")
    client.put_object.assert_called_once_with(
        Bucket="bucket", Key="path/results.json", Body=b"cloud", ContentType="text/plain",
    )


def test_upload_azure_requires_valid_uri_and_connection(monkeypatch):
    blob_service = MagicMock()
    azure_blob = SimpleNamespace(BlobServiceClient=blob_service)
    monkeypatch.setitem(sys.modules, "azure.storage.blob", azure_blob)
    monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", "UseDevelopmentStorage=true")

    runner._upload_uri("azureblob://account/container/path/results.json", b"data")
    blob_service.from_connection_string.assert_called_once_with("UseDevelopmentStorage=true")

    with pytest.raises(RuntimeError, match="need container/blob"):
        runner._upload_uri("azureblob://account/container-only", b"data")
    monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING")
    with pytest.raises(RuntimeError, match="Missing AZURE_STORAGE_CONNECTION_STRING"):
        runner._upload_uri("azureblob://account/container/path", b"data")


def test_command_and_nextflow_helpers():
    assert runner._normalize_result_uri("s3://bucket/run/results.json") == (
        "s3://bucket/run/results.json", "s3://bucket/run/outputs.json",
    )
    assert runner._normalize_result_uri("/tmp/results/") == (
        "/tmp/results/results.json", "/tmp/results/outputs.json",
    )
    with pytest.raises(RuntimeError, match="RESULT_URI is required"):
        runner._normalize_result_uri(" ")

    assert runner._s3_bucket_prefix_from_result_uri("s3://bucket/tes-runs/r1/results.json") == (
        "bucket", "tes-runs",
    )
    assert runner._s3_bucket_prefix_from_result_uri("file:///tmp/results.json") == ("", "")
    assert runner._is_nextflow_cmd(["nextflow", "run", "main.nf"])
    assert runner._is_nextflow_cmd("nextflow run main.nf")
    assert not runner._is_nextflow_cmd([])
    assert not runner._is_nextflow_cmd(None)
    assert not runner._is_nextflow_cmd(["snakemake", "-s", "Snakefile"])

    assert runner._force_profile(["nextflow", "run", "-profile", "local"], "awsbatch") == [
        "nextflow", "run", "-profile", "awsbatch",
    ]
    assert runner._force_profile(["nextflow", "run", "-profile"], "awsbatch")[-1] == "awsbatch"
    assert runner._force_profile(["nextflow", "run"], "awsbatch")[-2:] == ["-profile", "awsbatch"]


def test_patch_nextflow_for_aws_uses_result_uri_fallback(monkeypatch):
    monkeypatch.delenv("S3_RESULTS_BUCKET", raising=False)
    monkeypatch.delenv("S3_RESULTS_PREFIX", raising=False)
    command, extra_env = runner._patch_nextflow_for_aws(
        ["nextflow", "run", "main.nf"],
        run_id="run-1",
        results_uri="s3://bucket/custom-prefix/results.json",
    )
    assert command[-4:] == ["-profile", "awsbatch", "-work-dir", "s3://bucket/custom-prefix/run-1/nf-work"]
    assert extra_env == {"NXF_WORK": "s3://bucket/custom-prefix/run-1/nf-work"}


def test_patch_nextflow_for_aws_honors_configured_bucket_and_existing_work_dir(monkeypatch):
    monkeypatch.setenv("S3_RESULTS_BUCKET", "configured")
    monkeypatch.setenv("S3_RESULTS_PREFIX", "/prefix/")
    command, extra_env = runner._patch_nextflow_for_aws(
        ["nextflow", "run", "main.nf", "-work-dir", "local-work"],
        run_id="run-2", results_uri="s3://other/ignored/results.json",
    )
    assert command[-2:] == ["-profile", "awsbatch"]
    assert command[command.index("-work-dir") + 1] == "local-work"
    assert extra_env == {}


def test_stage_inputs_rewrites_nested_duplicate_paths(tmp_path, monkeypatch):
    source = tmp_path / "input.fastq"
    source.write_text("reads")
    (tmp_path / "exec").mkdir()
    uploaded = []
    monkeypatch.setattr(runner, "_s3_put_file", lambda bucket, key, path: uploaded.append((bucket, key, path)))

    rewritten, manifest = runner._stage_and_rewrite_inputs_to_s3(
        {"a": str(source), "nested": [str(source), "literal"]},
        bucket="bucket",
        base_prefix="runs/",
        run_id="r1",
        exec_root=tmp_path / "exec",
    )

    assert rewritten["a"] == rewritten["nested"][0]
    assert rewritten["nested"][1] == "literal"
    assert len(uploaded) == 1
    assert manifest["files"][0]["basename"] == "input.fastq"
    assert json.loads((tmp_path / "exec" / "stage_manifest.json").read_text()) == manifest


def test_stage_inputs_requires_bucket(tmp_path):
    with pytest.raises(RuntimeError, match="S3 bucket is empty"):
        runner._stage_and_rewrite_inputs_to_s3(
            {}, bucket="", base_prefix="runs", run_id="r1", exec_root=tmp_path,
        )


def test_stage_inputs_preserves_non_string_values(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_s3_put_file", MagicMock())
    rewritten, manifest = runner._stage_and_rewrite_inputs_to_s3(
        {"count": 3, "enabled": True},
        bucket="bucket", base_prefix="runs", run_id="r1", exec_root=tmp_path,
    )
    assert rewritten == {"count": 3, "enabled": True}
    assert manifest["files"] == []


def test_input_and_parameter_helpers_preserve_existing_values(tmp_path):
    local = tmp_path / "input.txt"
    local.write_text("x")
    assert runner._looks_like_local_path(str(local))
    assert not runner._looks_like_local_path(None)
    assert not runner._looks_like_local_path("s3://bucket/input")
    assert runner._is_file_path(str(local)) == local.resolve()
    assert runner._is_file_path("/does/not/exist") is None
    assert len(runner._hash_for_key(local)) == 16

    assert runner._set_nextflow_input_json_arg(["nextflow", "run"], "/new.json") == ["nextflow", "run"]
    assert runner._set_nextflow_input_json_arg(
        ["nextflow", "--input_json", "/old.json"], "/new.json"
    ) == ["nextflow", "--input_json", "/new.json"]
    assert runner._set_nextflow_input_json_arg(["nextflow", "--input_json"], "/new.json") == [
        "nextflow", "--input_json", "/new.json",
    ]
    assert runner._append_if_param_present(["cmd"], flag="--x", value=None) == ["cmd"]
    assert runner._append_if_param_present(["cmd"], flag="--x", value=" ") == ["cmd"]
    assert runner._append_if_param_present(["cmd", "--x"], flag="--x", value=3) == ["cmd", "--x", "3"]
    assert runner._append_if_param_present(["cmd"], flag="--x", value=3) == ["cmd", "--x", "3"]


def test_apply_aws_env_does_not_overwrite_existing_values():
    child_env = {"AWS_REGION": "existing"}
    runner._apply_aws_env_from_inputs(
        child_env, {"aws_queue": " queue ", "aws_region": " us-east-1 "}
    )
    assert child_env["OMNIBIOAI_AWS_BATCH_QUEUE"] == "queue"
    assert child_env["AWS_BATCH_JOB_QUEUE"] == "queue"
    assert child_env["AWS_DEFAULT_REGION"] == "us-east-1"
    assert child_env["AWS_REGION"] == "existing"


def test_main_local_success_writes_and_uploads_results(tmp_path, monkeypatch):
    uploads = []
    executed = {}

    def fake_run(cmd, cwd, env):
        executed.update(cmd=cmd, cwd=cwd, env=env)
        (cwd / "outputs.json").write_text(json.dumps({"result": "ok"}))
        return 0

    monkeypatch.setenv("RUN_ID", "r1")
    monkeypatch.setenv("RESULT_URI", str(tmp_path / "published"))
    monkeypatch.setenv("WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("INPUTS_JSON", json.dumps({"engine": "nextflow", "command": ["nextflow", "run", "main.nf"]}))
    monkeypatch.setenv("RESOURCES_JSON", json.dumps({"cpus": 2}))
    monkeypatch.setattr(runner, "_run_command", fake_run)
    monkeypatch.setattr(runner, "_upload_uri", lambda uri, data, content_type="application/json": uploads.append((uri, data, content_type)))

    assert runner.main() == 0
    assert executed["cmd"] == ["nextflow", "run", "main.nf"]
    assert executed["env"]["OMNIBIOAI_WORKFLOW_RUN_ID"] == "r1"
    assert len(uploads) == 2
    results = json.loads(uploads[-1][1])
    assert results["ok"] is True
    assert results["outputs"] == {"result": "ok"}


def test_main_failure_records_exit_code_and_upload_error(tmp_path, monkeypatch):
    uploads = []
    monkeypatch.setenv("RUN_ID", "failed")
    monkeypatch.setenv("RESULT_URI", str(tmp_path / "results.json"))
    monkeypatch.setenv("WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("INPUTS_JSON", json.dumps({"command": ["tool", "--bad"]}))
    monkeypatch.setattr(runner, "_run_command", lambda cmd, cwd, env: 7)
    monkeypatch.setattr(runner, "_upload_uri", lambda uri, data, content_type="application/json": (
        uploads.append((uri, data)) if uri.endswith("results.json") else (_ for _ in ()).throw(RuntimeError("output store down"))
    ))

    assert runner.main() == 1
    results = json.loads(uploads[-1][1])
    assert results["ok"] is False
    assert results["exit_code"] == 7
    assert results["outputs_upload_error"] == "output store down"


@pytest.mark.parametrize(
    ("engine", "workflow", "expected"),
    [
        ("nextflow", "main.nf", ["nextflow", "run", "main.nf"]),
        ("snakemake", "Snakefile", ["snakemake", "-s", "Snakefile"]),
        ("cwl", "workflow.cwl", ["cwltool", "workflow.cwl"]),
    ],
)
def test_main_selects_legacy_engine_commands(tmp_path, monkeypatch, engine, workflow, expected):
    captured = {}
    monkeypatch.setenv("RUN_ID", f"{engine}-run")
    monkeypatch.setenv("RESULT_URI", str(tmp_path / f"{engine}.json"))
    monkeypatch.setenv("WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("INPUTS_JSON", json.dumps({"engine": engine, "workflow": workflow}))
    monkeypatch.setattr(runner, "_run_command", lambda cmd, cwd, env: captured.update(cmd=cmd) or 0)
    monkeypatch.setattr(runner, "_upload_uri", lambda *args, **kwargs: None)

    assert runner.main() == 0
    assert captured["cmd"] == expected


def test_main_bundle_mode_downloads_entrypoint_and_input_json(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setenv("RUN_ID", "bundle-run")
    monkeypatch.setenv("RESULT_URI", str(tmp_path / "bundle-results.json"))
    monkeypatch.setenv("WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("INPUTS_JSON", json.dumps({
        "engine": "nextflow",
        "workflow_bundle_s3_uri": "s3://bucket/workflow.tgz",
        "workflow_entrypoint": "main.nf",
        "input_json_uri": "s3://bucket/input.json",
        "aws_queue": "queue",
        "aws_region": "region",
    }))

    def fake_download(uri, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("downloaded")

    def fake_extract(tgz_path, destination):
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "main.nf").write_text("workflow")

    monkeypatch.setattr(runner, "_download_uri_to_path", fake_download)
    monkeypatch.setattr(runner, "_extract_tgz", fake_extract)
    monkeypatch.setattr(runner, "_run_command", lambda cmd, cwd, env: captured.update(cmd=cmd) or 0)
    monkeypatch.setattr(runner, "_upload_uri", lambda *args, **kwargs: None)

    assert runner.main() == 0
    assert captured["cmd"][0:2] == ["nextflow", "run"]
    assert "--input_json" in captured["cmd"]
    assert "--aws_queue" in captured["cmd"]
    assert "--aws_region" in captured["cmd"]


def test_main_bundle_mode_rejects_missing_entrypoint(tmp_path, monkeypatch):
    monkeypatch.setenv("RUN_ID", "bad-bundle")
    monkeypatch.setenv("RESULT_URI", str(tmp_path / "results.json"))
    monkeypatch.setenv("WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("INPUTS_JSON", json.dumps({
        "engine": "nextflow", "workflow_bundle_s3_uri": "file:///bundle.tgz",
        "workflow_entrypoint": "missing.nf",
    }))
    monkeypatch.setattr(runner, "_download_uri_to_path", lambda uri, destination: destination.write_text("tgz"))
    monkeypatch.setattr(runner, "_extract_tgz", lambda tgz, destination: destination.mkdir(parents=True))

    with pytest.raises(RuntimeError, match="workflow_entrypoint not found"):
        runner.main()


def test_main_copies_local_bundle_and_uploads_normalized_outputs(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "config.json").write_text("config")
    uploads = []
    monkeypatch.setenv("RUN_ID", "local-bundle")
    monkeypatch.setenv("RESULT_URI", str(tmp_path / "results"))
    monkeypatch.setenv("WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("INPUTS_JSON", json.dumps({
        "command": ["workflow"], "local_bundle_path": str(bundle),
    }))

    def fake_run(cmd, cwd, env):
        assert (cwd / "bundle" / "config.json").read_text() == "config"
        (cwd / "outputs.json").write_text(json.dumps(["raw"]))
        (cwd / "outputs.normalized.json").write_text(json.dumps({"normalized": True}))
        return 0

    monkeypatch.setattr(runner, "_run_command", fake_run)
    monkeypatch.setattr(runner, "_upload_uri", lambda uri, data, content_type="application/json": uploads.append((uri, data)))

    assert runner.main() == 0
    assert uploads[1][0].endswith("outputs.normalized.json")
    results = json.loads(uploads[-1][1])
    assert results["outputs"] == {"_raw": ["raw"]}
    assert results["outputs_normalized_uri"].endswith("outputs.normalized.json")


def test_main_removes_existing_local_bundle_before_copy(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "new.txt").write_text("new")
    monkeypatch.setenv("RUN_ID", "replace-bundle")
    monkeypatch.setenv("RESULT_URI", str(tmp_path / "results.json"))
    monkeypatch.setenv("WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("INPUTS_JSON", json.dumps({
        "command": ["workflow"], "local_bundle_path": str(bundle),
    }))

    def fake_run(cmd, cwd, env):
        assert not (cwd / "bundle" / "old.txt").exists()
        assert (cwd / "bundle" / "new.txt").exists()
        return 0

    monkeypatch.setattr(runner, "_run_command", fake_run)
    monkeypatch.setattr(runner, "_upload_uri", lambda *args, **kwargs: None)
    exec_root = tmp_path / "work" / "workflow_runner_exec" / "replace-bundle"
    exec_root.mkdir(parents=True)
    (exec_root / "bundle").mkdir()
    (exec_root / "bundle" / "old.txt").write_text("old")

    assert runner.main() == 0


def test_main_uses_command_str_and_shlex_for_nextflow(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setenv("RUN_ID", "string-command")
    monkeypatch.setenv("RESULT_URI", str(tmp_path / "results.json"))
    monkeypatch.setenv("WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("INPUTS_JSON", json.dumps({"command_str": "nextflow run main.nf --input 'a b'"}))
    monkeypatch.setattr(runner, "_run_command", lambda cmd, cwd, env: captured.update(cmd=cmd) or 0)
    monkeypatch.setattr(runner, "_upload_uri", lambda *args, **kwargs: None)

    assert runner.main() == 0
    assert captured["cmd"] == ["nextflow", "run", "main.nf", "--input", "a b"]


def test_main_records_normalized_output_upload_failure(tmp_path, monkeypatch):
    uploads = []
    monkeypatch.setenv("RUN_ID", "normalized-failure")
    monkeypatch.setenv("RESULT_URI", str(tmp_path / "results.json"))
    monkeypatch.setenv("WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("INPUTS_JSON", json.dumps({"command": ["workflow"]}))

    def fake_run(cmd, cwd, env):
        (cwd / "outputs.normalized.json").write_text("normalized")
        return 0

    def fake_upload(uri, data, content_type="application/json"):
        if uri.endswith("outputs.normalized.json"):
            raise RuntimeError("normalized store down")
        uploads.append((uri, data))

    monkeypatch.setattr(runner, "_run_command", fake_run)
    monkeypatch.setattr(runner, "_upload_uri", fake_upload)
    assert runner.main() == 0
    results = json.loads(uploads[-1][1])
    assert results["outputs_normalized_upload_error"] == "normalized store down"


def test_main_omits_oversized_outputs_from_result(tmp_path, monkeypatch):
    uploads = []
    monkeypatch.setenv("RUN_ID", "large-output")
    monkeypatch.setenv("RESULT_URI", str(tmp_path / "results.json"))
    monkeypatch.setenv("WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("INPUTS_JSON", json.dumps({"command": ["workflow"]}))

    def fake_run(cmd, cwd, env):
        (cwd / "outputs.json").write_text(json.dumps({"data": "x" * 200_001}))
        return 0

    monkeypatch.setattr(runner, "_run_command", fake_run)
    monkeypatch.setattr(runner, "_upload_uri", lambda uri, data, content_type="application/json": uploads.append((uri, data)))

    assert runner.main() == 0
    results = json.loads(uploads[-1][1])
    assert results["outputs"] == {"note": "outputs too large; see outputs_uri"}


def test_main_catches_process_exception_and_parses_malformed_outputs(tmp_path, monkeypatch):
    uploads = []
    monkeypatch.setenv("RUN_ID", "exception")
    monkeypatch.setenv("RESULT_URI", str(tmp_path / "results.json"))
    monkeypatch.setenv("WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("INPUTS_JSON", json.dumps({"command": ["tool"]}))

    def fail_run(cmd, cwd, env):
        (cwd / "outputs.json").write_text("not-json")
        raise OSError("spawn failed")

    monkeypatch.setattr(runner, "_run_command", fail_run)
    monkeypatch.setattr(runner, "_upload_uri", lambda uri, data, content_type="application/json": uploads.append((uri, data)))

    assert runner.main() == 1
    results = json.loads(uploads[-1][1])
    assert results["error"] == "spawn failed"
    assert "outputs.json parse failed" in results["outputs"]["error"]


def test_main_aws_stages_inputs_and_patches_nextflow(tmp_path, monkeypatch):
    source = tmp_path / "reads.fastq"
    source.write_text("reads")
    uploads = []
    calls = {}

    monkeypatch.setenv("RUN_ID", "aws-1")
    monkeypatch.setenv("RESULT_URI", "s3://bucket/tes-runs/aws-1/results.json")
    monkeypatch.setenv("WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("INPUTS_JSON", json.dumps({
        "engine": "nextflow", "command": ["nextflow", "run", "main.nf"],
        "reads": str(source), "aws_queue": "queue", "aws_region": "us-east-1",
    }))
    monkeypatch.setattr(runner, "_s3_put_file", lambda bucket, key, local: calls.setdefault("staged", []).append((bucket, key, local)))

    def fake_run(cmd, cwd, env):
        calls.update(cmd=cmd, cwd=cwd, env=env)
        return 0

    monkeypatch.setattr(runner, "_run_command", fake_run)
    monkeypatch.setattr(runner, "_upload_uri", lambda uri, data, content_type="application/json": uploads.append(uri))

    assert runner.main() == 0
    assert calls["staged"]
    assert "-profile" in calls["cmd"]
    assert "awsbatch" in calls["cmd"]
    assert "-work-dir" in calls["cmd"]
    assert calls["env"]["AWS_BATCH_JOB_QUEUE"] == "queue"
    assert calls["env"]["NXF_WORK"].endswith("/aws-1/nf-work")
    assert len(uploads) == 2


def test_main_requires_run_id(monkeypatch):
    monkeypatch.delenv("RUN_ID", raising=False)
    with pytest.raises(RuntimeError, match="RUN_ID is required"):
        runner.main()
