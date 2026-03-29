from __future__ import annotations

import openpi_thor.doctor as doctor


def test_query_trtexec_version_uses_banner_from_nonzero_version_probe(monkeypatch) -> None:
    def fake_command_output(command: list[str]) -> tuple[bool, str]:
        assert command == ["trtexec", "--version"]
        return (
            False,
            "&&&& RUNNING TensorRT.trtexec [TensorRT v101303] [b9] # trtexec --version\n"
            "[03/29/2026-18:58:09] [E] Model missing or format not recognized\n"
            "&&&& FAILED TensorRT.trtexec [TensorRT v101303] [b9] # trtexec --version",
        )

    monkeypatch.setattr(doctor, "_command_output", fake_command_output)

    ok, version = doctor._query_trtexec_version()

    assert ok is True
    assert version == "v101303"


def test_query_trtexec_version_falls_back_to_help(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_command_output(command: list[str]) -> tuple[bool, str]:
        calls.append(command)
        if command == ["trtexec", "--version"]:
            return False, "trtexec version probe failed"
        if command == ["trtexec", "--help"]:
            return (
                True,
                "&&&& RUNNING TensorRT.trtexec [TensorRT v101303] [b9] # trtexec --help\n"
                "=== Model Options ===",
            )
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(doctor, "_command_output", fake_command_output)

    ok, version = doctor._query_trtexec_version()

    assert ok is True
    assert version == "v101303"
    assert calls == [["trtexec", "--version"], ["trtexec", "--help"]]
