import configparser
import os
from pathlib import Path

from comfystream.capabilities.fal_key import (
    PLACEHOLDER_KEY,
    apply_fal_key,
    parse_env_file,
    resolve_and_apply_fal_key,
    resolve_fal_key,
    write_fal_key_ini,
)


def test_parse_env_file_assignment_and_raw():
    assert parse_env_file("FAL_KEY=abc-123\n") == "abc-123"
    assert parse_env_file('export FAL_KEY="quoted"\n') == "quoted"
    assert parse_env_file("# comment\njust-the-key\n") == "just-the-key"
    assert parse_env_file("OTHER=1\n") == ""


def test_resolve_prefers_explicit_over_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FAL_KEY", "from-env")
    key, source = resolve_fal_key(
        explicit="from-arg",
        workspace=tmp_path,
        extra_files=(),
    )
    assert key == "from-arg"
    assert source == "argument"


def test_resolve_skips_empty_and_placeholder(monkeypatch, tmp_path):
    monkeypatch.setenv("FAL_KEY", "")
    env_file = tmp_path / "fal.env"
    env_file.write_text(f"FAL_KEY={PLACEHOLDER_KEY}\n", encoding="utf-8")
    key, source = resolve_fal_key(
        explicit="",
        workspace=tmp_path,
        extra_files=(str(env_file),),
    )
    assert key == ""
    assert source == "unset"


def test_resolve_reads_key_file_when_env_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("FAL_KEY", "")
    env_file = tmp_path / "fal.env"
    env_file.write_text("FAL_KEY=fleet-key\n", encoding="utf-8")
    monkeypatch.setenv("FAL_KEY_FILE", str(env_file))
    key, source = resolve_fal_key(
        explicit="",
        workspace=tmp_path,
        extra_files=(),
    )
    assert key == "fleet-key"
    assert source == "FAL_KEY_FILE"


def test_resolve_reads_config_ini_as_last_resort(monkeypatch, tmp_path):
    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.delenv("FAL_KEY_FILE", raising=False)
    config = tmp_path / "custom_nodes" / "ComfyUI-fal-API" / "config.ini"
    config.parent.mkdir(parents=True)
    config.write_text("[API]\nFAL_KEY = ini-key\n", encoding="utf-8")
    key, source = resolve_fal_key(
        explicit="",
        workspace=tmp_path,
        extra_files=(),
    )
    assert key == "ini-key"
    assert Path(source) == config


def test_apply_writes_ini_and_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("FAL_KEY", raising=False)
    apply_fal_key("applied-key", workspace=tmp_path)
    assert os.environ["FAL_KEY"] == "applied-key"
    config = tmp_path / "custom_nodes" / "ComfyUI-fal-API" / "config.ini"
    raw = config.read_text(encoding="utf-8")
    assert "FAL_KEY =" in raw or "FAL_KEY=" in raw
    parser = configparser.ConfigParser()
    parser.read(config)
    assert parser.get("API", "FAL_KEY") == "applied-key"


def test_apply_empty_clears_blank_environment(monkeypatch):
    monkeypatch.setenv("FAL_KEY", "")
    apply_fal_key("")
    assert "FAL_KEY" not in os.environ


def test_resolve_and_apply_preserves_other_ini_sections(monkeypatch, tmp_path):
    monkeypatch.setenv("FAL_KEY", "")
    config = tmp_path / "custom_nodes" / "ComfyUI-fal-API" / "config.ini"
    config.parent.mkdir(parents=True)
    config.write_text(
        "[API]\n# comment\n\n[dynamic_nodes]\nenabled = false\n",
        encoding="utf-8",
    )
    env_file = tmp_path / "fal.env"
    env_file.write_text("FAL_KEY=restored\n", encoding="utf-8")
    key, source = resolve_and_apply_fal_key(
        explicit="",
        workspace=tmp_path,
        extra_files=(str(env_file),),
    )
    assert key == "restored"
    assert source == str(env_file)
    parser = configparser.ConfigParser()
    parser.read(config)
    assert parser.get("API", "FAL_KEY") == "restored"
    assert parser.get("dynamic_nodes", "enabled") == "false"


def test_write_fal_key_ini_skips_blank_workspace():
    assert write_fal_key_ini("abc", workspace="") is None
