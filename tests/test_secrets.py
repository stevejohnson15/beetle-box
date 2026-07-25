# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
import pytest

from beetlebox import secrets


def test_env_var_takes_precedence(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")
    assert secrets.load_anthropic_key() == "sk-ant-from-env"


def test_parse_bare_key():
    assert secrets._parse_key_file("sk-ant-bare\n") == "sk-ant-bare"


def test_parse_env_style_key():
    assert secrets._parse_key_file("ANTHROPIC_API_KEY=sk-ant-eq") == "sk-ant-eq"


def test_parse_colon_style_and_quotes():
    assert secrets._parse_key_file('key: "sk-ant-colon"') == "sk-ant-colon"


def test_parse_skips_comments_and_blanks():
    assert secrets._parse_key_file("# comment\n\nsk-ant-x\n") == "sk-ant-x"


def test_load_from_file(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    (tmp_path / "anthropic.key").write_text("sk-ant-file\n")
    assert secrets.load_anthropic_key(start=tmp_path) == "sk-ant-file"


def test_missing_key_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        secrets.load_anthropic_key(start=tmp_path)
