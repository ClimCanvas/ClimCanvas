# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""GitHub の Issue フォーム (`.github/ISSUE_TEMPLATE/`) の構造検査。

GitHub はスキーマに合わないフォームを選択画面から黙って外す (公開するまで気づけない)
ので、GitHub の form schema のうち使っている範囲をここで確かめる。あわせて、
アプリの About (climcanvas/ui/about.py) が版の事前入力に使うフィールド id と、
About と同じ Google フォームの URL が選択画面に載っていることを固定する
(docs/feedback_channels_plan.md)。
"""

from __future__ import annotations

import pathlib
import re

import pytest
import yaml

from climcanvas.ui import about

TEMPLATE_DIR = pathlib.Path(__file__).resolve().parents[1] / ".github" / "ISSUE_TEMPLATE"
FORMS = ("bug.yml", "feature.yml")
FIELD_TYPES = {"input", "textarea", "dropdown", "checkboxes"}


def _load(name: str) -> dict:
    return yaml.safe_load((TEMPLATE_DIR / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", FORMS)
def test_issue_form_schema(name):
    form = _load(name)
    assert isinstance(form.get("name"), str) and form["name"]
    assert isinstance(form.get("description"), str) and form["description"]
    assert isinstance(form.get("labels"), list)
    body = form.get("body")
    assert isinstance(body, list) and body

    ids = []
    for elem in body:
        attrs = elem.get("attributes", {})
        if elem["type"] == "markdown":
            assert isinstance(attrs.get("value"), str) and attrs["value"].strip()
            assert "id" not in elem and "validations" not in elem
            continue
        assert elem["type"] in FIELD_TYPES, elem
        assert re.fullmatch(r"[A-Za-z0-9_-]+", elem["id"]), elem
        ids.append(elem["id"])
        assert isinstance(attrs.get("label"), str) and attrs["label"], elem
        if elem["type"] == "dropdown":
            opts = attrs.get("options")
            assert isinstance(opts, list) and opts, elem
            assert all(isinstance(o, str) and o for o in opts), elem
            assert len(set(opts)) == len(opts), elem
        if "validations" in elem:
            assert set(elem["validations"]) <= {"required"}, elem
    assert len(set(ids)) == len(ids), ids
    # アプリの About は選択画面に `?version=...` を渡し、選んだフォームの版の欄に入る
    assert about.ISSUE_VERSION_FIELD in ids


def test_issue_template_chooser():
    cfg = _load("config.yml")
    assert cfg["blank_issues_enabled"] is False
    links = cfg["contact_links"]
    for link in links:
        assert set(link) == {"name", "url", "about"}, link
        # GitHub のスキーマは https のみ (mailto は選択画面から外される)
        assert link["url"].startswith("https://"), link
    # 選択画面のフォームはアプリの About と同じ (版の事前入力は About 側だけ)
    urls = {link["url"] for link in links}
    assert set(about.FEEDBACK_FORMS.values()) <= urls
    # 先頭はサイトのマニュアル (About と同じサイト)
    assert links[0]["url"].startswith(about.SITE_URL), links[0]
