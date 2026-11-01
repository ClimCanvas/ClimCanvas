# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""公開デモの休止対策 (`.github/workflows/keep-demo-awake.yml` + `demo/keepalive.py`) の構造検査。

GitHub Actions は壊れた workflow を黙って無視する (push するまで気づけない) ので、YAML が読めること・
予定の間隔が Community Cloud の休止 (12 時間無訪問) に対して余裕があること・手動実行できること・
スクリプトを呼んでいることを固定する。スクリプト側は Playwright 無しで import できること
(定数の検査と、Actions 以外の環境での pytest のため) と、訪問先が README の公開デモの URL と
一致することを見る。経緯は docs/public_demo_plan.md「休止対策」。
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import subprocess
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "keep-demo-awake.yml"
SCRIPT = ROOT / "demo" / "keepalive.py"
# 休止の閾値 (12 時間) に対する余裕。GitHub の予定実行は混雑時に 1 時間以上遅れる
MAX_INTERVAL_HOURS = 6


def _workflow() -> dict:
    wf = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    # PyYAML (YAML 1.1) は `on:` を真偽値 True に読む
    wf["on"] = wf.get("on", wf.get(True))
    return wf


def _keepalive():
    spec = importlib.util.spec_from_file_location("climcanvas_demo_keepalive", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_workflow_schedule_leaves_margin_below_sleep_threshold():
    wf = _workflow()
    triggers = wf["on"]
    assert "workflow_dispatch" in triggers, "手動実行 (Run workflow) で試せること"
    schedule = triggers["schedule"]
    assert len(schedule) == 1
    minute, hour, dom, month, dow = schedule[0]["cron"].split()
    assert (dom, month, dow) == ("*", "*", "*")
    assert minute.isdigit() and minute != "0", "混雑する 0 分は避ける"
    m = re.fullmatch(r"\*/(\d+)", hour)
    assert m, f"時の欄は */N の形 (実際: {hour})"
    assert 1 <= int(m.group(1)) <= MAX_INTERVAL_HOURS
    assert 24 % int(m.group(1)) == 0, "24 を割り切らないと日付の境で間隔が伸びる"


def test_workflow_runs_the_keepalive_script_with_least_privilege():
    wf = _workflow()
    assert wf["permissions"] == {"contents": "read"}
    job, = wf["jobs"].values()
    assert job["timeout-minutes"] <= 30
    runs = [s.get("run", "") for s in job["steps"]]
    assert any("demo/keepalive.py" in r and "--state" in r for r in runs), "cookie の引き継ぎ付きで呼ぶ"
    uses = [s["uses"].split("@")[0] for s in job["steps"] if "uses" in s]
    assert all(u.startswith("actions/") for u in uses), f"GitHub 製の action だけを使う: {uses}"
    assert "actions/cache" in uses, "ブラウザの状態 (cookie) を run 間で引き継ぐ"


def test_keepalive_is_importable_without_playwright_and_points_at_the_demo():
    mod = _keepalive()  # Playwright は visit() の中で import するので、無い環境でも読める
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert mod.DEMO_URL in readme, "公開 README が案内する公開デモの URL と同じ"
    assert mod.DEMO_URL.endswith("/")
    assert mod.HEALTH_PATH.startswith(mod.APP_FRAME_PATH) and mod.HEALTH_PATH.endswith("_stcore/health")
    assert mod.WAKE_BUTTON_LABEL.search("Yes, get this app back up!")
    for selector in (mod.WAKE_BUTTON, mod.APP_READY):
        assert re.fullmatch(r'\[data-testid="[A-Za-z-]+"\]', selector)


def test_keepalive_help_runs_without_playwright():
    proc = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True,
                          cwd=ROOT, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert "--wake-timeout" in proc.stdout and "--state" in proc.stdout
