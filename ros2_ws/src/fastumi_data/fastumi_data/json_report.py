"""写入转换清单与逐 episode 质量报告使用的 JSON 文件。"""

import json
from pathlib import Path
from typing import Any, Dict


def write_json_report(path: str, report: Dict[str, Any]) -> None:
    """使用 UTF-8 和稳定缩进写入质量报告。"""
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
