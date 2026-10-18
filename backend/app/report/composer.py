"""把画像、指标、图表和模型结论整合成一份结构化分析报告。

章节骨架是固定的（保证结构稳定、可预期），叙述内容交给大模型，
表格与图表由程序渲染（保证数字与图片准确无误）。
"""

from __future__ import annotations

import html as html_lib
import re
from datetime import datetime
from typing import Any

from ..dataset.profiler import to_jsonable

MAX_TABLE_COLUMNS = 40


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_（无数据）_"
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        cells = []
        for v in row:
            text = "-" if v is None else str(v)
            cells.append(text.replace("|", "/").replace("\n", " "))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _inline_md_to_html(text: str) -> str:
    """极简 Markdown 行内渲染（加粗、代码、图片），足够报告使用。"""

    text = html_lib.escape(text, quote=False)
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r'<img alt="\1" src="\2">', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return text


def _block_to_html(block: str) -> str:
    """把一小段 Markdown 渲染成 HTML（支持标题/列表/表格/图片/段落）。"""

    lines = block.splitlines()
    out: list[str] = []
    in_list = False
    in_table = False

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    def close_table() -> None:
        nonlocal in_table
        if in_table:
            out.append("</tbody></table>")
            in_table = False

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            close_list()
            close_table()
            continue
        if line.lstrip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            if not in_table:
                close_list()
                out.append("<table><thead><tr>" + "".join(f"<th>{_inline_md_to_html(c)}</th>" for c in cells) + "</tr></thead><tbody>")
                in_table = True
            else:
                out.append("<tr>" + "".join(f"<td>{_inline_md_to_html(c)}</td>" for c in cells) + "</tr>")
            continue
        close_table()
        if line.lstrip().startswith(("- ", "* ")):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{_inline_md_to_html(line.lstrip()[2:])}</li>")
            continue
        close_list()
        stripped = line.strip()
        if stripped.startswith("###"):
            out.append(f"<h4>{_inline_md_to_html(stripped.lstrip('#').strip())}</h4>")
        elif stripped.startswith("##"):
            out.append(f"<h3>{_inline_md_to_html(stripped.lstrip('#').strip())}</h3>")
        elif stripped.startswith("#"):
            out.append(f"<h2>{_inline_md_to_html(stripped.lstrip('#').strip())}</h2>")
        else:
            out.append(f"<p>{_inline_md_to_html(stripped)}</p>")
    close_list()
    close_table()
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# 组装
# --------------------------------------------------------------------------- #
def build_report(
    *,
    dataset: dict[str, Any],
    profile: dict[str, Any],
    narrative: dict[str, Any],
    evidence: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    """生成报告 payload（含 Markdown 与 HTML 两种呈现）。"""

    name = str(dataset.get("name") or "数据集")
    quality = profile.get("quality", {}) if profile else {}
    columns = profile.get("columns", []) if profile else []

    column_rows = [
        [
            c.get("name"),
            c.get("semantic_type"),
            c.get("missing"),
            f"{float(c.get('missing_ratio') or 0):.1%}",
            c.get("unique"),
            "、".join(str(v) for v in (c.get("sample_values") or [])[:3]),
        ]
        for c in columns[:MAX_TABLE_COLUMNS]
    ]
    column_table = _md_table(["字段", "类型", "缺失数", "缺失率", "唯一值", "示例"], column_rows)

    issue_rows = [
        [i.get("level"), i.get("scope"), i.get("message"), i.get("suggestion")]
        for i in quality.get("issues", [])
    ]
    issue_table = _md_table(["严重度", "涉及范围", "问题", "建议处理方式"], issue_rows)

    findings = [str(x).strip() for x in (narrative.get("findings") or []) if str(x).strip()]
    recommendations = [str(x).strip() for x in (narrative.get("recommendations") or []) if str(x).strip()]

    evidence_rows = [
        [e.get("tool"), (e.get("summary") or "")[:120]] for e in evidence if e.get("summary")
    ]
    evidence_table = _md_table(["工具", "结论摘要"], evidence_rows)

    payload: dict[str, Any] = {
        "title": str(narrative.get("title") or f"{name} 数据分析报告"),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {
            "name": name,
            "filename": dataset.get("filename"),
            "sheet": dataset.get("sheet"),
            "rows": profile.get("rows", dataset.get("rows")),
            "cols": profile.get("cols", dataset.get("cols")),
            "revision": dataset.get("revision"),
            "quality_score": quality.get("score"),
        },
        "column_groups": profile.get("column_groups", {}) if profile else {},
        "sections": [
            {
                "id": "overview",
                "heading": "一、数据概览",
                "body_md": str(narrative.get("overview") or "").strip(),
                "extra_md": f"**字段清单**\n\n{column_table}",
            },
            {
                "id": "quality",
                "heading": "二、数据质量体检",
                "body_md": str(narrative.get("quality") or "").strip(),
                "extra_md": (
                    f"**质量评分：{quality.get('score', '-')} / 100**"
                    f"（重复行 {quality.get('duplicate_rows', 0)}，缺失单元格 {quality.get('total_missing', 0)}）\n\n"
                    f"{issue_table}"
                ),
            },
            {
                "id": "findings",
                "heading": "三、关键发现",
                "items": findings,
                "extra_md": f"**分析过程留痕（工具调用）**\n\n{evidence_table}" if evidence_rows else "",
            },
            {
                "id": "charts",
                "heading": "四、图表证据",
                "charts": artifacts,
                "extra_md": "",
            },
            {
                "id": "conclusion",
                "heading": "五、结论与建议",
                "items": recommendations,
                "extra_md": "",
            },
        ],
        "stats": {
            "evidence_count": len(evidence),
            "chart_count": len(artifacts),
            "column_count": len(columns),
        },
    }
    payload["markdown"] = render_markdown(payload)
    payload["html"] = render_html(payload)
    return to_jsonable(payload)


def render_markdown(payload: dict[str, Any]) -> str:
    ds = payload["dataset"]
    lines = [
        f"# {payload['title']}",
        "",
        f"> 数据集：**{ds['name']}** ｜ 规模：{ds['rows']} 行 × {ds['cols']} 列"
        f" ｜ 质量评分：{ds.get('quality_score', '-')}/100 ｜ 生成时间：{payload['generated_at']}",
        "",
    ]
    for section in payload["sections"]:
        lines.append(f"## {section['heading']}")
        lines.append("")
        body = section.get("body_md")
        if body:
            lines.append(body)
            lines.append("")
        items = section.get("items")
        if items:
            lines.extend(f"{i + 1}. {item}" for i, item in enumerate(items))
            lines.append("")
        charts = section.get("charts")
        if charts:
            for chart in charts:
                lines.append(f"### {chart.get('title')}")
                lines.append("")
                lines.append(f"![{chart.get('title')}]({chart.get('url')})")
                lines.append("")
                if chart.get("description"):
                    lines.append(f"_{chart['description']}_")
                    lines.append("")
        extra = section.get("extra_md")
        if extra:
            lines.append(extra)
            lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("_本报告由数据分析 Agent 自动生成：数据画像 → 工具调用（统计/清洗/可视化）→ 结论汇总。_")
    return "\n".join(lines)


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ font-family: "Microsoft YaHei", "PingFang SC", "Helvetica Neue", Arial, sans-serif;
         max-width: 980px; margin: 40px auto; padding: 0 24px 80px; color: #1f2328; line-height: 1.75; }}
  h1 {{ font-size: 28px; border-bottom: 3px solid #4C72B0; padding-bottom: 12px; }}
  h2 {{ font-size: 21px; margin-top: 38px; color: #24447a; border-left: 5px solid #4C72B0; padding-left: 10px; }}
  h3, h4 {{ font-size: 16px; margin-top: 22px; }}
  .meta {{ background: #f2f5fb; border-radius: 10px; padding: 14px 18px; margin: 18px 0 8px;
           font-size: 14px; color: #38445c; }}
  .meta b {{ color: #24447a; }}
  table {{ border-collapse: collapse; width: 100%; margin: 14px 0; font-size: 13.5px; }}
  th, td {{ border: 1px solid #dbe1ec; padding: 7px 10px; text-align: left; vertical-align: top; }}
  th {{ background: #eef2fa; font-weight: 600; }}
  tr:nth-child(even) td {{ background: #fafbfe; }}
  img {{ max-width: 100%; border: 1px solid #e3e8f0; border-radius: 8px; margin: 10px 0; }}
  ol, ul {{ padding-left: 24px; }}
  li {{ margin: 6px 0; }}
  code {{ background: #f2f4f8; padding: 1px 5px; border-radius: 4px; font-size: 13px; }}
  footer {{ margin-top: 46px; color: #7a869a; font-size: 12.5px; text-align: center; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="meta">{meta}</div>
{body}
<footer>本报告由数据分析 Agent 自动生成 ｜ {generated_at}</footer>
</body>
</html>
"""


def render_html(payload: dict[str, Any]) -> str:
    ds = payload["dataset"]
    meta = (
        f"数据集：<b>{html_lib.escape(str(ds['name']))}</b> ｜ "
        f"规模：<b>{ds['rows']}</b> 行 × <b>{ds['cols']}</b> 列 ｜ "
        f"质量评分：<b>{ds.get('quality_score', '-')}</b>/100 ｜ "
        f"生成时间：{payload['generated_at']}"
    )
    blocks: list[str] = []
    for section in payload["sections"]:
        blocks.append(f"<h2>{html_lib.escape(section['heading'])}</h2>")
        if section.get("body_md"):
            blocks.append(_block_to_html(section["body_md"]))
        if section.get("items"):
            items = "".join(f"<li>{_inline_md_to_html(i)}</li>" for i in section["items"])
            blocks.append(f"<ol>{items}</ol>")
        for chart in section.get("charts") or []:
            title = html_lib.escape(str(chart.get("title") or "图表"))
            blocks.append(f"<h3>{title}</h3>")
            blocks.append(
                f'<img alt="{title}" src="{chart.get("url")}">'
            )
            if chart.get("description"):
                blocks.append(f"<p><em>{html_lib.escape(str(chart['description']))}</em></p>")
        if section.get("extra_md"):
            blocks.append(_block_to_html(section["extra_md"]))

    return _HTML_TEMPLATE.format(
        title=html_lib.escape(str(payload["title"])),
        meta=meta,
        body="\n".join(blocks),
        generated_at=payload["generated_at"],
    )
