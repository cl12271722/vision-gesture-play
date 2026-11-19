from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

from .logic import AttemptRecord, Direction, RoundOutcome


@dataclass(frozen=True)
class ReportPaths:
    csv_path: Path
    summary_path: Path


def build_advice(
    records: Sequence[AttemptRecord],
    accuracy: float,
) -> list[str]:
    """根据本轮表现生成非医学性的用眼建议。"""

    advice: list[str] = []
    if accuracy >= 0.9:
        advice.append("方向判断很稳定，可以继续保持由远到近、由慢到快的练习。")
    elif accuracy >= 0.75:
        advice.append("整体表现良好，建议重点复习容易混淆的左右方向。")
    else:
        advice.append("建议先保证环境光线充足，再放慢节奏完成下一轮测试。")

    direction_mistakes: dict[Direction, int] = defaultdict(int)
    for record in records:
        if record.outcome is not RoundOutcome.CORRECT:
            direction_mistakes[record.direction] += 1

    if direction_mistakes:
        hardest = max(direction_mistakes, key=direction_mistakes.get)
        advice.append(
            f"本轮“{hardest.label}”方向出错最多，下一轮可重点观察该方向。"
        )

    answered = [
        record.response_time_ms
        for record in records
        if record.outcome is not RoundOutcome.MISS
    ]
    if answered and sum(answered) / len(answered) > 850:
        advice.append("作答反应稍慢，可以提前将食指放在舒适位置准备。")

    advice.append("建议遵循 20-20-20 用眼习惯：每用眼 20 分钟，看 6 米外约 20 秒。")
    advice.append("本测试用于趣味训练和课程演示，不能代替专业视力检查。")
    return advice


def save_report(
    records: Sequence[AttemptRecord],
    score: int,
    accuracy: float,
    output_dir: Path,
) -> ReportPaths:
    """保存逐题 CSV 日志和可读的总结报告。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"session_{timestamp}.csv"
    summary_path = output_dir / f"session_{timestamp}_summary.txt"
    advice = build_advice(records, accuracy)

    with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "题号",
                "E字方向",
                "字号",
                "用户作答",
                "判定结果",
                "反应时间毫秒",
                "累计得分",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.question_number,
                    record.direction.label,
                    record.font_size,
                    (
                        record.user_answer.label
                        if record.user_answer is not None
                        else "未作答"
                    ),
                    record.outcome.value,
                    record.response_time_ms,
                    record.score_after,
                ]
            )

    correct_count = sum(
        record.outcome is RoundOutcome.CORRECT for record in records
    )
    wrong_count = sum(record.outcome is RoundOutcome.WRONG for record in records)
    miss_count = sum(record.outcome is RoundOutcome.MISS for record in records)

    summary_lines = [
        "指尖识 E 手势视力训练报告",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"完成题目：{len(records)}",
        f"正确题数：{correct_count}",
        f"错误题数：{wrong_count}",
        f"未作答题数：{miss_count}",
        f"正确率：{accuracy:.1%}",
        f"总得分：{score}",
        "",
        "用眼建议：",
        *[f"- {item}" for item in advice],
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    return ReportPaths(csv_path=csv_path, summary_path=summary_path)
