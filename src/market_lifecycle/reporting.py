from pathlib import Path


def write_markdown_report(report: dict, path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_markdown_report(report), encoding="utf-8")


def render_markdown_report(report: dict) -> str:
    latest = report["latest_signal"]
    full = report["full_period"]
    walk = report["walk_forward"]
    walk_summary = walk.get("summary", {})

    lines = [
        "# 大盤生命周期預測報告",
        "",
        "## 最新訊號",
        "",
        f"- 日期：{latest['date']}",
        f"- 生命周期階段：{latest['lifecycle_stage']}",
        f"- 風險狀態：{latest['risk_regime']}",
        f"- 階段分數：{latest['stage_score']:.2f}",
        f"- 未來 1 週看法：{latest['forward_1w_view']}",
        f"- 未來 4 週看法：{latest['forward_4w_view']}",
        f"- 信心分數：{latest['confidence']:.2%}",
        f"- 記憶體產業風險：{latest.get('memory_risk_level', 'low')} / {latest.get('memory_risk_points', 0)} 分",
        f"- 產業價格確認：{'是' if latest.get('memory_market_confirmed') else '否'}",
        f"- 記憶體風險原因：{latest.get('memory_risk_reasons') or '無'}",
        "",
        "## 全期間驗證",
        "",
        f"- 驗證期間：{full['start_date']} 至 {full['end_date']}",
        f"- 資料筆數：{full['rows']}",
        f"- 策略總報酬：{full['strategy_total_return']:.2%}",
        f"- 大盤總報酬：{full['market_total_return']:.2%}",
        f"- 策略最大回撤：{full['strategy_max_drawdown']:.2%}",
        f"- 大盤最大回撤：{full['market_max_drawdown']:.2%}",
        f"- 策略 Sharpe：{full['strategy_sharpe']:.2f}",
        f"- 5 日方向命中率：{full['direction_accuracy_5d']:.2%}",
        f"- Risk-off 天數：{full['risk_off_days']}",
        "",
        "## Walk-forward 驗證",
        "",
        f"- 視窗數：{walk.get('window_count', 0)}",
    ]

    if walk_summary:
        lines.extend(
            [
                f"- 平均 5 日方向命中率：{walk_summary['avg_direction_accuracy_5d']:.2%}",
                f"- 平均策略最大回撤：{walk_summary['avg_strategy_max_drawdown']:.2%}",
                f"- 最差策略最大回撤：{walk_summary['worst_strategy_max_drawdown']:.2%}",
                f"- 平均策略 Sharpe：{walk_summary['avg_strategy_sharpe']:.2f}",
                f"- 正報酬視窗數：{walk_summary['positive_windows']}",
            ]
        )

    lines.extend(
        [
            "",
            "## 解讀原則",
            "",
            "- 本報告是研究與風險控管工具，不是保證獲利訊號。",
            "- 若策略最大回撤沒有低於大盤，代表 risk-off 模組需要調整。",
            "- 若 walk-forward 表現集中在少數區間，代表模型穩健性不足。",
        ]
    )

    return "\n".join(lines) + "\n"


def write_point_validation_report(validation: dict, path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_point_validation_report(validation), encoding="utf-8")


def render_point_validation_report(validation: dict) -> str:
    result = "命中" if validation["is_hit"] else "未命中"
    return "\n".join(
        [
            "# 單日大盤走勢預測驗證",
            "",
            f"- 輸入日期：{validation['requested_date']}",
            f"- 使用訊號日期：{validation['signal_date']}",
            f"- 驗證日期：{validation['validation_date']}",
            f"- 驗證天數：{validation['horizon_days']}",
            f"- 當日收盤指數：{validation['input_close']:.2f}",
            f"- 驗證日收盤指數：{validation['future_close']:.2f}",
            f"- 實際報酬：{validation['future_return']:.2%}",
            f"- 生命周期階段：{validation['lifecycle_stage']}",
            f"- 階段分數：{validation['stage_score']:.2f}",
            f"- 風險狀態：{validation['risk_regime']}",
            f"- 模型預測方向：{validation['predicted_direction']}",
            f"- 實際方向：{validation['actual_direction']}",
            f"- 驗證結果：{result}",
            f"- 信心分數：{validation['confidence']:.2%}",
            "",
        ]
    )


def write_market_context_report(analysis: dict, path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_market_context_report(analysis), encoding="utf-8")


def render_market_context_report(analysis: dict) -> str:
    input_check = analysis["input_check"]
    if input_check["provided"]:
        consistency = "一致" if input_check["is_consistent"] else "不一致"
        input_line = (
            f"- 輸入指數：{analysis['input_index']:.2f}，資料收盤：{analysis['data_close']:.2f}，"
            f"差異：{input_check['difference']:.2f} ({input_check['difference_pct']:.2%})，檢查：{consistency}"
        )
    else:
        input_line = f"- 資料收盤指數：{analysis['data_close']:.2f}"

    result = "命中" if analysis["is_hit"] else "未命中"
    return "\n".join(
        [
            "# 指定日期大盤前後走勢分析",
            "",
            f"- 輸入日期：{analysis['requested_date']}",
            f"- 使用交易日：{analysis['signal_date']}",
            input_line,
            "",
            "## 前期走勢",
            "",
            f"- 起始日期：{analysis['lookback_start_date']}",
            f"- 起始收盤：{analysis['lookback_start_close']:.2f}",
            f"- 期間：前 {analysis['lookback_days']} 個交易日",
            f"- 前期報酬：{analysis['prior_return']:.2%}",
            f"- 前期趨勢：{analysis['prior_trend']}",
            "",
            "## 後期走勢",
            "",
            f"- 驗證日期：{analysis['forward_end_date']}",
            f"- 驗證收盤：{analysis['forward_end_close']:.2f}",
            f"- 期間：後 {analysis['forward_days']} 個交易日",
            f"- 後期報酬：{analysis['forward_return']:.2%}",
            f"- 後期趨勢：{analysis['forward_trend']}",
            f"- 轉折型態：{analysis['turning_point']}",
            "",
            "## 模型判斷",
            "",
            f"- 生命周期階段：{analysis['lifecycle_stage']}",
            f"- 階段分數：{analysis['stage_score']:.2f}",
            f"- 風險狀態：{analysis['risk_regime']}",
            f"- 預測方向：{analysis['predicted_direction']}",
            f"- 實際方向：{analysis['actual_direction']}",
            f"- 驗證結果：{result}",
            f"- 信心分數：{analysis['confidence']:.2%}",
            "",
        ]
    )


def write_success_evaluation_report(evaluation: dict, walk_forward: dict, path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_success_evaluation_report(evaluation, walk_forward), encoding="utf-8")


def render_success_evaluation_report(evaluation: dict, walk_forward: dict) -> str:
    lines = [
        "# 預測成功率總評估",
        "",
        "## 全樣本多週期成功率",
        "",
        "| 週期 | 樣本 | 整體成功率 | 可行動成功率 | 可行動覆蓋率 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for item in evaluation["horizons"]:
        lines.append(
            f"| {item['horizon_days']} 日 | {item['sample_count']} | {item['overall_hit_rate']:.2%} | "
            f"{item['actionable_hit_rate']:.2%} | {item['actionable_coverage']:.2%} |"
        )

    summary = evaluation["summary"]
    lines.extend(
        [
            "",
            "## 全樣本摘要",
            "",
            f"- 平均整體成功率：{summary['avg_overall_hit_rate']:.2%}",
            f"- 平均可行動成功率：{summary['avg_actionable_hit_rate']:.2%}",
            f"- 平均可行動覆蓋率：{summary['avg_actionable_coverage']:.2%}",
            f"- 最佳可行動週期：{_format_best(summary.get('best_actionable_horizon'), 'actionable_hit_rate')}",
            "",
            "## Walk-forward 多週期成功率",
            "",
            "| 週期 | 視窗數 | 平均整體成功率 | 平均可行動成功率 | 平均覆蓋率 | >50%視窗 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for item in walk_forward["horizons"]:
        wf_summary = item.get("summary", {})
        lines.append(
            f"| {item['horizon_days']} 日 | {item['window_count']} | "
            f"{wf_summary.get('avg_overall_hit_rate', 0):.2%} | "
            f"{wf_summary.get('avg_actionable_hit_rate', 0):.2%} | "
            f"{wf_summary.get('avg_actionable_coverage', 0):.2%} | "
            f"{wf_summary.get('positive_edge_windows', 0)} |"
        )

    wf = walk_forward["summary"]
    lines.extend(
        [
            "",
            "## Walk-forward 摘要",
            "",
            f"- 平均整體成功率：{wf['avg_walk_forward_overall_hit_rate']:.2%}",
            f"- 平均可行動成功率：{wf['avg_walk_forward_actionable_hit_rate']:.2%}",
            f"- 最佳 walk-forward 週期：{_format_best(wf.get('best_walk_forward_horizon'), 'avg_actionable_hit_rate')}",
            "",
            "## 解讀",
            "",
            "- `整體成功率` 包含中性預測，適合看模型完整輸出是否合理。",
            "- `可行動成功率` 只統計明確看漲或看跌，較接近實際預測訊號品質。",
            "- `可行動覆蓋率` 越低，代表模型越常不出手；成功率提高時也要同時檢查覆蓋率是否過低。",
            "- Walk-forward 比全樣本更嚴格，應以 Walk-forward 作為是否真的有效的主要依據。",
            "",
        ]
    )
    return "\n".join(lines)


def _format_best(best: dict | None, metric: str) -> str:
    if not best:
        return "無"
    return f"{best['horizon_days']} 日 ({best[metric]:.2%})"


def write_probability_forecast_report(forecast: dict, simulation: dict | None, path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_probability_forecast_report(forecast, simulation), encoding="utf-8")


def render_probability_forecast_report(forecast: dict, simulation: dict | None) -> str:
    lines = [
        "# 大盤機率預報",
        "",
        f"- 預報日期：{forecast['forecast_date']}",
        f"- 收盤指數：{forecast['close']:.2f}",
        f"- 生命周期階段：{forecast['lifecycle_stage']}",
        f"- 風險狀態：{forecast['risk_regime']}",
        f"- 階段分數：{forecast['stage_score']:.2f}",
        f"- 分數區間：{forecast['score_bucket']}",
        "",
        "## 未來走勢機率",
        "",
        "| 週期 | 預測 | 上漲機率 | 下跌機率 | 盤整機率 | 信心 | 相似案例 | 比對層級 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for item in forecast["forecasts"]:
        lines.append(
            f"| {item['horizon_days']} 日 | {item['predicted_direction']} | "
            f"{item['probability_up']:.2%} | {item['probability_down']:.2%} | "
            f"{item['probability_sideways']:.2%} | {item['confidence']:.2%} | "
            f"{item['case_count']} | {item['match_level']} |"
        )

    if simulation:
        lines.extend(
            [
                "",
                "## 自我模擬成功率",
                "",
                "| 週期 | 樣本 | 命中率 | 平均信心 | 高信心樣本 | 高信心命中率 |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in simulation["by_horizon"]:
            lines.append(
                f"| {item['horizon_days']} 日 | {item['sample_count']} | {item['hit_rate']:.2%} | "
                f"{item['avg_confidence']:.2%} | {item['confident_count']} | {item['confident_hit_rate']:.2%} |"
            )
        summary = simulation["summary"]
        lines.extend(
            [
                "",
                f"- 平均命中率：{summary.get('avg_hit_rate', 0):.2%}",
                f"- 平均高信心命中率：{summary.get('avg_confident_hit_rate', 0):.2%}",
                f"- 最佳週期：{_format_best(summary.get('best_horizon'), 'hit_rate')}",
            ]
        )

    lines.extend(
        [
            "",
            "## 解讀",
            "",
            "- 這是機率預報，不是保證結果。",
            "- 預報只使用預報日以前的歷史相似狀態。",
            "- 自我模擬用逐日歷史回放檢查命中率，較接近真實使用情境。",
        ]
    )
    return "\n".join(lines) + "\n"


def write_edge_research_report(edges: dict, walk_forward: dict, path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_edge_research_report(edges, walk_forward), encoding="utf-8")


def render_edge_research_report(edges: dict, walk_forward: dict) -> str:
    lines = [
        "# 高勝率情境研究",
        "",
        "## 全樣本高勝率條件",
        "",
    ]
    for horizon in edges["horizons"]:
        lines.extend(
            [
                f"### {horizon['horizon_days']} 日",
                "",
                "| 預測 | 條件 | 樣本 | 命中率 | 覆蓋率 |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for condition in horizon["top_conditions"][:10]:
            lines.append(
                f"| {condition['prediction']} | {_format_condition(condition)} | "
                f"{condition['sample_count']} | {condition['hit_rate']:.2%} | {condition['coverage']:.2%} |"
            )
        if not horizon["top_conditions"]:
            lines.append("| 無 | 無達標條件 | 0 | 0.00% | 0.00% |")
        lines.append("")

    lines.extend(["## Walk-forward 條件驗證", ""])
    lines.extend(
        [
            "| 週期 | 視窗數 | 平均命中率 | 平均樣本 | 達 60% 視窗 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for horizon in walk_forward["horizons"]:
        summary = horizon.get("summary", {})
        lines.append(
            f"| {horizon['horizon_days']} 日 | {horizon['window_count']} | "
            f"{summary.get('avg_combined_hit_rate', 0):.2%} | "
            f"{summary.get('avg_combined_sample_count', 0):.1f} | "
            f"{summary.get('positive_edge_windows', 0)} |"
        )

    lines.extend(
        [
            "",
            "## 判讀",
            "",
            "- 全樣本高勝率條件可能過度配適，必須看 Walk-forward。",
            "- 若 Walk-forward 平均低於 60%，該條件不能視為穩定可用。",
            "- 樣本數太少的條件即使命中率高，也不能直接採用。",
            "",
        ]
    )
    return "\n".join(lines)


def _format_condition(condition: dict) -> str:
    return ", ".join(
        f"{factor}={value}" for factor, value in zip(condition["factors"], condition["values"])
    )


def write_crash_research_report(analysis: dict, forecast: dict | None, path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_crash_research_report(analysis, forecast), encoding="utf-8")


def render_crash_research_report(analysis: dict, forecast: dict | None) -> str:
    lines = [
        "# 大跌後走勢研究",
        "",
        "## 歷史統計",
        "",
    ]
    for scenario in analysis["scenarios"]:
        lines.extend(
            [
                f"### {scenario['scenario']}",
                "",
                f"- 條件：{scenario['condition']}",
                f"- 事件數：{scenario['event_count']}",
                f"- 首次：{scenario['first_event']}",
                f"- 最近：{scenario['last_event']}",
                "",
                "| 週期 | 樣本 | 最可能方向 | 上漲機率 | 下跌機率 | 盤整機率 | 平均報酬 | 中位報酬 |",
                "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for horizon in scenario["horizons"]:
            lines.append(
                f"| {horizon['horizon_days']} 日 | {horizon['sample_count']} | {horizon['best_direction']} | "
                f"{horizon['probability_up']:.2%} | {horizon['probability_down']:.2%} | "
                f"{horizon['probability_sideways']:.2%} | {horizon['avg_return']:.2%} | {horizon['median_return']:.2%} |"
            )
        lines.append("")

    if forecast:
        lines.extend(
            [
                "## 指定日大跌情境",
                "",
                f"- 日期：{forecast['target_date']}",
                f"- 收盤：{forecast['close']:.2f}",
                "",
            ]
        )
        if not forecast["active_crash_scenarios"]:
            lines.append("- 指定日未觸發大跌條件。")
        for scenario in forecast["active_crash_scenarios"]:
            lines.extend(
                [
                    f"### {scenario['scenario']}",
                    "",
                    f"- 條件：{scenario['condition']}",
                    f"- 歷史案例：{scenario['historical_cases']}",
                    "",
                    "| 週期 | 樣本 | 最可能方向 | 上漲機率 | 下跌機率 | 盤整機率 |",
                    "| --- | ---: | --- | ---: | ---: | ---: |",
                ]
            )
            for horizon in scenario["horizons"]:
                lines.append(
                    f"| {horizon['horizon_days']} 日 | {horizon['sample_count']} | {horizon['best_direction']} | "
                    f"{horizon['probability_up']:.2%} | {horizon['probability_down']:.2%} | {horizon['probability_sideways']:.2%} |"
                )
            lines.append("")

    lines.extend(
        [
            "## 判讀",
            "",
            "- 大跌後不一定續跌，需分辨急跌、波段跌、深回撤。",
            "- 若樣本數不足，機率只能當研究線索。",
            "- 若要達 90% 命中率，通常需要非常稀有且明確的條件，覆蓋率會很低。",
        ]
    )
    return "\n".join(lines)


def write_scenario_research_report(analysis: dict, walk_forward: dict, path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_scenario_research_report(analysis, walk_forward), encoding="utf-8")


def render_scenario_research_report(analysis: dict, walk_forward: dict) -> str:
    lines = [
        "# 通用市場情境研究",
        "",
        f"- 目標命中率：{analysis['target_hit_rate']:.2%}",
        f"- 最少樣本數：{analysis['min_samples']}",
        f"- 掃描情境數：{analysis['scenario_count']}",
        f"- 達標情境數：{analysis['passing_count']}",
        "",
        "## 最佳情境",
        "",
        "| 情境 | 說明 | 最佳週期 | 預測 | 樣本 | 命中率 | 平均報酬 | 中位報酬 | 是否達標 |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for scenario in analysis["scenarios"][:25]:
        best = scenario["best_horizon"] or {}
        lines.append(
            f"| {scenario['scenario']} | {scenario['description']} | {best.get('horizon_days', 0)} | "
            f"{best.get('best_direction', 'none')} | {best.get('sample_count', 0)} | "
            f"{best.get('best_probability', 0):.2%} | {best.get('avg_return', 0):.2%} | "
            f"{best.get('median_return', 0):.2%} | {'是' if scenario['passes_target'] else '否'} |"
        )

    lines.extend(
        [
            "",
            "## 達標情境明細",
            "",
        ]
    )
    passing = [item for item in analysis["scenarios"] if item["passes_target"]]
    if not passing:
        lines.append("- 無情境同時達到目標命中率與樣本數門檻。")
    for scenario in passing:
        lines.extend(
            [
                f"### {scenario['scenario']}",
                "",
                f"- 說明：{scenario['description']}",
                f"- 事件數：{scenario['event_count']}",
                f"- 首次：{scenario['first_event']}",
                f"- 最近：{scenario['last_event']}",
                "",
                "| 週期 | 預測 | 樣本 | 上漲 | 下跌 | 盤整 | 最佳命中率 | 平均報酬 |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for horizon in scenario["horizons"]:
            lines.append(
                f"| {horizon['horizon_days']} 日 | {horizon['best_direction']} | {horizon['sample_count']} | "
                f"{horizon['probability_up']:.2%} | {horizon['probability_down']:.2%} | "
                f"{horizon['probability_sideways']:.2%} | {horizon['best_probability']:.2%} | "
                f"{horizon['avg_return']:.2%} |"
            )
        lines.append("")

    wf_summary = walk_forward.get("summary", {})
    lines.extend(
        [
            "## Walk-forward 驗證",
            "",
            f"- 視窗數：{walk_forward.get('window_count', 0)}",
            f"- 平均命中率：{wf_summary.get('avg_hit_rate', 0):.2%}",
            f"- 平均樣本數：{wf_summary.get('avg_sample_count', 0):.1f}",
            f"- 含 90% 規則的視窗數：{wf_summary.get('windows_with_90pct_rule', 0)}",
            "",
            "## 結論規則",
            "",
            "- 若全樣本達標但 Walk-forward 不達標，只能視為研究線索。",
            "- 若樣本數太少或覆蓋率太低，不適合當主要預報模型。",
            "- 90% 命中率通常只可能出現在非常窄的情境，必須接受低覆蓋率。",
        ]
    )
    return "\n".join(lines)


def write_causal_database_report(summary: dict, path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_causal_database_report(summary), encoding="utf-8")


def render_causal_database_report(summary: dict) -> str:
    lines = [
        "# 因果情境資料庫摘要",
        "",
        f"- 最少樣本數：{summary['min_samples']}",
        f"- 最低命中率門檻：{summary['min_hit_rate']:.2%}",
        "",
    ]
    for horizon in summary["horizons"]:
        lines.extend(
            [
                f"## 後 {horizon['horizon_days']} 日",
                "",
                "| 因果簽名 | 預測 | 樣本 | 命中率 | 上漲 | 下跌 | 盤整 | 平均後續報酬 |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        if not horizon["edges"]:
            lines.append("| 無 | 無達標情境 | 0 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |")
        for edge in horizon["edges"][:20]:
            lines.append(
                f"| {edge['causal_signature']} | {edge['best_direction']} | {edge['sample_count']} | "
                f"{edge['hit_rate']:.2%} | {edge['probability_up']:.2%} | {edge['probability_down']:.2%} | "
                f"{edge['probability_sideways']:.2%} | {edge['avg_effect_return']:.2%} |"
            )
        lines.append("")
    lines.extend(
        [
            "## 判讀",
            "",
            "- 每一列是一種前因 + 當下狀態組合。",
            "- 只有樣本數足夠且命中率超過門檻者才列入。",
            "- 若無達標情境，代表該週期暫時不應輸出預測。",
        ]
    )
    return "\n".join(lines)


def write_causal_edge_refinement_report(refinement: dict, path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_causal_edge_refinement_report(refinement), encoding="utf-8")


def render_causal_edge_refinement_report(refinement: dict) -> str:
    baseline = refinement["baseline"]
    lines = [
        "# 因果線索失敗樣本分析",
        "",
        f"- 因果簽名：{refinement['signature']}",
        f"- 預測週期：後 {refinement['horizon_days']} 日",
        f"- 目標方向：{refinement['target_direction']}",
        f"- 是否達標：{'是' if refinement['passes_target'] else '否'}",
        "",
        "## 原始表現",
        "",
        f"- 樣本數：{baseline['sample_count']}",
        f"- 成功數：{baseline['success_count']}",
        f"- 失敗數：{baseline['fail_count']}",
        f"- 命中率：{baseline['hit_rate']:.2%}",
        f"- 平均報酬：{baseline['avg_return']:.2%}",
        f"- 中位報酬：{baseline['median_return']:.2%}",
        "",
        "## 成功組 vs 失敗組差異",
        "",
        "| 因子 | 成功組平均 | 失敗組平均 | 差異 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in refinement["success_failure_comparison"][:12]:
        lines.append(
            f"| {row['factor']} | {row['success_mean']:.4f} | {row['failure_mean']:.4f} | {row['difference']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 排除條件測試",
            "",
            "| 排除條件 | 保留樣本 | 移除樣本 | 移除樣本失敗率 | 命中率 | 平均報酬 | 是否達標 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in refinement["candidate_filters"][:20]:
        lines.append(
            f"| {row['filter']} | {row['sample_count']} | {row['removed_count']} | "
            f"{row['removed_fail_rate']:.2%} | {row['hit_rate']:.2%} | {row['avg_return']:.2%} | "
            f"{'是' if row['passes_target'] else '否'} |"
        )

    if refinement["best_filter"]:
        best = refinement["best_filter"]
        lines.extend(
            [
                "",
                "## 最佳排除條件",
                "",
                f"- {best['filter']}",
                f"- 保留樣本：{best['sample_count']}",
                f"- 命中率：{best['hit_rate']:.2%}",
                f"- 平均報酬：{best['avg_return']:.2%}",
            ]
        )
    return "\n".join(lines)


def write_failure_analysis_report(
    before: dict,
    after: dict,
    rules: list[dict],
    path: str,
    steps: list[dict] | None = None,
    walk_forward: dict | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        render_failure_analysis_report(before, after, rules, steps or [], walk_forward or {}),
        encoding="utf-8",
    )


def render_failure_analysis_report(
    before: dict,
    after: dict,
    rules: list[dict],
    steps: list[dict],
    walk_forward: dict,
) -> str:
    lines = [
        "# 失敗因子分析報告",
        "",
        "## 排除前",
        "",
        f"- 驗證天數：{before['horizon_days']}",
        f"- 可行動樣本數：{before['sample_count']}",
        f"- 可行動成功率：{before['overall_hit_rate']:.2%}",
        "",
        "## 主要失敗因子",
        "",
    ]

    for factor in before["worst_factors"][:10]:
        lines.append(
            f"- {factor['factor']} = {factor['value']}：成功率 {factor['hit_rate']:.2%}，樣本 {factor['sample_count']}"
        )

    lines.extend(["", "## 已排除規則", ""])
    if rules:
        for rule in rules:
            lines.append(
                f"- {rule['factor']} = {rule['value']}：訓練成功率 {rule['hit_rate']:.2%}，樣本 {rule['sample_count']}"
            )
    else:
        lines.append("- 無符合排除條件的失敗因子。")

    if steps:
        lines.extend(["", "## 逐一排除後變化", ""])
        for index, step in enumerate(steps, start=1):
            rule = step["rule"]
            lines.append(
                f"- 第 {index} 次：排除 {rule['factor']} = {rule['value']}，可行動成功率 {step['actionable_hit_rate']:.2%}，可行動樣本 {step['actionable_sample_count']}"
            )

    lines.extend(
        [
            "",
            "## 排除後",
            "",
            f"- 可行動樣本數：{after['sample_count']}",
            f"- 可行動成功率：{after['overall_hit_rate']:.2%}",
            "",
        ]
    )
    if walk_forward:
        summary = walk_forward.get("summary", {})
        lines.extend(
            [
                "## Walk-forward 排除驗證",
                "",
                f"- 視窗數：{walk_forward.get('window_count', 0)}",
                f"- 排除前平均可行動成功率：{summary.get('avg_before_actionable_hit_rate', 0):.2%}",
                f"- 排除後平均可行動成功率：{summary.get('avg_after_actionable_hit_rate', 0):.2%}",
                f"- 排除前平均可行動樣本：{summary.get('avg_before_actionable_count', 0):.1f}",
                f"- 排除後平均可行動樣本：{summary.get('avg_after_actionable_count', 0):.1f}",
                f"- 改善視窗數：{summary.get('improved_windows', 0)}",
                "",
            ]
        )
    return "\n".join(lines)
