import shutil
import subprocess
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.io import ensure_dir, read_json
from src.utils.logging import setup_logger


def _build_results_table(metrics: Dict[str, Dict]) -> str:
    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\caption{Retrieval metrics on demo test set.}",
        r"\label{tab:results}",
        r"\begin{tabular}{l c c c c c c}",
        r"\toprule",
        r"Method & R@1 & R@5 & R@10 & MedR & MeanR & mAP \\",
        r"\midrule",
    ]
    for baseline, vals in metrics.items():
        r1 = vals.get("R@1", 0) * 100
        r5 = vals.get("R@5", 0) * 100
        r10 = vals.get("R@10", 0) * 100
        medr = vals.get("MedR", 0)
        meanr = vals.get("MeanR", 0)
        map_ = vals.get("mAP", 0) * 100
        lines.append(f"{baseline} & {r1:.1f} & {r5:.1f} & {r10:.1f} & {medr:.1f} & {meanr:.2f} & {map_:.1f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def _build_time_table(times: Dict[str, Dict]) -> str:
    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\caption{Average per-query retrieval time (ms).}",
        r"\label{tab:time}",
        r"\begin{tabular}{l c c}",
        r"\toprule",
        r"Method & Text+Search & Search-only \\",
        r"\midrule",
    ]
    for baseline, vals in times.items():
        ts = vals.get("text_and_search_ms", 0)
        so = vals.get("search_only_ms", 0)
        lines.append(f"{baseline} & {ts:.2f} & {so:.3f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def _build_ablation_table(metrics: Dict[str, Dict]) -> str:
    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\caption{Ablation on frame count (Recall@1 \%).}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{l c}",
        r"\toprule",
        r"Frames & R@1 \\",
        r"\midrule",
    ]
    for baseline, vals in metrics.items():
        if baseline in ("random",):
            continue
        if baseline.startswith("single_frame"):
            fc = 1
        elif baseline.startswith("multi_frame_4") or baseline == "proposed":
            fc = 4
        elif baseline.startswith("multi_frame_8"):
            fc = 8
        else:
            fc = "?"
        r1 = vals.get("R@1", 0) * 100
        lines.append(f"{fc} & {r1:.1f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def _save_figs(metrics: Dict[str, Dict], figs_dir: Path) -> None:
    ensure_dir(figs_dir)
    names = []
    r1_vals = []
    for b, v in metrics.items():
        names.append(b)
        r1_vals.append(v.get("R@1", 0) * 100)
    plt.figure(figsize=(6, 4))
    plt.bar(names, r1_vals, color="steelblue")
    plt.ylabel("R@1 (%)")
    plt.xlabel("Method")
    plt.title("Recall@1 by baseline")
    plt.tight_layout()
    plt.savefig(figs_dir / "recall_at_1.png", dpi=150)
    plt.close()


def compile_latex(report_root: Path) -> Path | None:
    logger = setup_logger()
    latexmk = shutil.which("latexmk")
    pdflatex = shutil.which("pdflatex")
    if latexmk:
        cmd = [latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
    elif pdflatex:
        cmd = [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
    else:
        logger.warning("Neither latexmk nor pdflatex found. Cannot compile PDF.")
        return None
    result = subprocess.run(cmd, cwd=report_root, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("LaTeX compile error:\n%s", result.stdout + result.stderr)
        return None
    pdf_path = report_root / "main.pdf"
    if pdf_path.exists():
        return pdf_path
    return None


def build_report(cfg: Dict, run_dir: Path, report_root: Path) -> Path | None:
    logger = setup_logger()
    metrics = read_json(run_dir / "metrics.json")
    times = read_json(run_dir / "retrieval_time.json")

    tables_dir = report_root / "tables"
    ensure_dir(tables_dir)

    (tables_dir / "results_table.tex").write_text(_build_results_table(metrics))
    (tables_dir / "time_table.tex").write_text(_build_time_table(times))
    (tables_dir / "ablation_table.tex").write_text(_build_ablation_table(metrics))

    figs_dir = run_dir / "figs"
    _save_figs(metrics, figs_dir)

    pdf = compile_latex(report_root)
    if pdf is not None:
        logger.info("Compiled report PDF at %s", pdf)
    else:
        logger.info("TeX sources prepared in %s. Compile manually with: pdflatex main.tex", report_root)
    return pdf


if __name__ == "__main__":
    import argparse
    from src.utils.io import read_yaml

    parser = argparse.ArgumentParser(description="Build LaTeX report from metrics")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    cfg = read_yaml(Path(args.config))
    run_dir = Path(args.run_dir)
    report_root = Path(args.config).parent.parent / "report"
    build_report(cfg, run_dir, report_root)
