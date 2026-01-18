import argparse
import datetime
import shutil
from pathlib import Path

from src.data.demo_generate import generate_demo
from src.data.preprocess import preprocess_videos
from src.encode_videos import encode_videos_for_baseline
from src.evaluate import evaluate_all, save_results_csv
from src.seed import set_global_seed
from src.utils.io import ensure_dir, read_yaml, write_json
from src.utils.logging import setup_logger


def build_run_id(name: str, override: str | None) -> str:
    if override:
        return override
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{name}_{now}"


def main():
    parser = argparse.ArgumentParser(description="One-command text-to-video retrieval pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    project_root = Path(__file__).parent.resolve()
    cfg = read_yaml(config_path)
    cfg["project_root"] = str(project_root)

    run_cfg = cfg.get("run", {})
    run_id = build_run_id(run_cfg.get("name", "run"), run_cfg.get("run_id"))
    run_dir = project_root / run_cfg.get("output_dir", "outputs/runs") / run_id
    ensure_dir(run_dir)
    ensure_dir(run_dir / "tables")
    ensure_dir(run_dir / "figs")

    logger = setup_logger(log_file=run_dir / "pipeline.log")
    logger.info("Starting pipeline with config %s", config_path)
    logger.info("Run directory: %s", run_dir)

    set_global_seed(cfg.get("seed"))

    # 1) Demo dataset generation (only if demo or missing)
    dataset_name = cfg.get("dataset", {}).get("name", "demo")
    dataset_root = project_root / cfg.get("dataset", {}).get("root", "data/demo")
    if dataset_name == "demo" or not dataset_root.exists():
        generate_demo(cfg, project_root)

    # 2) Preprocess frames
    preprocess_videos(cfg, project_root)

    # 3) Encode videos for each non-random baseline
    for baseline_cfg in cfg.get("eval", {}).get("baselines", []):
        if baseline_cfg.get("name") == "random":
            continue
        encode_videos_for_baseline(
            cfg,
            baseline_cfg.get("name"),
            int(baseline_cfg.get("frame_count", cfg.get("model", {}).get("frame_count", 4))),
            run_dir,
            use_head=bool(baseline_cfg.get("head", False)),
        )

    # 4) Evaluate
    metrics, times = evaluate_all(cfg, run_dir)
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "retrieval_time.json", times)
    save_results_csv(metrics, times, run_dir / "tables" / "results.csv")
    logger.info("Metrics saved to %s", run_dir / "metrics.json")

    # 5) Build report
    from report.make_report import build_report

    report_root = project_root / "report"
    pdf_path = build_report(cfg, run_dir, report_root)
    final_pdf = run_dir / "report.pdf"
    if pdf_path is not None and pdf_path.exists():
        shutil.copy(pdf_path, final_pdf)
        logger.info("Report copied to %s", final_pdf)
    else:
        logger.warning("Report PDF not built (LaTeX missing?). TeX sources are ready in %s", report_root)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
