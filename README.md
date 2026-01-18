# Text-to-Video Retrieval (Offline Demo Ready)

A fully self-contained, CPU-first text-to-video retrieval pipeline. It runs end-to-end offline with a synthetic demo dataset generated locally (no downloads, no GPU, no ffmpeg). Optional configs support MSVD and MSR-VTT subsets when available.

## Quickstart (one command)
1. Install dependencies (CPU-safe):
   ```bash
   pip install -r requirements.txt
   ```
2. Run the full pipeline (generate demo data → preprocess frames → encode → evaluate → report):
   ```bash
   python run_all.py --config configs/demo.yaml
   ```

This produces a new run directory under `outputs/runs/<run_id>/` containing metrics, tables, figures, and the PDF report (if LaTeX is available). CPU is used by default; GPU is auto-detected but never required.

## What the pipeline does
- **Demo dataset**: Generates ~20 tiny mp4 videos (moving colored shapes) plus captions and train/val/test splits.
- **Preprocess**: Extracts frames with OpenCV (no ffmpeg).
- **Encode**: Loads CLIP from `open_clip_torch`, encodes sampled frames, mean-pools to video embeddings; optional small projection head (kept off by default for speed).
- **Baselines**: random ranking, single-frame CLIP, multi-frame CLIP (4 and 8 frames), and proposed (same as multi-frame with optional head).
- **Evaluate**: Computes R@1/5/10, MedR, MeanR, mAP, and retrieval time (text+search and search-only).
- **Report**: Auto-fills LaTeX tables/figures from metrics; compiles `report.pdf` when `latexmk` exists, otherwise leaves the TeX sources ready.

## Configs
- `configs/demo.yaml`: Offline default using the generated synthetic dataset.
- `configs/msvd.yaml`: Template for MSVD if the dataset is locally available.
- `configs/msrvtt_subset.yaml`: Template for a small MSR-VTT subset.

## Notes
- All paths use `pathlib` and are relative to the project root for cross-platform portability.
- Deterministic seeds are set for reproducibility.
- If LaTeX is missing, the run still succeeds and prints how to compile later.
- The code avoids non-portable system calls and depends only on Python packages listed in `requirements.txt`.
