import argparse
import sys
import os
import subprocess
def install_requirements():
   
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    req_file = os.path.join(root_dir, "requirements.txt")

    if os.path.exists(req_file):
        logger.info(f"Installing dependencies from {req_file} ...")

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", req_file]
        )

        logger.info("Dependencies installed successfully.")
    else:
        logger.warning("requirements.txt not found in project root. Skipping dependency installation.")


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.corpus_prep import run_ingestion
from src.fuzzy_clustering import run_clustering
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full preprocessing and clustering pipeline for 20NG semantic search.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_pipeline.py --dataset-path data/20_newsgroups
  python scripts/run_pipeline.py --dataset-path /tmp/20news --skip-k-selection
  python scripts/run_pipeline.py --dataset-path data/20_newsgroups --output-dir my_data
        """
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/20_newsgroups",
        help=(
            "Path to the root of the 20 Newsgroups dataset directory. "
            "Should contain subdirectories like alt.atheism/, comp.graphics/, etc. "
            "(default: data/20_newsgroups)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory to write embeddings, corpus metadata, and clustering artifacts (default: data/)",
    )
    parser.add_argument(
        "--chroma-dir",
        type=str,
        default="data/chroma_db",
        help="Directory for ChromaDB persistence (default: data/chroma_db)",
    )
    parser.add_argument(
        "--skip-k-selection",
        action="store_true",
        help=(
            "Skip the cluster number selection analysis (saves ~30 mins). "
            "Use this if you've already run it or want to go straight to k=15."
        ),
    )
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help=(
            "Skip embedding and ingestion steps. "
            "Use if embeddings already exist at output-dir/embeddings.npy"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    install_requirements()

    logger.info("=" * 60)
    logger.info("20 Newsgroups Semantic Search — Setup Pipeline")
    logger.info("=" * 60)
    logger.info(f"Dataset path:       {args.dataset_path}")
    logger.info(f"Output directory:   {args.output_dir}")
    logger.info(f"ChromaDB directory: {args.chroma_dir}")
    logger.info(f"Skip k-selection:   {args.skip_k_selection}")
    logger.info(f"Skip embedding:     {args.skip_embedding}")

    os.makedirs(args.output_dir, exist_ok=True)

    embeddings_path = os.path.join(args.output_dir, "embeddings.npy")
    corpus_meta_path = os.path.join(args.output_dir, "corpus_meta.parquet")

   
    if not args.skip_embedding:
        logger.info("\n[PART 1] Loading corpus, cleaning, embedding, ingesting into ChromaDB...")
        df, embeddings = run_ingestion(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            chroma_persist_dir=args.chroma_dir,
        )
        logger.info(f"[PART 1] Complete. {len(df)} documents embedded and stored.")
    else:
        import numpy as np
        import pandas as pd
        if not os.path.exists(embeddings_path):
            logger.error(f"--skip-embedding specified but {embeddings_path} not found!")
            sys.exit(1)
        logger.info(f"[PART 1] Skipped. Using existing embeddings at {embeddings_path}")

    
    logger.info("\n[PART 2] Running Fuzzy C-Means clustering...")
    run_clustering(
        embeddings_path=embeddings_path,
        corpus_meta_path=corpus_meta_path,
        output_dir=args.output_dir,
        run_k_selection=not args.skip_k_selection,
    )
    logger.info("[PART 2] Complete. Clustering artifacts saved.")

   
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline complete! To start the API server, run:")
    logger.info("")
    logger.info("  uvicorn src.main:app --host 0.0.0.0 --port 8000")
    logger.info("")
    logger.info("Or with auto-reload for development:")
    logger.info("  uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
