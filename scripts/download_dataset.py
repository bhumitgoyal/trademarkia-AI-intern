import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def download_via_sklearn(output_dir: str) -> str:
    
    from sklearn.datasets import fetch_20newsgroups
    from loguru import logger

    logger.info("Downloading 20 Newsgroups via sklearn (all categories, all subsets)...")
    logger.info("This downloads ~14MB and may take a few minutes...")


    dataset = fetch_20newsgroups(
        subset="all",
        remove=(),          
        download_if_missing=True,
    )

    logger.info(f"Downloaded {len(dataset.data)} articles")


    out_path = Path(output_dir) / "20_newsgroups"
    out_path.mkdir(parents=True, exist_ok=True)


    category_names = dataset.target_names


    for cat in category_names:
        (out_path / cat).mkdir(exist_ok=True)


    category_counts = {cat: 0 for cat in category_names}
    for i, (text, label_idx) in enumerate(zip(dataset.data, dataset.target)):
        category = category_names[label_idx]
        count = category_counts[category]
        article_path = out_path / category / str(count)
        article_path.write_text(text, encoding="utf-8")
        category_counts[category] += 1

    total = sum(category_counts.values())
    logger.info(f"Written {total} articles to {out_path}")
    for cat, count in sorted(category_counts.items()):
        logger.info(f"  {cat}: {count} articles")

    return str(out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Download the 20 Newsgroups dataset to disk."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory to write dataset (default: data/). "
             "Dataset will be at <output-dir>/20_newsgroups/",
    )
    args = parser.parse_args()

    dataset_path = download_via_sklearn(args.output_dir)
    print(f"\nDataset ready at: {dataset_path}")
    print("\nNext step:")
    print(f"  python scripts/run_pipeline.py --dataset-path {dataset_path}")


if __name__ == "__main__":
    main()
