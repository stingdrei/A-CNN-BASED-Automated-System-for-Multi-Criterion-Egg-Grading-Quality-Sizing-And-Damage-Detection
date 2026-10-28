import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.build_cleanliness_dataset import (
    cluster_images,
    filter_rows,
    split_clusters,
)


def row(source="a.jpg", stock="", weak="0", width="100", height="100"):
    return {
        "source_image": source,
        "stock_keyword": stock,
        "weak_label": weak,
        "crop_width": width,
        "crop_height": height,
    }


def test_filter_rows_drops_stock_weak_small_and_excluded():
    rows = [
        row("good.jpg"),
        row("stock.jpg", stock="dreamstime"),
        row("weak.jpg", weak="1"),
        row("small.jpg", width="23"),
        row("excluded.jpg"),
    ]
    kept, dropped = filter_rows(
        rows,
        min_crop_size=24,
        include_weak=False,
        include_stock_keyword=False,
        exclude_images={"excluded.jpg"},
    )

    assert [item["source_image"] for item in kept] == ["good.jpg"]
    assert dropped == Counter(
        {
            "stock_keyword": 1,
            "weak_label": 1,
            "small_crop": 1,
            "excluded_image": 1,
        }
    )


def test_filter_rows_can_keep_weak_and_stock():
    rows = [
        row("stock.jpg", stock="dreamstime"),
        row("weak.jpg", weak="1"),
    ]
    kept, dropped = filter_rows(
        rows,
        min_crop_size=24,
        include_weak=True,
        include_stock_keyword=True,
        exclude_images=set(),
    )

    assert len(kept) == 2
    assert dropped == Counter()


def test_cluster_images_unions_exact_and_near_duplicates():
    metadata = {
        "a.jpg": {"sha256": "same", "phash": 0},
        "b.jpg": {"sha256": "same", "phash": 0},
        "c.jpg": {"sha256": "other", "phash": 1},
        "d.jpg": {"sha256": "different", "phash": 0b1111},
    }

    cluster_ids, report = cluster_images(metadata, near_duplicate_threshold=2)

    assert cluster_ids["a.jpg"] == cluster_ids["b.jpg"] == cluster_ids["c.jpg"]
    assert cluster_ids["d.jpg"] != cluster_ids["a.jpg"]
    assert len(report) == 2


def test_split_clusters_keeps_groups_whole_and_classes_present():
    clusters = [
        {"cluster_id": "c1", "label": "0", "rows": [{}] * 10},
        {"cluster_id": "c2", "label": "0", "rows": [{}] * 2},
        {"cluster_id": "c3", "label": "0", "rows": [{}] * 1},
        {"cluster_id": "s1", "label": "1", "rows": [{}] * 8},
        {"cluster_id": "s2", "label": "1", "rows": [{}] * 2},
        {"cluster_id": "s3", "label": "1", "rows": [{}] * 1},
    ]

    result = split_clusters(clusters, val_size=0.2, test_size=0.2, seed=42)

    seen = []
    for split, split_clusters_list in result.items():
        labels = {cluster["label"] for cluster in split_clusters_list}
        assert labels == {"0", "1"}, split
        seen.extend(cluster["cluster_id"] for cluster in split_clusters_list)
    assert sorted(seen) == sorted(cluster["cluster_id"] for cluster in clusters)
    assert len(seen) == len(set(seen))
    counts = {
        split: sum(len(cluster["rows"]) for cluster in split_clusters_list)
        for split, split_clusters_list in result.items()
    }
    assert counts["train"] > counts["valid"] > 0
    assert counts["train"] > counts["test"] > 0
