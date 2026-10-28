# Checkpoint Checksums

Generated 2026-10-28 with `shasum -a 256`.

| Artifact | SHA-256 |
|---|---|
| `models/egg_grader_crop_resnet18_v1.pth` | `9fe5d113a5c1c5faebec17b806a09975c50f2cac3cbf67f9a06d5432b26f1861` |
| `models/egg_grader_cleanliness_v1.pth` | `b00884a1f6b03b87e2f8b3378081cbadbbdab9e68218de082d3c624e3c866a6f` |
| `egg_detection/finetune_egg_v1/weights/best.pt` | `c755a7a6cf5ad883486e0396dc0dfda5030ada3007315e6ef479b26a99a931cf` |
| `egg_detection/train1/weights/best.pt` | `2dbe8d58fc25776d86cbb970ea9ef04e11f78b5bd3a4f2ee0c50670d3ad9bf2c` |

The damage ResNet18 and cleanliness ResNet18 checkpoints are the application
defaults. The cleanliness checkpoint is a weak web/stock candidate, not a
release-approved model. Verify these hashes before deploying or reproducing
test reports.
