# Cleanliness candidate crops

Generated from the web/stock Roboflow sources `clean eggs/` and
`Eggs degrading quality/`. This is weak pretraining data, not a release test
set. It contains background/domain bias and possible visible watermarks.

Mapping:
- `Clean`: Brownegg, Lightbrownegg, Whiteegg from `clean eggs/`
- `Stained`: dirt stained brown/white egg from `Eggs degrading quality/`
- Excluded: weak `white egg` labels, stock-keyword images, known text overlays,
  invalid/small boxes, and near-duplicate source images.

Splits are assigned by source image cluster, so all crops from one source image
stay in one split. Source datasets have no session IDs; this is the strongest
leakage control available.

Train only after manually reviewing the previews in
`reports/cleanliness_source_previews/`.
