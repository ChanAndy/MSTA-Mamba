# Data manifests

No medical data is distributed with this repository.

Each dataset uses a `manifest.csv` with these columns:

```csv
video_id,video_path,split,patient_id,labels
example_001,videos/example_001.avi,train,patient_001,"[0,0,1,0,0,0,2,0]"
```

- `video_path` is relative to the manifest directory (absolute paths are also accepted).
- `split` is `train`, `val`, or `test`.
- `patient_id` exists to audit patient-level separation; it is not returned to the model.
- `labels` is a JSON array aligned with decoded video frames: background=0, ED=1, ES=2.
- POCUS uses only labels 0 and 1. EchoNet-Dynamic uses all three labels.

Before training, verify that no `patient_id` occurs in more than one split. Do not put
identifiable patient IDs into a public manifest.

