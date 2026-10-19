# Configuration

| Where | What | Why there |
|---|---|---|
| `configs/model.json` | sensor list, model paths, LightGBM threshold, ONNX threads | you choose them; sensors define everything downstream |
| `configs/stream.json` | broker, topic, QoS, DB path, batch size, max wait, queue size, `flat_eps`, `max_gap_s`, log level | operational choices; reviewable in Git |
| `artifacts/<model>_meta.json` | threshold, score offset, window size, feature list (Isolation Forest) | training produces them; stored with the model so they can't go stale |

**Precedence** (highest first): command-line flag > environment `IOT_<NAME>` (e.g. `IOT_BATCH_SIZE=64`) >
`configs/stream.json` > built-in default. `IOT_STREAM_CONFIG` or `--config` selects another file.
Unknown keys and out-of-range values stop the process at startup. Model-produced values are not overridden by
flags or the config (only `IOT_THRESHOLD` can override a threshold, for experiments).

Every consumer start logs `effective settings, value [source]`, and the same dictionary is stored in
`runs.config`, so any past run can be reproduced:

```sql
SELECT json_extract(config, '$.settings') FROM runs ORDER BY started_at DESC LIMIT 1;
```

`input_name` in `configs/model.json` is no longer used (the real input name is read from the model); you can delete it.
The dashboard reads the same database path as the consumer (`IOT_STREAM_DB` still works as an override).
