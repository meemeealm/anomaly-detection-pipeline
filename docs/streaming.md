# Real-time streaming practice

```
simulate_publisher.py --MQTT--> broker (mosquitto) --MQTT--> consume_score.py
   (Faker + NumPy fleet)         sensors/<id>/readings          micro-batch -> ONNX -> SQLite
                                 fleet/<id>/meta (retained)               |
                                                                streamlit dashboard
```

- Readings are published in physical units; the consumer standardizes them (z-scores) before scoring.
- `label` in each message is simulator ground truth, used only for precision/recall.
  A real device would not send it.
- Latency `e2e_ms` = scored-time minus device timestamp (same machine clock here).
- The consumer sheds load when its queue is full (`dropped` counter) instead of blocking.

## Experiments
1. Raise `--devices` / `--rate` until `queue` or `dropped` grows; find the saturation point.
2. Compare `--batch-size 1` vs `32` and `--max-wait-ms` values (throughput vs latency).
3. Add `--drift 0.2 --drift-feature temperature` and watch precision/recall decay.
4. Retrain on data from the same generator and compare.
5. Kill the broker mid-run; observe reconnect behaviour. Try `--qos 1`.
