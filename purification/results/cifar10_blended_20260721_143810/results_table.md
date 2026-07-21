| Method | CA (%) | DR (%) | Notes |
|--------|--------|--------|-------|
| No Defense | 85.8 | 100.0 | poisoned model, no purification |
| Direct Removal | 87.8 | 95.8 | Direct Removal |
| Fine-Tuning | 10.1 | 100.0 | Fine-Tuning |
| Ours (Purification) | 10.2 | 100.0 | Purification + retrain (200 purified + 2000 clean) |
| NAD | 11.7 | 100.0 | NAD (simplified) |

### Purification Details
- Samples purified: 200
- Correct after purification: 200
- Purification accuracy: 100.0%