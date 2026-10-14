# Checkpoint release plan

The [registry.json](registry.json) file is the single source for reserved checkpoint IDs, filenames, availability and HF destinations. **Placeholder entries are plans, not downloadable weights.** No `.pt` or `.pth` dummy files are created.

## Existing inference release

The existing [agaresd/anychest-inference](https://huggingface.co/agaresd/anychest-inference) bundle remains separately recorded as `legacy_available`. It contains the existing PA, lateral and oblique inference weights, not full training state. Preparing the next release does not replace that bundle.

## Next release placeholders

| Model | Full training checkpoint slots | Inference-only slot | Status |
| --- | --- | --- | --- |
| PA | `pa/latest.pth`, `pa/best_no_jtr.pth`, `pa/best_jtr.pth` | `pa/inference.pt` | Placeholder |
| Lateral | `la/latest.pth`, `la/best_no_jtr.pth`, `la/best_jtr.pth` | `la/inference.pt` | Placeholder |
| Oblique | `oblique/latest.pth`, `oblique/best_no_jtr.pth`, `oblique/best_jtr.pth` | `oblique/inference.pt` | Placeholder |
| Combined PA + lateral + oblique | — | `anychest_inference_bundle.pt` | Placeholder |

These are **13 reserved artifacts**: 9 full checkpoints, 3 individual inference checkpoints and 1 combined inference bundle. Six angle-specific oblique profiles use the shared oblique model; they are not six independent checkpoints. `latest` is intended for continuation, `best_no_jtr` for the no-CAR phase and `best_jtr` for the CAR phase. The checkpoint selected for each inference export is not decided yet.

## Two HF release types

| Release | Intended contents | Model card | HF destination |
| --- | --- | --- | --- |
| Full training | Selected full checkpoints with optimizer/scheduler and training metadata; configuration and split provenance | [Full model-card template](../hf/full/README.md) | Unassigned |
| Inference-only | Slim model weights and required model/view/normalization metadata; reference JSON; combined bundle | [Inference model-card template](../hf/inference/README.md) | Unassigned |

For every future artifact, the HF repository/revision, SHA-256, byte size, source checkpoint, epoch and metrics remain `null` until actual weights and verification are available. The filenames above reserve locations; they do not establish that a checkpoint exists or reproduces a particular reported experiment. Full checkpoints do not promise exact continuation unless the required continuation state has been verified.

Check the registry offline:

```bash
python scripts/check_checkpoint_registry.py
```

Require all future artifacts to be ready:

```bash
python scripts/check_checkpoint_registry.py --require-ready
```

The first command validates the preparation structure. The second intentionally fails while placeholders remain. Neither command downloads or uploads weights. HF account/repository names and publication are deferred; see [HF preparation](../hf/README.md).
