# Hugging Face release preparation

This directory contains local model-card templates for two future releases:

- [Full training checkpoints](full/README.md)
- [Inference-only checkpoints](inference/README.md)

No new HF repository or model weight has been published by this preparation. Repository IDs are deliberately unassigned rather than represented by links to nonexistent releases. The existing `agaresd/anychest-inference` release remains available independently.

Artifact names and availability are maintained in [checkpoints/registry.json](../checkpoints/registry.json). Release provenance fields are reserved in [release-metadata.template.json](release-metadata.template.json): code commit, configuration hash, dataset version, frozen split hash, class-reference hash, license and validation receipt. All unspecified fields are `null`.

When the next checkpoints are supplied, identify their actual training source/configuration, select the exports, check model compatibility and class order, and fill hashes, sizes and immutable revisions after verification. Review the training source and model-weight license terms before preparing a release. The current templates omit a final license tag because checkpoint terms are not yet established.

Keep actual binaries under an ignored local directory such as `checkpoints/artifacts/` or `hf/staging/`. Publish model cards and weights only as a later release action. These templates contain no download commands for unassigned repositories.
