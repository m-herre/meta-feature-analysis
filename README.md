# Meta-Feature Analysis

Standalone package for running pairwise TabArena meta-feature analyses.

## Repository Layout

This package is intended to live in its own Git repository, separate from the sibling `tabarena/` checkout.

## Local Development

The project depends on a local `tabarena` installation or sibling checkout during development and testing.

To record which TabArena revision an analysis used, save the output of:

```bash
git -C ../tabarena rev-parse HEAD
```
