# Phase D - Multimodal and Extended Evaluation

Objective: add ViT/PaliGemma path and broaden eval coverage once text and platform surfaces are stable.

Legend:
- Priority: `D-P0` (highest), `D-P1`
- Status: `done`, `in_progress`, `todo`

## D-P0

1. `D-P0-L` ViT minimal forward path
- Status: `todo`
- Scope: replace placeholder `vit.rs` with working minimal inference path
- Target files:
  - `crates/gemma-core/src/vit.rs`

2. `D-P0-M` Image preprocessing module
- Status: `todo`
- Scope: preprocessing and feature wiring for multimodal input

3. `D-P0-M` Multimodal prompt flow smoke tests
- Status: `todo`
- Scope: gated tests validating image + text flow end-to-end

## D-P1

4. `D-P1-M` Expanded eval harness and reporting
- Status: `todo`
- Scope: golden prompt/image reports and CI summary artifacts

5. `D-P1-S` Docs and usage examples
- Status: `todo`
- Scope: multimodal usage guidance and constraints
