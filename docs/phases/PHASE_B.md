# Phase B - API and Integration Surfaces

Objective: expose stable serving/client/ABI surfaces after Phase A parity core is locked.

Legend:
- Priority: `B-P0` (highest), `B-P1`, `B-P2`
- Status: `done`, `in_progress`, `todo`

## B-P0

1. `B-P0-M` API server implementation
- Status: `todo`
- Scope: real `/health`, `/generate`, and session cache flows
- Target files:
  - `crates/gemma-core/src/api_server.rs`

2. `B-P0-S` API client implementation
- Status: `todo`
- Scope: request/response path against server endpoints
- Target files:
  - `crates/gemma-core/src/api_client.rs`

3. `B-P0-M` C API implementation
- Status: `todo`
- Scope: ABI-safe `init/generate/free` surface and ownership rules
- Target files:
  - `crates/gemma-core/src/c_api.rs`

## B-P1

4. `B-P1-M` Integration tests for server/client/C API
- Status: `todo`
- Scope: smoke coverage + error handling + basic concurrency behavior
- Target files:
  - `crates/gemma-core/tests/*` (new integration tests)

5. `B-P1-S` API docs and compatibility notes
- Status: `todo`
- Scope: endpoint and ABI contracts + versioning policy

## B-P2

6. `B-P2-S` Packaging and examples
- Status: `todo`
- Scope: runnable local examples for CLI/server/C API usage
