# Mobile save and history stability

Baseline: main `cb18b3d4f47cb2932baa759876be28de0a6f2228`.
Branch: `codex/main-stability-recovery`.

## Changes

1. Apply shared leave guards to the current mobile UI: in-app navigation, logout, browser history and document unload during capture/save. Intentional completion releases the guard before navigation.
2. Ignore participant save responses after unmount; block immediate duplicate submission. An uncertain create response asks the user to check the participant list before creating again.
3. Add authenticated, bounded recording summaries (20 rows per page) and a single-participant count/date summary. PostgreSQL projects only fields used by existing GDI/cadence functions. Lists and participant detail no longer download every full report; full history is loaded only when the user explicitly requests history interpretation/comparison. Last-row deletion returns to the preceding page.
4. Persist the request ID and exact upload body in IndexedDB before submission. Account/API/participant keys isolate pending requests. Reload recovery checks the existing record first, posts only on 404 and removes the pending body only after a matching record is confirmed. Atomic first-writer-wins prevents competing attempts from replacing the saved body. Logout retains pending requests for recovery under the same account. Browser storage deletion cannot be recovered by this mechanism; permanent rejected requests require investigation rather than automatic creation of a second ID.
5. Preserve existing analysis-start dates and schema-validated legacy graph arrays. Invalid or unknown graph schemas stay explicitly unavailable. Zero-value curves remain finite in report and zoom views.
6. Extend the existing browser integration test discovery to cover the actual mobile UI as well as legacy UI, including real API persistence, delayed responses, reload recovery, pagination and legacy graphs.

## Unchanged

No cloud/deployment configuration, environment files, DB migration, model weights, inference calculations or worker behavior changes. Existing full-detail/history endpoints remain compatible. New summary endpoints must be deployed with the updated UI. Existing model-derived GDI/cadence functions are reused without formula changes.

## Validation

- Existing Node suite: 174 tests passed.
- Mobile and legacy TypeScript checks passed.
- Mobile production build passed; model asset checksum verified by the existing preparation script.
- Python compilation of backend application/tests passed.
- Local Chromium: 7 scenarios passed using actual mobile components and IndexedDB, with mocked HTTP/extraction (late edit, history/unload guards, accepted/unsent receipt recovery after reload, pagination, zero curves/zoom, outbox concurrency/isolation).
- PostgreSQL WASM (PGlite): executed the summary projection with synthetic rows; pagination, compact metrics and private-field exclusion passed. This is supplemental SQL validation, not a substitute for API integration tests.
- Real API/PostgreSQL tests added in `backend/tests/test_measurement_postgres.py` and `backend/tests/e2e/test_mobile_ui.py`. They have NOT been executed locally because Docker/PostgreSQL is unavailable in this environment. Existing Backend CI and Browser Integration CI must pass before merge. Synthetic pose extraction in browser tests does not assess model accuracy.

CI entry point: `cd backend && python -m unittest discover -s tests/e2e -v`, using the existing isolated disposable PostgreSQL CI setup.

## PR preparation

Suggested title: `fix: stabilize mobile saves, recover uploads and paginate history`
Target: `main`; head: `codex/main-stability-recovery`; create as draft until integration CI passes.

Use this document as the PR body. From this checkout, in a terminal authenticated to GitHub:

```powershell
git push -u origin codex/main-stability-recovery
gh pr create --base main --head codex/main-stability-recovery --draft --title "fix: stabilize mobile saves, recover uploads and paginate history" --body-file docs/integration/main-stability-recovery.md
```

Remote publication was attempted but not completed: Git credential-helper failure and GitHub integration HTTP 403 (Resource not accessible by integration). No PR URL or successful remote CI run is claimed.
