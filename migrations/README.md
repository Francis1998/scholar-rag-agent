# Migrations

The first release creates SQLite tables automatically for local-first usage. Future Alembic revisions should be added here when schema changes require explicit migration history.

## Additive independent-screening tables

Dual-reviewer screening adds `paper_screening_reviewer_decisions`,
`paper_screening_reviewer_history`, `paper_screening_resolutions` and
`paper_screening_protocols`, created with `CREATE TABLE IF NOT EXISTS` at startup.
Existing `paper_screening_decisions` and `paper_screening_history` rows are not
altered, copied or backfilled; they are read in place as the reserved reviewer
`default`. Rolling back to an earlier release leaves the new tables unused. See
[the dual-reviewer guide](../docs/guides/DUAL_REVIEWER_SCREENING_GUIDE.md#storage-and-migration).
