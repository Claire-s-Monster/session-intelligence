-- Issue #226 backfill: unwrap double-encoded decisions.artifacts
-- (JSONB string scalar -> JSONB array).
-- Run ONLY after the #226 fix is deployed, or the old server keeps writing
-- string rows.
-- Run: psql -d session_intelligence -v ON_ERROR_STOP=1 -f scripts/sql/issue226_backfill.sql
-- Atomic: any guard failure or bad cast aborts the whole transaction.
BEGIN;
DO $$
DECLARE bad integer;
BEGIN
  SELECT count(*) INTO bad FROM decisions
   WHERE jsonb_typeof(artifacts) = 'string' AND left(artifacts #>> '{}', 1) <> '[';
  IF bad > 0 THEN
    RAISE EXCEPTION 'issue226: % string rows are not JSON array text; aborting', bad;
  END IF;
END $$;
UPDATE decisions SET artifacts = (artifacts #>> '{}')::jsonb
 WHERE jsonb_typeof(artifacts) = 'string';
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM decisions WHERE jsonb_typeof(artifacts) <> 'array') THEN
    RAISE EXCEPTION 'issue226: non-array artifacts remain after backfill; rolling back';
  END IF;
END $$;
COMMIT;
SELECT jsonb_typeof(artifacts) AS kind, count(*) FROM decisions GROUP BY 1 ORDER BY 1;
