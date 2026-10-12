-- Issue #226: read-only survey of decisions.artifacts encoding.
-- Run: psql -d session_intelligence -v ON_ERROR_STOP=1 -f scripts/sql/issue226_survey.sql
SET default_transaction_read_only = on;
BEGIN;
SELECT jsonb_typeof(artifacts) AS kind, count(*) FROM decisions GROUP BY 1 ORDER BY 1;
SELECT (artifacts #>> '{}') = '[]' AS empty, count(*)
  FROM decisions WHERE jsonb_typeof(artifacts) = 'string' GROUP BY 1 ORDER BY 1;
SELECT count(*) AS not_array_text
  FROM decisions
 WHERE jsonb_typeof(artifacts) = 'string' AND left(artifacts #>> '{}', 1) <> '[';
ROLLBACK;
