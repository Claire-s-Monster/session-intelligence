-- Issue #216 read-only detail: real (non-archive, non-/tmp) project_names
-- spanning more than one project_path, with each path's session count. SELECT only.
WITH real_sessions AS (
    SELECT project_name, project_path, started_at
    FROM sessions
    WHERE project_name IS NOT NULL
      AND project_name NOT IN ('_unbound_', '_archive_pre_fix_')
      AND project_path IS NOT NULL
      AND project_path NOT IN ('', '_unknown_', '.claude')
      AND project_path NOT LIKE '/tmp/%'
), split AS (
    SELECT project_name
    FROM real_sessions
    GROUP BY project_name
    HAVING count(DISTINCT project_path) > 1
)
SELECT r.project_name,
       r.project_path,
       count(*) AS sessions,
       max(r.started_at)::date AS last_started
FROM real_sessions r
JOIN split USING (project_name)
GROUP BY r.project_name, r.project_path
ORDER BY r.project_name, sessions DESC
LIMIT 200;
