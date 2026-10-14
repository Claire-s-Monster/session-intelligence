-- Issue #216 read-only survey: project_names that span more than one
-- project_path (and would split under worktree-distinct naming). SELECT only.
SELECT count(*) AS split_project_names,
       sum(paths) AS total_paths
FROM (
    SELECT project_name, count(DISTINCT project_path) AS paths
    FROM sessions
    WHERE project_name IS NOT NULL AND project_name <> '_unbound_'
      AND project_path IS NOT NULL AND project_path NOT IN ('', '_unknown_')
    GROUP BY project_name
    HAVING count(DISTINCT project_path) > 1
) s;

SELECT project_name,
       project_path,
       count(*) AS sessions,
       max(started_at) AS last_started
FROM sessions
WHERE project_name IN (
    SELECT project_name
    FROM sessions
    WHERE project_name IS NOT NULL AND project_name <> '_unbound_'
      AND project_path IS NOT NULL AND project_path NOT IN ('', '_unknown_')
    GROUP BY project_name
    HAVING count(DISTINCT project_path) > 1
)
  AND project_path IS NOT NULL AND project_path NOT IN ('', '_unknown_')
GROUP BY project_name, project_path
ORDER BY project_name, sessions DESC
LIMIT 60;

SELECT project_name,
       count(DISTINCT project_path) AS paths,
       count(*) AS learnings
FROM project_learnings
WHERE project_name IS NOT NULL
  AND project_path IS NOT NULL AND project_path NOT IN ('', '_unknown_')
GROUP BY project_name
HAVING count(DISTINCT project_path) > 1
ORDER BY paths DESC
LIMIT 20;
