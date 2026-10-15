-- Issue #241 read-only survey: sessions stamped with the session-intelligence
-- server's own cwd (hook-bound sessions created without working_directory).
\pset pager off

\echo '== sessions whose project_path is the SI server cwd, by project_name/status =='
SELECT project_name, status, count(*) AS n,
       min(started_at) AS first_seen, max(started_at) AS last_seen
FROM sessions
WHERE project_path = '/home/memento/ClaudeCode/Servers/session-intelligence/development'
GROUP BY project_name, status
ORDER BY n DESC;

\echo '== of those, how many are hook-bound (tag) in the last 14 days =='
SELECT count(*) FILTER (WHERE metadata::text LIKE '%hook-bound%') AS hook_bound,
       count(*) AS total
FROM sessions
WHERE project_path = '/home/memento/ClaudeCode/Servers/session-intelligence/development'
  AND started_at >= now() - interval '14 days';

\echo '== recent ocaml sessions (project_name, path, status) =='
SELECT id, project_name, project_path, status, started_at
FROM sessions
WHERE (project_name ILIKE '%ocaml%' OR project_path ILIKE '%ocaml%')
  AND started_at >= now() - interval '7 days'
ORDER BY started_at DESC
LIMIT 15;
