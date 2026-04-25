// Aura-9 v2.4 initial schema
// Index on entity_id for fast lookup
CREATE INDEX ON :TaskCompletion(task_id);
CREATE INDEX ON :Session(session_id);
CREATE INDEX ON :Skill(skill_id);
