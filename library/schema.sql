-- Character library schema
-- SQLite for now; designed so column names + types map cleanly to Postgres later.

CREATE TABLE IF NOT EXISTS characters (
    id          TEXT PRIMARY KEY,          -- uuid4
    name        TEXT NOT NULL UNIQUE,
    trigger     TEXT NOT NULL,             -- LoRA trigger token, e.g. "ohwx_aria"
    base_model  TEXT NOT NULL DEFAULT 'sdxl', -- 'sdxl' | 'flux-dev' | 'flux-schnell'
    lora_path   TEXT,                      -- relative to project root
    video_lora_path TEXT,                  -- Wan 2.2 video LoRA, may be null
    ref_count   INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    notes       TEXT
);

CREATE TABLE IF NOT EXISTS character_refs (
    id           TEXT PRIMARY KEY,
    character_id TEXT NOT NULL REFERENCES characters(id) ON DELETE CASCADE,
    path         TEXT NOT NULL,            -- relative path under characters/<name>/reference-images/
    caption      TEXT,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS generations (
    id           TEXT PRIMARY KEY,
    character_id TEXT REFERENCES characters(id) ON DELETE SET NULL,
    mode         TEXT NOT NULL,            -- 'still' | 'video_mode1' | 'video_mode2' | 'video_mode3'
    output_path  TEXT NOT NULL,
    prompt       TEXT,
    params       TEXT,                     -- JSON blob of workflow params
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_generations_character ON generations(character_id);
CREATE INDEX IF NOT EXISTS idx_generations_mode      ON generations(mode);
