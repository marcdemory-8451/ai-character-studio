"""Character library — CRUD helpers over SQLite.

Usage:
    from library.library import CharacterLibrary
    lib = CharacterLibrary()          # opens ./library/characters.db
    char = lib.create_character("Aria", trigger="ohwx_aria")
    lib.set_lora(char["id"], "engine/training/loras/aria.safetensors")
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).parent
_SCHEMA = _HERE / "schema.sql"
_DEFAULT_DB = _HERE / "characters.db"


class CharacterLibrary:
    def __init__(self, db_path: Path | str = _DEFAULT_DB):
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._migrate()

    def _migrate(self):
        self._conn.executescript(_SCHEMA.read_text())
        self._conn.commit()

    # ── characters ──────────────────────────────────────────────────────────

    def create_character(self, name: str, trigger: str = "", base_model: str = "sdxl", notes: str = "") -> dict:
        if not trigger:
            trigger = "ohwx_" + name.lower().replace(" ", "_")
        cid = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO characters (id, name, trigger, base_model, notes) VALUES (?,?,?,?,?)",
            (cid, name, trigger, base_model, notes),
        )
        self._conn.commit()
        char_dir = _HERE.parent / "characters" / name
        for sub in ("reference-images", "captions", "loras", "turnarounds", "samples"):
            (char_dir / sub).mkdir(parents=True, exist_ok=True)
        (char_dir / "metadata.json").write_text(
            json.dumps({"id": cid, "name": name, "trigger": trigger, "base_model": base_model}, indent=2)
        )
        return self.get_character(cid)

    def get_character(self, char_id: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM characters WHERE id=?", (char_id,)).fetchone()
        return dict(row) if row else None

    def get_character_by_name(self, name: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM characters WHERE name=?", (name,)).fetchone()
        return dict(row) if row else None

    def list_characters(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM characters ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def set_lora(self, char_id: str, lora_path: str):
        self._conn.execute("UPDATE characters SET lora_path=? WHERE id=?", (lora_path, char_id))
        self._conn.commit()

    def set_video_lora(self, char_id: str, lora_path: str):
        self._conn.execute("UPDATE characters SET video_lora_path=? WHERE id=?", (lora_path, char_id))
        self._conn.commit()

    # ── reference images ────────────────────────────────────────────────────

    def add_ref(self, char_id: str, path: str, caption: str = "") -> dict:
        rid = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO character_refs (id, character_id, path, caption) VALUES (?,?,?,?)",
            (rid, char_id, path, caption),
        )
        self._conn.execute("UPDATE characters SET ref_count=ref_count+1 WHERE id=?", (char_id,))
        self._conn.commit()
        return {"id": rid, "character_id": char_id, "path": path, "caption": caption}

    def list_refs(self, char_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM character_refs WHERE character_id=? ORDER BY created_at", (char_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── generations ─────────────────────────────────────────────────────────

    def log_generation(self, mode: str, output_path: str, character_id: str = None, prompt: str = "", params: dict = None) -> dict:
        gid = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO generations (id, character_id, mode, output_path, prompt, params) VALUES (?,?,?,?,?,?)",
            (gid, character_id, mode, output_path, prompt, json.dumps(params or {})),
        )
        self._conn.commit()
        return {"id": gid, "output_path": output_path}

    def list_generations(self, character_id: str = None, mode: str = None, limit: int = 50) -> list[dict]:
        q = "SELECT * FROM generations WHERE 1=1"
        args = []
        if character_id:
            q += " AND character_id=?"; args.append(character_id)
        if mode:
            q += " AND mode=?"; args.append(mode)
        q += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        return [dict(r) for r in self._conn.execute(q, args).fetchall()]

    def close(self):
        self._conn.close()
