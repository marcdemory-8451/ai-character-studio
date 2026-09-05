"""Mode 3 — Agentic video generation orchestrator.

Usage:
    from agent.agent import Agent
    from agent.llm_provider import LLMProvider
    from engine.comfyui.setup.comfy_client import ComfyClient
    from library.library import CharacterLibrary

    agent = Agent(
        llm=LLMProvider.default(),
        comfy=ComfyClient("http://localhost:8188"),
        library=CharacterLibrary(),
    )
    output = agent.run("Aria", "Aria sneaks through a moonlit forest to rescue a stolen artifact")
    print(output)  # path to final .mp4
"""

from __future__ import annotations
import json
import subprocess
import time
import uuid
from enum import Enum
from pathlib import Path

from .llm_provider import LLMProvider
from .scene_planner import ScenePlan, expand_prompt


class State(str, Enum):
    EXPAND    = "expand"
    KEYFRAMES = "keyframes"
    VIDEOS    = "videos"
    STITCH    = "stitch"
    DONE      = "done"
    ERROR     = "error"


class Agent:
    STILLS_WORKFLOW  = "engine/comfyui/workflows/02_stills.json"
    VIDEO_WORKFLOW   = "engine/comfyui/workflows/03_video_mode1_ltx.json"
    OUTPUTS_DIR      = Path("outputs/videos")

    def __init__(self, llm: LLMProvider, comfy, library):
        self.llm     = llm
        self.comfy   = comfy
        self.library = library

    def run(
        self,
        character_name: str,
        prompt: str,
        advanced: bool = False,
        run_id: str = None,
    ) -> Path:
        run_id = run_id or str(uuid.uuid4())[:8]
        char   = self.library.get_character_by_name(character_name)
        if not char:
            raise ValueError(f"Character '{character_name}' not found in library.")

        run_dir = self.OUTPUTS_DIR / character_name / "agentic" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        state_file = run_dir / "state.json"

        state = self._load_state(state_file) or {
            "state": State.EXPAND,
            "run_id": run_id,
            "character": character_name,
            "prompt": prompt,
            "advanced": advanced,
            "plan": None,
            "keyframe_paths": [],
            "clip_paths": [],
            "final_path": None,
        }

        try:
            # ── 1. Expand ────────────────────────────────────────────────────
            if state["state"] == State.EXPAND:
                print(f"[Agent] Expanding prompt...")
                plan = expand_prompt(prompt, char, self.llm, advanced=advanced)
                state["plan"] = {
                    "title": plan.title,
                    "tone": plan.tone,
                    "scenes": plan.scenes,
                    "keyframes": [
                        {"scene_id": kf.scene_id, "position": kf.position, "prompt": kf.prompt}
                        for kf in plan.keyframes
                    ],
                }
                state["state"] = State.KEYFRAMES
                self._save_state(state_file, state)
                print(f"[Agent] Plan: '{plan.title}' — {len(plan.scenes)} scenes, {len(plan.keyframes)} keyframes")

            # ── 2. Generate keyframes ────────────────────────────────────────
            if state["state"] == State.KEYFRAMES:
                plan_data = state["plan"]
                kf_paths  = list(state.get("keyframe_paths", []))
                kf_list   = plan_data["keyframes"]

                for i, kf in enumerate(kf_list):
                    if i < len(kf_paths) and kf_paths[i]:
                        print(f"[Agent] Keyframe {i+1}/{len(kf_list)} already done, skipping.")
                        continue
                    print(f"[Agent] Generating keyframe {i+1}/{len(kf_list)}: {kf['prompt'][:60]}...")
                    result = self.comfy.run_workflow(
                        self.STILLS_WORKFLOW,
                        params={
                            "lora_path": char["lora_path"],
                            "trigger_token": char["trigger"],
                            "positive_prompt": kf["prompt"],
                            "batch_size": 1,
                            "seed": 42 + i,
                        },
                        output_dir=run_dir / "keyframes",
                    )
                    kf_paths.append(result["path"])
                    state["keyframe_paths"] = kf_paths
                    self._save_state(state_file, state)

                state["state"] = State.VIDEOS
                self._save_state(state_file, state)

            # ── 3. Interpolate between consecutive keyframes ─────────────────
            if state["state"] == State.VIDEOS:
                kf_paths   = state["keyframe_paths"]
                clip_paths = list(state.get("clip_paths", []))
                scenes     = state["plan"]["scenes"]

                for i in range(len(kf_paths) - 1):
                    if i < len(clip_paths) and clip_paths[i]:
                        print(f"[Agent] Clip {i+1} already done, skipping.")
                        continue
                    scene_idx = min(i, len(scenes) - 1)
                    hint = scenes[scene_idx].get("interpolation_hint", "")
                    prompt_for_clip = f"{char['trigger']}, {hint}" if hint else char["trigger"]

                    print(f"[Agent] Generating clip {i+1}/{len(kf_paths)-1}...")
                    result = self.comfy.run_workflow(
                        self.VIDEO_WORKFLOW,
                        params={
                            "start_frame_path": kf_paths[i],
                            "end_frame_path": kf_paths[i + 1],
                            "positive_prompt": prompt_for_clip,
                            "trigger_token": char["trigger"],
                            "video_lora_path": char.get("video_lora_path"),
                            "num_frames": 97,
                            "fps": 24,
                        },
                        output_dir=run_dir / "clips",
                    )
                    clip_paths.append(result["path"])
                    state["clip_paths"] = clip_paths
                    self._save_state(state_file, state)

                state["state"] = State.STITCH
                self._save_state(state_file, state)

            # ── 4. Stitch clips ──────────────────────────────────────────────
            if state["state"] == State.STITCH:
                final_path = run_dir / "final.mp4"
                self._stitch(state["clip_paths"], final_path)
                state["final_path"] = str(final_path)
                state["state"] = State.DONE
                self._save_state(state_file, state)
                self.library.log_generation(
                    "video_mode3", str(final_path),
                    character_id=char["id"], prompt=prompt,
                    params={"run_id": run_id, "scenes": len(state["plan"]["scenes"])},
                )
                print(f"[Agent] Done. Final video: {final_path}")

            return Path(state["final_path"])

        except Exception as exc:
            state["state"] = State.ERROR
            state["error"] = str(exc)
            self._save_state(state_file, state)
            raise

    @staticmethod
    def _stitch(clip_paths: list[str], output: Path, crossfade_s: float = 0.0):
        """Concatenate clips with ffmpeg. Set crossfade_s > 0 for xfade filter."""
        concat_list = output.parent / "concat.txt"
        with open(concat_list, "w") as f:
            for p in clip_paths:
                f.write(f"file '{Path(p).resolve()}'\n")

        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", str(output)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg stitch failed: {result.stderr}")

    @staticmethod
    def _load_state(path: Path) -> dict | None:
        if path.exists():
            return json.loads(path.read_text())
        return None

    @staticmethod
    def _save_state(path: Path, state: dict):
        path.write_text(json.dumps(state, indent=2, default=str))
