"""Scene planner — expands a basic prompt into a structured scene plan via LLM."""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from .llm_provider import LLMProvider


EXPAND_SYSTEM = """You are a cinematic script supervisor for AI video generation.
Your job is to take a user's brief prompt and expand it into a structured, scene-by-scene
plan that can drive AI image and video generation.

Rules:
- Return ONLY valid JSON, no prose, no markdown fences.
- 3 to 6 scenes for a short prompt; more only if the prompt demands it.
- Each scene description should be 2-3 sentences, cinematically specific.
- Keyframe prompts must be detailed, concrete image generation prompts.
- Every keyframe prompt must begin with the character's trigger token.
- Prompts should specify: composition, lighting, setting, expression, action, clothing.
"""

EXPAND_USER_TEMPLATE = """
User prompt: "{prompt}"

Character name: {character_name}
Character trigger token: {trigger}
Character notes: {notes}

Return a JSON object with exactly this schema:
{{
  "title": "short evocative title for this story beat",
  "tone": "overall cinematic tone (e.g. 'tense thriller', 'warm adventure', etc.)",
  "scenes": [
    {{
      "id": 1,
      "description": "what happens in this scene",
      "setting": "specific location and time of day",
      "action": "what the character is doing",
      "start_keyframe_prompt": "detailed diffusion model prompt for opening frame of this scene",
      "end_keyframe_prompt": "detailed diffusion model prompt for closing frame of this scene",
      "interpolation_hint": "description of the motion/change between start and end frames"
    }}
  ]
}}
"""


@dataclass
class Keyframe:
    scene_id: int
    position: str  # 'start' or 'end'
    prompt: str
    image_path: str = ""  # filled in after generation


@dataclass
class ScenePlan:
    title: str
    tone: str
    scenes: list[dict]
    keyframes: list[Keyframe] = field(default_factory=list)

    def build_keyframes(self) -> list[Keyframe]:
        """Derive the deduplicated ordered keyframe list from scenes."""
        kfs = []
        seen_prompts = set()
        for scene in self.scenes:
            for pos in ("start", "end"):
                prompt = scene[f"{pos}_keyframe_prompt"]
                if prompt not in seen_prompts:
                    kfs.append(Keyframe(
                        scene_id=scene["id"],
                        position=pos,
                        prompt=prompt,
                    ))
                    seen_prompts.add(prompt)
        self.keyframes = kfs
        return kfs


def expand_prompt(
    prompt: str,
    character: dict,
    llm: LLMProvider,
    advanced: bool = False,
) -> ScenePlan:
    """Expand a basic prompt into a ScenePlan. Advanced prompts are returned as-is (single scene)."""
    if advanced:
        # Treat the prompt as a single finished scene with no expansion
        return ScenePlan(
            title="Advanced prompt",
            tone="user-defined",
            scenes=[{
                "id": 1,
                "description": prompt,
                "setting": "",
                "action": "",
                "start_keyframe_prompt": f"{character['trigger']}, {prompt}, opening frame",
                "end_keyframe_prompt": f"{character['trigger']}, {prompt}, closing frame",
                "interpolation_hint": prompt,
            }],
        )

    user = EXPAND_USER_TEMPLATE.format(
        prompt=prompt,
        character_name=character.get("name", "character"),
        trigger=character.get("trigger", ""),
        notes=character.get("notes", "no additional notes"),
    )

    raw = llm.complete(EXPAND_SYSTEM, user, json_mode=True)

    # Strip markdown fences if the model adds them despite instructions
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip().rstrip("```").strip()

    data = json.loads(raw)
    plan = ScenePlan(title=data["title"], tone=data["tone"], scenes=data["scenes"])
    plan.build_keyframes()
    return plan
