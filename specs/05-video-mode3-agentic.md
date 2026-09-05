# Spec 05 — Video Mode 3: Agentic Scene Generation

## Goal
Given a character + a short text prompt (basic or advanced), generate a multi-scene video:
1. (Basic prompts only) Expand the prompt into a cinematic, scene-segmented script via LLM.
2. Break the script into scenes/shots with keyframe descriptions.
3. Generate character-consistent keyframes for each scene transition (via Mode 2 stills).
4. Interpolate between consecutive keyframes (via Mode 1 video).
5. Chain + stitch clips into a continuous video.
6. Optionally upscale via SeedVR2.

## Agent architecture

The agent is a **Python state machine** in `agent/agent.py`. The LLM provider is injected
via a swappable interface (`agent/llm_provider.py`). Default: local Ollama (Qwen2.5-7B or
Llama3.1-8B). Fallback: Claude API for quality comparison.

```
agent/
├── agent.py           # orchestrator state machine
├── llm_provider.py    # abstract LLM interface + Ollama / Claude implementations
├── scene_planner.py   # prompt → scene plan (JSON) 
├── keyframe_gen.py    # scene descriptions → stills (calls Mode 2 ComfyUI workflow)
├── video_gen.py       # keyframe pairs → clips (calls Mode 1 ComfyUI workflow)
└── stitch.py          # clips → final .mp4 via ffmpeg
```

## Step-by-step pipeline

### Step 1 — Prompt expansion (basic prompts only)

```python
# scene_planner.py
EXPAND_SYSTEM = """You are a cinematic script supervisor for AI video generation.
Expand the user's brief prompt into a structured scene plan.
Return valid JSON only, no prose."""

EXPAND_USER = """
User prompt: "{prompt}"
Character: {character_name} — {character_notes}

Return a JSON object with this exact schema:
{{
  "title": "short title",
  "tone": "cinematic tone description",
  "scenes": [
    {{
      "id": 1,
      "description": "what happens in this scene (2-3 sentences, cinematic)",
      "setting": "location/environment",
      "action": "what the character does",
      "start_keyframe_prompt": "detailed image prompt for the opening frame of this scene",
      "end_keyframe_prompt": "detailed image prompt for the closing frame of this scene",
      "transition_to_next": "how this scene flows into the next"
    }}
  ]
}}

Rules:
- 3–6 scenes for a basic prompt
- Each keyframe prompt must include the character's trigger token: {trigger}
- Prompts should describe composition, lighting, expression, and setting concretely
- Keep scenes sequentially logical
"""
```

### Step 2 — Scene plan → keyframe descriptions

The `scenes[].start_keyframe_prompt` and `scenes[].end_keyframe_prompt` from Step 1 become
inputs to the Mode 2 stills pipeline. Each scene boundary is one keyframe.

For N scenes:
- scene 1 start → scene 1 end = scene 2 start → ... → scene N end
- Total keyframes: N+1 (deduplicated boundaries)

### Step 3 — Generate keyframes (parallel where VRAM allows)

```python
# keyframe_gen.py
def generate_keyframe(scene_desc: str, character: dict, client: ComfyClient) -> Path:
    return client.run_workflow(
        "engine/comfyui/workflows/02_stills.json",
        params={
            "lora_path": character["lora_path"],
            "trigger_token": character["trigger"],
            "positive_prompt": scene_desc,
            "batch_size": 1,
            "seed": -1,
        }
    )["path"]
```

Output: one PNG per keyframe, saved to `outputs/images/<character>/agentic/<run_id>/kf_<n>.png`.

### Step 4 — Interpolate between consecutive keyframes

For each consecutive pair (kf_0→kf_1, kf_1→kf_2, …):

```python
# video_gen.py
def interpolate(kf_a: Path, kf_b: Path, prompt: str, character: dict, client: ComfyClient) -> Path:
    return client.run_workflow(
        "engine/comfyui/workflows/03_video_mode1_ltx.json",
        params={
            "start_frame_path": str(kf_a),
            "end_frame_path": str(kf_b),
            "positive_prompt": prompt,
            "trigger_token": character["trigger"],
            "video_lora_path": character.get("video_lora_path"),
            "num_frames": 97,
            "fps": 24,
        }
    )["path"]
```

Output: one .mp4 per scene segment, in `outputs/videos/<character>/agentic/<run_id>/seg_<n>.mp4`.

### Step 5 — Stitch

```python
# stitch.py  — wraps ffmpeg
def stitch(segments: list[Path], output: Path, crossfade_s: float = 0.5):
    # ffmpeg concat with optional xfade between segments
    ...
```

Final output: `outputs/videos/<character>/agentic/<run_id>/final.mp4`.

## agent.py state machine

```python
class AgentState(Enum):
    EXPAND      = "expand"
    PLAN        = "plan"
    KEYFRAMES   = "keyframes"
    VIDEOS      = "videos"
    STITCH      = "stitch"
    DONE        = "done"
    ERROR       = "error"

class Agent:
    def __init__(self, llm: LLMProvider, comfy: ComfyClient, library: CharacterLibrary):
        ...
    def run(self, character_name: str, prompt: str, advanced: bool = False) -> Path:
        ...
```

State is persisted to `outputs/videos/<character>/agentic/<run_id>/state.json` so a failed
run can be resumed from the last completed step.

## LLM provider interface

```python
# llm_provider.py
class LLMProvider(ABC):
    @abstractmethod
    def complete(self, system: str, user: str, json_mode: bool = False) -> str: ...

class OllamaProvider(LLMProvider):
    def __init__(self, model: str = "qwen2.5:7b", base_url: str = "http://localhost:11434"):
        ...

class ClaudeProvider(LLMProvider):
    def __init__(self, model: str = "claude-opus-4-8", api_key: str = ...):
        ...
```

On Colab: run Ollama in a background subprocess, or skip local LLM and use Claude API
(set `ANTHROPIC_API_KEY` env var).

## Models required
All from Specs 02 and 03 for generation. Additionally:
| Tool | Purpose |
|---|---|
| Ollama + Qwen2.5-7B or Llama3.1-8B | Local LLM for prompt expansion |
| ffmpeg | Clip stitching |

## Validation
- Expanded script should have 3–6 coherent scenes with valid JSON structure.
- Keyframes should all feature the same recognizable character.
- Consecutive keyframe pairs should be visually "connectable" (similar setting/lighting).
- Final stitched video should play smoothly; crossfades should hide hard cuts.
- Run `state.json` check: every step should be `completed` before final output is logged.
