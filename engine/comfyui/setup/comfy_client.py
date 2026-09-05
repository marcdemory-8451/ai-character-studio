"""Thin ComfyUI API client — submit workflows, poll for results.

Usage:
    client = ComfyClient("http://localhost:8188")
    result = client.run_workflow("engine/comfyui/workflows/02_stills.json", params={...})
    print(result["path"])  # path to output file
"""

import copy
import json
import time
import uuid
import urllib.request
import urllib.parse
import websocket
from pathlib import Path


class ComfyClient:
    def __init__(self, base_url: str = "http://localhost:8188"):
        self.base_url = base_url.rstrip("/")
        self.client_id = str(uuid.uuid4())

    # ── low-level API ────────────────────────────────────────────────────────

    def queue_prompt(self, workflow: dict) -> str:
        """POST a workflow dict to /prompt, return prompt_id."""
        payload = json.dumps({"prompt": workflow, "client_id": self.client_id}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/prompt",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())["prompt_id"]

    def get_history(self, prompt_id: str) -> dict:
        url = f"{self.base_url}/history/{prompt_id}"
        with urllib.request.urlopen(url) as resp:
            return json.loads(resp.read())

    def get_image(self, filename: str, subfolder: str, folder_type: str) -> bytes:
        params = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": folder_type})
        with urllib.request.urlopen(f"{self.base_url}/view?{params}") as resp:
            return resp.read()

    def wait_for_prompt(self, prompt_id: str, timeout: int = 600) -> dict:
        """Poll history until prompt completes or timeout."""
        start = time.time()
        while time.time() - start < timeout:
            history = self.get_history(prompt_id)
            if prompt_id in history:
                return history[prompt_id]
            time.sleep(2)
        raise TimeoutError(f"Prompt {prompt_id} did not complete within {timeout}s")

    # ── high-level helpers ───────────────────────────────────────────────────

    def _patch_workflow(self, workflow: dict, params: dict) -> dict:
        """Inject params into workflow by matching node title or input key."""
        wf = copy.deepcopy(workflow)
        # Simple strategy: nodes carry a _meta.title; patch by scanning all node inputs.
        # For production, use node IDs from the workflow JSON.
        for node_id, node in wf.items():
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs", {})
            for key, value in params.items():
                if key in inputs:
                    inputs[key] = value
        return wf

    def run_workflow(
        self,
        workflow_path: str | Path,
        params: dict = None,
        output_dir: str | Path = "outputs",
        timeout: int = 900,
    ) -> dict:
        """Load a workflow JSON, patch params, submit, wait, download outputs."""
        wf_path = Path(workflow_path)
        workflow = json.loads(wf_path.read_text())
        if params:
            workflow = self._patch_workflow(workflow, params)

        prompt_id = self.queue_prompt(workflow)
        print(f"[ComfyClient] queued {prompt_id} from {wf_path.name}")

        history = self.wait_for_prompt(prompt_id, timeout=timeout)
        outputs = history.get("outputs", {})

        saved_paths = []
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for node_output in outputs.values():
            for image_info in node_output.get("images", []):
                filename = image_info["filename"]
                subfolder = image_info.get("subfolder", "")
                ftype = image_info.get("type", "output")
                data = self.get_image(filename, subfolder, ftype)
                dest = out_dir / filename
                dest.write_bytes(data)
                saved_paths.append(str(dest))
            for video_info in node_output.get("videos", []):
                filename = video_info["filename"]
                subfolder = video_info.get("subfolder", "")
                ftype = video_info.get("type", "output")
                data = self.get_image(filename, subfolder, ftype)
                dest = out_dir / filename
                dest.write_bytes(data)
                saved_paths.append(str(dest))

        result = {
            "prompt_id": prompt_id,
            "paths": saved_paths,
            "path": saved_paths[0] if saved_paths else None,
        }
        print(f"[ComfyClient] done — {len(saved_paths)} file(s): {saved_paths}")
        return result

    def run_chain(
        self,
        workflow_path: str | Path,
        frames: list[str | Path],
        base_params: dict = None,
        output_dir: str | Path = "outputs",
    ) -> list[dict]:
        """Run workflow for consecutive frame pairs (clip chaining for long videos)."""
        results = []
        for i in range(len(frames) - 1):
            params = dict(base_params or {})
            params["start_frame_path"] = str(frames[i])
            params["end_frame_path"] = str(frames[i + 1])
            result = self.run_workflow(workflow_path, params=params, output_dir=output_dir)
            results.append(result)
        return results
