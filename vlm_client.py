"""
Wrapper around FastVLM inference.

FastVLM (apple/ml-fastvlm) isn't a pip package, so it's called via its
documented predict.py CLI entry point as a subprocess. If config.ENABLE_FASTVLM
is False (default), a stub description is returned instead — this lets you
run and test the whole pipeline (detection, tracking, zone, logging) before
downloading the ~GB-sized FastVLM checkpoints.
"""

import os
import subprocess

import config


def query_fastvlm(image_path: str, cls_name: str) -> str:
    if not config.ENABLE_FASTVLM:
        return f"[stub] appears to be a {cls_name}"

    try:
        result = subprocess.run(
            [
                "python", os.path.join(config.FASTVLM_REPO_PATH, "predict.py"),
                "--model-path", config.FASTVLM_MODEL_PATH,
                "--image-file", image_path,
                "--prompt", config.FASTVLM_PROMPT,
            ],
            capture_output=True, text=True, timeout=config.FASTVLM_TIMEOUT_SEC,
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            return f"[FastVLM error: {result.stderr.strip()[:150]}]"
        return output if output else "[FastVLM: no output]"
    except subprocess.TimeoutExpired:
        return "[FastVLM: timed out]"
    except FileNotFoundError:
        return "[FastVLM: predict.py not found — check FASTVLM_REPO_PATH]"
    except Exception as e:
        return f"[FastVLM error: {e}]"
