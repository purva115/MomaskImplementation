import os
import glob
import uuid
import subprocess

import gradio as gr

# Optional: set this if you want GPU on Spaces
# (We are on CPU Basic, so keep this False.)
USE_GPU = False


def generate_motion(prompt: str):
    """
    Runs MoMask's gen_t2m.py with a given text prompt and returns the path to the generated mp4.
    """
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a non-empty text prompt.")

    # Unique id for this run
    run_id = str(uuid.uuid4())[:8]

    # Build command
    gpu_flag = "-1"  # CPU only on Spaces
    cmd = [
        "python",
        "gen_t2m.py",
        "--gpu_id", gpu_flag,
        "--ext", run_id,
        "--text_prompt", prompt,
    ]

    # Running the generation
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise gr.Error(f"Generation failed. See logs. (Error: {e})")

    # ------------------------------------------------------------------
    # Find the generated mp4 (be very flexible about where it is)
    # ------------------------------------------------------------------

    # Find all mp4 files anywhere in the repo
    all_mp4s = glob.glob("**/*.mp4", recursive=True)

    # Prefer files whose path contains this run_id
    mp4_candidates = [f for f in all_mp4s if run_id in f]

    # If nothing has run_id, prefer files under "generation/"
    if not mp4_candidates:
        mp4_in_generation = [f for f in all_mp4s if "generation" in f]
        if mp4_in_generation:
            mp4_candidates = mp4_in_generation

    # If still nothing, fall back to any mp4 at all
    if not mp4_candidates and all_mp4s:
        mp4_candidates = all_mp4s

    # If we still have no mp4s, something went wrong in gen_t2m.py
    if not mp4_candidates:
        raise gr.Error("No MP4 file was generated. Please check server logs.")

    # Pick the most recently modified mp4 as the output
    mp4_path = max(mp4_candidates, key=os.path.getmtime)

    # Return the video path (Gradio will display it)
    return mp4_path


# Build the Gradio interface
demo = gr.Interface(
    fn=generate_motion,
    inputs=gr.Textbox(
        lines=2,
        label="Text prompt",
        placeholder="A person is walking forward.",
    ),
    outputs=gr.Video(label="Generated motion"),
    title="MoMask: Text-to-Motion Generation",
    description=(
        "Enter a natural language description (e.g., 'A person is walking forward.') "
        "and generate a 3D motion animation using MoMask."
    ),
)


if __name__ == "__main__":
    # For local testing; on Hugging Face Spaces they call `python app.py`
    demo.launch()
