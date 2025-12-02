# MomaskImplementation

This repository contains an implementation of the MoMask motion generation/inference codebase adapted from the original project and packaged with a minimal backend and UI so it can be run locally and deployed to a Hugging Face Space.

Use this repo to run inference locally, test the backend, and deploy a demo on Hugging Face Spaces.

**Get You Ready**
- **Source repo:** `https://github.com/EricGuo5513/momask-codes` (original implementation referenced)
- **Deployed demo:** `https://huggingface.co/spaces/Hemanth0004/Momask-Demo-Project`

**Requirements**
- **Python / PyTorch tested:** Python 3.7.13 with PyTorch 1.7.1 (Conda route), or Python 3.10 for the pip route.

**Installation (Conda, recommended)**
- Create the environment from the provided `environment.yml`:

```
conda env create -f environment.yml
conda activate momask
```

- Install OpenAI CLIP from source (required by the project):

```
pip install git+https://github.com/openai/CLIP.git
```

Notes:
- We tested the conda-based environment with Python 3.7.13 and PyTorch 1.7.1. If you use a newer Python/PyTorch combo you may encounter dependency differences.

**Alternative: pip installation**
- If you cannot use conda, install with pip using the provided `requirements.txt`:

```
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

- This pip route was tested on Python 3.10.

**Quick Tests / Inference**
- Run a backend test inference using the script `gen_t2m.py`:

```
python gen_t2m.py --gpu_id 1 --ext exp1 --text_prompt "A person is running on a treadmill."
```

- The repo also includes a small web backend and UI. To run the full demo locally (end-to-end):

```
python app.py
```

Notes:
- Make sure model checkpoints are available under the `checkpoints/` directory before running inference. The exact checkpoint file and option flags may vary depending on which model variant you want to use (see the `checkpoints/` tree for available models).
- If you don't have a GPU, use `--gpu_id -1` or set the appropriate option in the code to run on CPU (expected to be very slow).

**Hugging Face Spaces deployment**
- The demo has been deployed at: `https://huggingface.co/spaces/Hemanth0004/Momask-Demo-Project`.
- To deploy your own Space, create a new Space and push this repo (or a minimal subset) to it. Ensure the `requirements.txt` contains all dependencies and the Space's hardware supports GPU if you want real-time performance.

Deployment tips:
- Keep large checkpoints out of the Git tree; instead host them in a model hub or storage and download during Space startup.
- Adjust `app.py` or the Space start command to load checkpoints from an accessible path or an environment variable.

**Troubleshooting**
- If you see CUDA/PyTorch version errors, verify the local CUDA toolkit and PyTorch build are compatible.
- If `CLIP` is not found after pip install, ensure your active Python environment is the same one used to run scripts (`which python` / `Get-Command python` on Windows).
- For missing model files, check `checkpoints/` and confirm `opt.txt` paths and `model/` files are present.

**Attribution**
- This project is adapted from `https://github.com/EricGuo5513/momask-codes`. Please cite and follow the original project's license and attribution requirements when sharing or publishing results.

**Contact / Next steps**
- If you'd like, I can:
	- Add instructions to automatically download recommended checkpoints at first run.
	- Create a minimal `app`/`Space` launcher that streams smaller test models so Spaces can demonstrate functionality without large uploads.


---
title: Momask Demo Project
emoji: ⚡
colorFrom: purple
colorTo: gray
sdk: gradio
sdk_version: 6.0.1
app_file: app.py
pinned: false
license: mit
short_description: Text-to-Motion Generation using MoMask (HumanML3D)
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
