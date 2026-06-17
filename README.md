# Zero-Shot Image Segmentation

A click-to-segment image app powered by Meta's [Segment Anything Model (SAM)](https://huggingface.co/facebook/sam-vit-base).
Upload any image and click on any object to instantly highlight it with a segmentation mask —
no labels, no training, no predefined categories needed.

---

## Demo

![App demo](assets/demo.gif)

---

## Features

- Click anywhere on an image to segment that object instantly
- Works on any object in any image without prior training
- Confidence score displayed for each segmentation
- Click a different point to re-segment without re-uploading
- Runs 100% locally

---

## Setup

**1. Clone the repo**

```bash
git clone https://github.com/aminzafar/image-segmentation.git
cd image-segmentation
```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the app**

```bash
python app.py
# Opens automatically at http://127.0.0.1:7860
# SAM base model (~375MB) downloads automatically on first run
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| [SAM (facebook/sam-vit-base)](https://huggingface.co/facebook/sam-vit-base) | Zero-shot image segmentation |
| [HuggingFace Transformers](https://huggingface.co/docs/transformers) | Model loading and inference |
| [Gradio](https://gradio.app/) | Web UI with click detection |
| [Pillow](https://pillow.readthedocs.io/) / NumPy | Mask overlay rendering |
| Python 3.11 | Core language |

---

Built by [Amin Zafar](https://github.com/aminzafar)