# Local Semantic Image Search

A local web app for semantic image search using Python, Flask, and CLIP. It scans a folder of images, embeds them with a vision-language model, stores the embeddings on disk, and lets you search with natural-language queries like:

- `sunset over mountains`
- `moon in picture`
- `anime wallpaper`
- `city skyline at night`

The app runs locally and uses no cloud APIs. The CLIP model may be downloaded on first run, then reused from the local Hugging Face/Transformers cache.

---

## Features

- Recursive image folder scanning
- CLIP image and text embeddings via `torch` + `transformers`
- Persistent on-disk index:
  - `data/index.json`
  - `data/embeddings.npy`
- Incremental re-indexing:
  - skips unchanged images
  - embeds new/changed images
  - removes deleted images from the index
- Local Flask web UI
- Thumbnail result grid
- Filename, path, and similarity score display
- Similarity threshold filtering to hide low-confidence matches
- CPU by default

---

## Project structure

```text
.
├── app.py                  # Flask web backend and routes
├── config.py               # App configuration and environment variables
├── indexer.py              # Folder scanning and index creation/update
├── models.py               # CLIP model loading and embedding helpers
├── search.py               # Index loading and cosine similarity search
├── requirements.txt        # Python dependencies
├── TODO.md                 # Implementation checklist
├── README.md               # This file
├── image/		            # Contains a set of sample images
├── data/                   # Generated index files
│   ├── index.json
│   └── embeddings.npy
├── static/
│   └── styles.css          # Minimal web UI styling
└── templates/
    └── index.html          # Search form and result grid
```

---

## Requirements

- Python 3.10+
- Conda or virtualenv
- Enough disk space for the CLIP model cache
- CPU is supported by default

Python packages are listed in:

```text
requirements.txt
```

Core dependencies:

- Flask
- torch
- transformers
- Pillow
- NumPy

---

## Setup

### Option A: Conda environment

If you already have a conda environment named `venv`:

```bash
cd "/home/user/Documents/Image Search"
conda activate venv
pip install -r requirements.txt
```

If you need to create one:

```bash
cd "/home/user/Documents/Image Search"
conda create -n venv python=3.10
conda activate venv
pip install -r requirements.txt
```

### Option B: Python virtualenv

```bash
cd "/home/user/Documents/Image Search"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## Configuration

The app is configured through environment variables.

### Required: image folder

Set the folder containing images you want to search:

```bash
export IMAGE_SEARCH_FOLDERS="/path/to/your/images"
```

Example:

```bash
export IMAGE_SEARCH_FOLDERS="/home/user/Pictures"
```

You can provide multiple folders separated by `:`:

```bash
export IMAGE_SEARCH_FOLDERS="/home/user/Pictures:/home/user/Wallpapers"
```

Current note: the Flask app currently uses the first configured folder.

### Optional settings

```bash
export CLIP_MODEL_NAME="openai/clip-vit-base-patch32"
export IMAGE_SEARCH_DEVICE="cpu"
export IMAGE_SEARCH_DEFAULT_TOP_K="20"
export IMAGE_SEARCH_MAX_TOP_K="100"
export IMAGE_SEARCH_DEFAULT_MIN_SCORE="0.22"
export IMAGE_SEARCH_HOST="127.0.0.1"
export IMAGE_SEARCH_PORT="5000"
export IMAGE_SEARCH_DEBUG="0"
```

---

## First model download

The first time you run indexing or search, Transformers may download the CLIP model from Hugging Face:

```text
openai/clip-vit-base-patch32
```

After the model is cached locally, the app can run offline using the cached files.

If you change the model name, rebuild the index because image embeddings from one model are not compatible with text embeddings from another model.

---

## Build the image index

Activate your environment first:

```bash
cd "/home/user/Documents/Image Search"
conda activate venv
```

Set your image folder:

```bash
export IMAGE_SEARCH_FOLDERS="/path/to/your/images"
```

Build a fresh index:

```bash
python indexer.py "/path/to/your/images" "data" --rebuild
```

Example:

```bash
python indexer.py "/home/user/Pictures" "data" --rebuild
```

Update an existing index incrementally:

```bash
python indexer.py "/home/user/Pictures" "data"
```

The indexer creates:

```text
data/index.json
data/embeddings.npy
```

---

## Run the Flask app

```bash
cd "/home/user/Documents/Image Search"
conda activate venv
export IMAGE_SEARCH_FOLDERS="/path/to/your/images"
python app.py
```

Open your browser:

```text
http://127.0.0.1:5000
```

From the web UI you can:

1. Enter a natural-language query.
2. Set `Top K`.
3. Set `Min score` to filter weak matches.
4. Click `Search`.
5. Click `Re-index folder` to update the index.

---

## Example queries

Try queries that describe visual content:

```text
sunset over mountains
forest trail
ocean waves
night city skyline
terminal screenshot
code editor window
server rack in a data center
anime style wallpaper
abstract blue geometric background
photo of a cat
```

CLIP works better with descriptive phrases than single words.

Instead of:

```text
terminal
```

try:

```text
screenshot of a computer terminal window with text
```

Instead of:

```text
server
```

try:

```text
photo of server racks in a data center
```

---

## Similarity threshold tuning

The app includes a `Min score` field to hide low-confidence results.

Start with:

```text
0.22
```

Suggested ranges:

```text
0.18–0.21  loose, more results
0.22–0.26  moderate
0.27–0.30  strict
0.30+      very strict
```

If unrelated images appear, increase `Min score`.

If good images disappear, decrease it.

A useful workflow:

1. Search with `Min score = 0.00`.
2. Look at the scores for good and bad results.
3. Choose a threshold between them.

---

## Command-line search test

After building an index:

```bash
python -c "from search import load_index, search_text; load_index('data'); print(search_text('sunset over mountains', top_k=5, min_score=0.22))"
```

You should see a list of matches like:

```python
[
    {
        'id': 3,
        'path': '/home/user/Pictures/image.jpg',
        'filename': 'image.jpg',
        'score': 0.287,
        'metadata': {...}
    }
]
```

---

## Syntax check

Run:

```bash
python -m py_compile app.py config.py models.py indexer.py search.py
```

No output means the files compiled successfully.

---

## Troubleshooting

### No results appear

Possible causes:

- The index has not been built yet.
- The image folder path is wrong.
- `Min score` is too high.
- The query is too vague.

Try:

```text
Min score = 0.00
```

Then search again and inspect the scores.

### Results seem unrelated

CLIP always returns the closest images, even if none are truly relevant. Use the `Min score` threshold to suppress weak matches.

Also try more descriptive queries:

```text
photo of a mountain landscape at sunset
screenshot of a terminal with command line text
anime illustration of a character
```

### Model fails to load

Make sure dependencies are installed:

```bash
pip install -r requirements.txt
```

If running offline, make sure the model was downloaded at least once while online.

### Changed CLIP model but results are bad

Rebuild the index:

```bash
python indexer.py "/path/to/your/images" "data" --rebuild
```

Image embeddings and text embeddings must come from the same CLIP model.
