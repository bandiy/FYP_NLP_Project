# FYP_NLP_Project — Code Overview & Run Instructions

This repository contains multiple experimental pipelines for phonetic, semantic and hybrid analyses. The code is organised by version folders (v1, v2, semanticsv2, analysisv2, v3_french_russian, v4). A chronological appendix listing scripts and recommended run order is provided in `docs/appendix_code_overview.md`. A CSV export is available at `docs/code_overview.csv`.

Important: run everything inside a Python 3.11 virtual environment.

Start:

1. Create and activate a virtual environment (Windows PowerShell):

```powershell
python3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Or for Command Prompt:

```cmd
python3.11 -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

Notes on heavy packages:
- `torch` may need a specific wheel for your platform/GPU. If CUDA build is needed, install `torch` following the instructions at https://pytorch.org/ before installing the remainder with `pip install -r requirements.txt`.
- After installing `spacy`, download the French model used by V3:

```powershell
python -m spacy download fr_core_news_md
```

- Some scripts call `nltk.download("stopwords")` at runtime; you can pre-download via:

```powershell
python -c "import nltk; nltk.download('stopwords')"
```

Files added by this helper:
- `docs/appendix_code_overview.md` — Markdown appendix with scripts grouped & ordered by version.
- `data/code_overview.csv` — CSV export of the same table.

To quickly set up run:
```powershell
python3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download fr_core_news_md
python -c "import nltk; nltk.download('stopwords')"
```

Recommended high level workflow

- Run data extraction & preprocessing scripts in each version folder first (see appendix for exact order).
- Run embedding-building scripts (`semanticsv2/build_embeddings.py`, `v1/embeddings/run_phonetic_embeddings.py`, etc.).
- Compute distance matrices / KNN graphs (`v2/phonetics/build_distance_matrix.py`, `v2/visualisation/build_knn_graph.py`).
- Run analysis & fusion experiments (`analysisv2/*`, `v3_french_russian/*`, `v4/*`) as needed.
