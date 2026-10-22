# Part 1 — Text Processing and Exploratory Data Analysis

Tag: `IRWA-2026-part-1`. Cloning the repository is covered in the [main README](../../README.md).

| file | content |
|---|---|
| `IRWA-2026-u231867-u231747-u233694-part-1.pdf` | report (LaTeX source: the `.tex` file with the same name) |
| `IRWA_Part1.ipynb` | notebook: data preparation (points 1–4) and exploratory data analysis |
| `irwa_preprocessing.py` | pre-processing functions, reused in the next parts to clean the queries the same way |
| `figures/` | plots produced by the notebook |

The notebook reads `data/fashion_products_dataset.json` and `data/validation_labels.csv`,
and writes `data/processed_corpus.pkl`.

## Reproduce the results

From the repository root:

1. Use Python 3.10 or newer. Create a virtual environment, install the requirements and download the spaCy model:
   ```bash
   python -m venv irwa_venv
   irwa_venv\Scripts\activate        # Windows
   source irwa_venv/bin/activate     # macOS / Linux
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
2. Copy `fashion_products_dataset.json` (from Aula Global) into the `data/` folder.
   It is not in the repository because of its size (66 MB).
3. Run the notebook from its folder:
   ```bash
   cd project_progress/part_1
   jupyter notebook IRWA_Part1.ipynb
   ```
   Select *Run → Run All Cells* (or run it without the browser:
   `jupyter nbconvert --to notebook --execute --inplace IRWA_Part1.ipynb`).
4. It takes a few minutes and produces:
   * `figures/*.png`: the plots used in the report;
   * `data/processed_corpus.pkl`: the processed corpus (one row per `pid`), used in the next parts.
