# Interest-Conditioned Paper Summarization
## Does Personalization Hurt Faithfulness?

---

## 1. Original Source

This project is original work. The research question and experimental design are
inspired by:

> Zhongxiang Sun et al. (2024). *When Personalization Misleads: Understanding and
> Mitigating Hallucinations in Personalized LLMs.*
> https://arxiv.org/abs/2404.01920

ArXiv paper data is fetched live via the public ArXiv API:
https://export.arxiv.org/api/query

AlignScore evaluation library:
https://github.com/yuh-zha/AlignScore

---

## 2. Files Modified

### `summarization_experiment.ipynb`
The main experiment notebook. All analysis, generation, and evaluation is run here.
Key sections:
- **Section 1–2**: Setup, imports, profile definitions
- **Section 3**: `fetch_arxiv()` — paper collection from ArXiv API
- **Section 4**: Summary generation (`generate_conditioned_summaries`,
  `generate_baseline_summaries`, `generate_irrelevant_summaries`,
  `generate_prompt_level_summaries`, `generate_multi_model_summaries`)
- **Section 5**: Evaluation metrics (BERTScore, cosine distance, profile
  alignment, LLM pairwise judge via `pairwise_judge` /
  `run_pairwise_comparisons`)
- **Section 6**: Extended evaluation (BERTScore by prompt level / model /
  domain, AlignScore, statistical significance tests)

### `utils.py`
Helper functions and constants used throughout the notebook.
Functions defined here:
- `fetch_arxiv(categories, per_cat)` — ArXiv paper fetching
- `build_prompt(profile, abstract)` — personalized summarization prompt
- `build_prompt_level(profile, abstract, level)` — prompt at a given intensity
- `PROMPT_LEVELS` — dict of mild / moderate / aggressive prompt templates
- `load_or_score(data_df, scorer, csv_path)` — AlignScore caching helper
- `compute_cosine_distances(df, encoder)` — pairwise cosine distance

---

## 3. How to Run

### Prerequisites
1. Set your OpenAI API key in cell 2 of the notebook (replace the placeholder).
2. For local models (llama3.2:3b, mistral:7b), install and start Ollama:
   ```
   ollama serve
   ollama pull llama3.2:3b
   ollama pull mistral:7b
   ```

### Run the full experiment
Open `summarization_experiment.ipynb` in Jupyter and run all cells top to bottom:
```
jupyter notebook summarization_experiment.ipynb
```

Or in VS Code / JupyterLab, use "Run All Cells".

### Run order
The notebook is self-contained and sequential. Each section caches results to
`results/` as CSV files so expensive cells (API calls, AlignScore) are skipped on
re-runs. To regenerate any section, delete the corresponding CSV:

| Section | Cache file |
|---------|-----------|
| Main summaries | `results/summaries_raw.csv` |
| Baseline summaries | `results/baseline_summaries.csv` |
| Irrelevant summaries | `results/irrelevant_summaries.csv` |
| Prompt-level summaries | `results/prompt_level_summaries.csv` |
| Multi-model summaries | `results/multi_model_summaries.csv` |
| Pairwise judge | `results/llm_judge_pairwise.csv` |
| AlignScore (conditioned) | `results/alignscore_conditioned.csv` |
| AlignScore (baseline) | `results/alignscore_baseline.csv` |
| AlignScore (prompt) | `results/alignscore_prompt.csv` |
| AlignScore (model) | `results/alignscore_model.csv` |
| AlignScore (irrelevant) | `results/alignscore_irrelevant.csv` |

---

## 4. Models and Training Data

**No models were trained.** All language models are used via API or local inference only.

| Model | Access | Notes |
|-------|--------|-------|
| gpt-4o | OpenAI API | Requires API key |
| gpt-4o-mini | OpenAI API | Default model for main experiments |
| gpt-3.5-turbo | OpenAI API | Requires API key |
| llama3.2:3b | Ollama (local) | Free, requires Ollama installed |
| mistral:7b | Ollama (local) | Free, requires Ollama installed |

**AlignScore** uses a pre-trained RoBERTa-base checkpoint downloaded automatically
on first run (~400 MB):
- Checkpoint: `AlignScore/AlignScore-base.ckpt`

---

## 5. Prompts

All prompts used in the experiment are documented in `prompts.txt`.

There are four prompt types:
1. **Conditioned summarization** — standard profile-personalized summary
2. **Mild / Moderate / Aggressive** — three intensity variants for the prompt
   strength experiment
3. **Baseline summarization** — no profile, used as the control group
4. **Pairwise judge** — asks GPT to compare two summaries for a given reader

---

## 6. Software Requirements

```
Python              3.10
openai              >= 1.0.0
bert-score          >= 0.3.13
sentence-transformers >= 2.2.2
scikit-learn        >= 1.3.0
pandas              >= 2.0.0
numpy               >= 1.24.0
matplotlib          >= 3.7.0
seaborn             >= 0.12.0
scipy               >= 1.11.0
requests            >= 2.31.0
torch               >= 2.0.0
pytorch-lightning   >= 1.9.0
transformers        >= 4.30.0
nltk                >= 3.8.0
```

**AlignScore** must be installed from source:
```
git clone https://github.com/yuh-zha/AlignScore.git
pip install -e AlignScore
python -m nltk.downloader punkt punkt_tab
```

**Ollama** (optional, for local models):
https://ollama.com/download
