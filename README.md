# Bias Detection and Mitigation in News Retrieval

This project explores how to detect bias in retrieved news evidence and reduce source bias when answering user questions with a retrieval-augmented generation (RAG) workflow.

The work is implemented in the notebook `bias_detection (2).ipynb` and uses the cleaned dataset `bias_clean.csv`.

## Project Goal

The goal is to:

- analyze a news dataset labeled by political leaning,
- segment article content into smaller retrieval chunks,
- embed those chunks with a sentence-transformer model,
- retrieve semantically relevant evidence for a query,
- detect the dominant source bias among retrieved documents,
- re-rank the evidence to reduce over-representation from any single bias group,
- build a balanced prompt for a language model or research assistant.

## Dataset

The project uses a cleaned dataset named `bias_clean.csv`.

Columns include:

- `url`
- `topic`
- `date`
- `title`
- `site`
- `bias`
- `page_text`

The dataset contains cleaned news article text with bias labels such as:

- `left`
- `leaning-left`
- `center`
- `leaning-right`
- `right`

After cleaning, the notebook reports approximately 10,832 usable documents.

## Pipeline Overview

The notebook follows this workflow:

1. Load and clean the dataset
   - drop missing `page_text` values
   - convert text to string format
   - remove duplicate article texts

2. Split long texts into chunks
   - chunk size: 150 words
   - overlap: 30 words
   - chunks shorter than 50 words are discarded

3. Generate semantic embeddings
   - uses `all-MiniLM-L6-v2` from `sentence-transformers`
   - normalizes embeddings for semantic similarity search

4. Build a FAISS vector index
   - uses cosine-like similarity via normalized inner-product search
   - retrieves the top-K relevant chunks for a query

5. Map retrieved chunks back to source bias labels
   - each chunk inherits the bias of the originating article

6. Detect source bias in the retrieved evidence
   - compute raw bias distribution
   - compute weighted bias distribution using retrieval similarity scores

7. Mitigate bias in evidence selection
   - assign a weight to each retrieved chunk based on similarity and source-bias dominance
   - re-rank evidence so that any one bias category does not dominate the answer
   - select a diverse set of evidence across bias categories

8. Build an LLM prompt
   - include the selected evidence in a balanced prompt
   - instruct the model to answer using only evidence and avoid unsupported or stereotyped claims

## Example Retrieval Workflow

The notebook uses a question such as:

`Are women underrepresented in AI?`

It then:

- converts the query into an embedding,
- searches the FAISS index for the most relevant text chunks,
- receives the top matching evidence,
- calculates which bias categories are overrepresented,
- re-ranks the evidence to produce a balanced answer context.

## Bias Mitigation Logic

The notebook applies a lightweight mitigation strategy:

- similarities are converted into non-negative weights
- weighted bias contribution is aggregated by bias class
- evidence is re-ranked using a mitigation score
- only a diverse, balanced subset of evidence is used in the final context

This helps avoid a situation where one ideological source category dominates the answer, even if it is highly ranked semantically.

## Required Libraries

The notebook uses:

- `pandas`
- `numpy`
- `torch`
- `sentence-transformers`
- `faiss-cpu`

Typical installation commands:

```bash
pip install pandas numpy torch sentence-transformers faiss-cpu
```

## Repository Contents

```text
Bias-Detection/
├── bias_clean.csv
├── bias_detection (2).ipynb
├── README.md
```

## How to Use

1. Open the notebook `bias_detection (2).ipynb` in Jupyter or Google Colab.
2. Run the cells in order.
3. Ensure `bias_clean.csv` is in the same working directory.
4. Change the query text in the retrieval section to ask your own questions.
5. Inspect the retrieved evidence and the final bias-aware prompt.

## Notes

This is a research-oriented notebook and proof-of-concept workflow rather than a production system. It demonstrates the concept of combining semantic retrieval, bias-aware evidence aggregation, and balanced answer generation for media bias analysis.

## Summary

This project combines:

- news text preprocessing,
- semantic search with embeddings,
- FAISS indexing,
- bias-label analysis,
- evidence re-ranking,
- and mitigation of source bias in answer generation.

It is a practical example of using retrieval-based techniques to make AI answers more balanced and evidence-aware.
