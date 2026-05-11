"""
Helper functions and constants shared across the notebook.
"""

import os
import time
import itertools
import requests
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ArXiv fetching
# Pull the most recent `per_cat` papers from each ArXiv category.
def fetch_arxiv(categories, per_cat):
    papers = {} 
    ns = "http://www.w3.org/2005/Atom"
    for cat in categories:
        print(f"Fetching {per_cat} papers from {cat}...")
        url = (
            f"https://export.arxiv.org/api/query"
            f"?search_query=cat:{cat}"
            f"&sortBy=submittedDate&sortOrder=descending"
            f"&max_results={per_cat}"
        )
        root = ET.fromstring(requests.get(url, timeout=20).text)
        for entry in root.findall(f"{{{ns}}}entry"):
            aid = entry.find(f"{{{ns}}}id").text.split("/abs/")[-1].strip()
            title = entry.find(f"{{{ns}}}title").text.replace("\n", " ").strip()
            abstract = entry.find(f"{{{ns}}}summary").text.replace("\n", " ").strip()
            authors = [a.find(f"{{{ns}}}name").text for a in entry.findall(f"{{{ns}}}author")]
            pub = entry.find(f"{{{ns}}}published").text[:10]
            if aid not in papers and len(abstract) > 100:
                papers[aid] = {
                    "arxiv_id": aid, "title": title, "abstract": abstract,
                    "authors": ", ".join(authors[:3]), "published": pub, "category": cat,
                }
        time.sleep(1)
    result = list(papers.values())
    print(f"\nCollected {len(result)} unique papers.")
    return result



# Prompt builders
# Standard personalized summarization prompt. 
# It lets us measure how much personalization affects factual accuracy.
def build_prompt(profile, abstract):
    return (
        f"You are summarizing an academic paper for a specific reader.\n\n"
        f"Reader profile: {profile['name']} — {profile['desc']}\n\n"
        f"Paper abstract:\n{abstract}\n\n"
        f"Write a 3-sentence summary of this paper tailored to the reader's background "
        f"and interests. Highlight aspects most relevant to their profile. Be specific and "
        f"faithful to what the paper actually says. Do not add information not in the abstract."
    )


# 3 intensity levels to test how much personalization pressure affects faithfulness
PROMPT_LEVELS = {
    "mild": (
        "Write a 3-sentence summary of this paper for a {name}. "
        "Be faithful to what the abstract says."
    ),
    "moderate": (
        "Write a 3-sentence summary of this paper tailored to a {name} — {desc}. "
        "Highlight aspects most relevant to their profile. "
        "Be faithful to what the paper actually says. Do not add information not in the abstract."
    ),
    "aggressive": (
        "Write a 3-sentence summary of this paper SPECIFICALLY for a {name} — {desc}. "
        "Heavily emphasize what matters most to this reader. "
        "Actively reinterpret the paper through their lens and background. "
        "Use language and framing that speaks directly to their expertise and concerns."
    ),
}

# Same as build_prompt but with a variable intensity instruction.
def build_prompt_level(profile, abstract, level):
    template    = PROMPT_LEVELS[level]
    instruction = template.format(name=profile["name"], desc=profile["desc"])
    return (
        f"You are summarizing an academic paper for a specific reader.\n\n"
        f"Reader profile: {profile['name']} — {profile['desc']}\n\n"
        f"Paper abstract:\n{abstract}\n\n"
        f"{instruction}"
    )



# AlignScore helper

# Load AlignScore results from CSV if available, otherwise compute and save.
def load_or_score(data_df, scorer, csv_path):
    if os.path.exists(csv_path):
        print(f"Loaded {csv_path}")
        return pd.read_csv(csv_path)
    print(f"Computing AlignScore -> {csv_path}...")
    result = data_df.copy()
    result["alignscore"] = scorer.score(
        contexts=result["abstract"].tolist(),
        claims=result["summary"].tolist()
    )
    result.to_csv(csv_path, index=False)
    print("Saved.")
    return result



# Cosine distance between profile summaries
# For each paper, compute pairwise cosine distance between all profile summaries.
# Larger distance = more different summaries
def compute_cosine_distances(df, encoder):
    records = []
    for arxiv_id, group in df.groupby("arxiv_id"):
        group = group[group["summary"].str.len() > 0]
        if len(group) < 2:
            continue
        embeddings = encoder.encode(group["summary"].tolist(), show_progress_bar=False)
        sim_matrix = cosine_similarity(embeddings)
        profiles = group["profile_id"].tolist()
        for (i, pi), (j, pj) in itertools.combinations(enumerate(profiles), 2):
            records.append({
                "arxiv_id": arxiv_id,
                "category": group["category"].iloc[0],
                "profile_a": pi,
                "profile_b": pj,
                "cosine_sim": float(sim_matrix[i, j]),
                "cosine_dist": float(1 - sim_matrix[i, j]),
            })
    return records
