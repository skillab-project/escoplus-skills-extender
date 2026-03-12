from fastapi import FastAPI, APIRouter, Query
from fastapi.responses import StreamingResponse
from pathlib import Path
from typing import Optional
import json, math, re, time, os, requests
import numpy as np
import datetime


class _NumpyEncoder(json.JSONEncoder):
    """Converts numpy int64/float64/bool_ to native Python types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _safe_json_dumps(obj) -> str:
    return json.dumps(obj, cls=_NumpyEncoder, ensure_ascii=False, indent=4)


def _sanitize(obj):
    """Recursively convert numpy scalars to native Python types."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
import pandas as pd
import networkx as nx
from dotenv import load_dotenv
from collections import defaultdict
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import traceback

# ============================================================
#  LOAD & VALIDATE ENVIRONMENT
# ============================================================
load_dotenv()

_REQUIRED_ENV = ["TRACKER_API", "TRACKER_USERNAME", "TRACKER_PASSWORD"]
_missing = [k for k in _REQUIRED_ENV if not os.getenv(k)]
if _missing:
    raise RuntimeError(
        f"❌ Missing required environment variables: {_missing}\n"
        f"   Make sure your .env file contains all of: {_REQUIRED_ENV}"
    )

print("✅ Environment loaded:")
print(f"   TRACKER_API  = {os.getenv('TRACKER_API')}")
print(f"   TRACKER_USERNAME = {os.getenv('TRACKER_USERNAME')}")
print(f"   TRACKER_PASSWORD = {'*' * len(os.getenv('TRACKER_PASSWORD', ''))}")

app = FastAPI(title="SKILLAB ESCOPlus Skills Extender API",
              root_path="/escoplus-skills-extender")

analysis_router = APIRouter(prefix="/api/analysis", tags=["Analysis"])
forecast_router = APIRouter(prefix="/api/forecasting", tags=["Forecasting"])


# ============================================================
#  SHARED HELPERS
# ============================================================

def _get_analysis_state(file_path: Path):
    """
    Checks if analysis is completed, in progress, or available.
    Returns: (state_string, data_to_return)
    States: 'completed', 'busy', 'available'
    """
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                cached_data = json.loads(f.read())
            
            # Case 1: Finished
            if cached_data.get("status") == "completed":
                return "completed", cached_data.get("result")

            # Case 2: Running
            if cached_data.get("status") == "in_progress":
                started_at = datetime.datetime.fromisoformat(cached_data.get("started_at"))
                elapsed = (datetime.datetime.now() - started_at).total_seconds()
                
                # Timeout after 20 minutes (assume crash/restart)
                if elapsed < 1200: 
                    return "busy", {
                        "status": "processing",
                        "message": f"This analysis is already running (started {int(elapsed // 60)}m ago). Please wait.",
                        "estimated_completion": "Variable depending on API page count"
                    }
                else:
                    print(f"⚠️ Stale analysis detected (>20m). Overwriting...")
        except Exception as e:
            print(f"⚠️ Error reading status file: {e}")
            
    return "available", None

def _set_analysis_state(file_path: Path, status: str, result: dict = None):
    """Writes the current state to the JSON file."""
    data = {
        "status": status,
        "started_at": datetime.datetime.now().isoformat() if status == "in_progress" else None,
        "completed_at": datetime.datetime.now().isoformat() if status == "completed" else None,
        "result": result
    }
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(_safe_json_dumps(_sanitize(data)))


def _get_token() -> str:
    """Authenticate and return a fresh Bearer token. Reads all values from .env."""
    api_url = os.getenv("TRACKER_API")
    res = requests.post(
        f"{api_url}/login",
        json={"username": os.getenv("TRACKER_USERNAME"), "password": os.getenv("TRACKER_PASSWORD")},
        timeout=15
    )
    res.raise_for_status()
    return res.text.replace('"', "")


def _ensure_cache_folder() -> Path:
    folder = Path("Completed_Analyses")
    if not folder.exists():
        folder.mkdir(parents=True)
        print(f"📁 Folder '{folder}' created.")
    else:
        print(f"📁 Folder '{folder}' already exists.")
    return folder


def _batch_resolve_skills(headers: dict, unique_uris: list) -> dict:
    """Resolve only the URIs actually found in the data — 50 per batch. Reads API URL from env."""
    api_url = os.getenv("TRACKER_API")
    id_to_label = {}
    if not unique_uris:
        return id_to_label
    batch_size = 50
    total_batches = math.ceil(len(unique_uris) / batch_size)
    print(f"📚 Resolving {len(unique_uris)} unique skill URIs in {total_batches} batches...")
    for batch_num, start in enumerate(range(0, len(unique_uris), batch_size), 1):
        batch = unique_uris[start:start + batch_size]
        skill_payload = [("ids", sid) for sid in batch]
        try:
            r = requests.post(f"{api_url}/skills", headers=headers, data=skill_payload, timeout=60)
            r.raise_for_status()
            for s in r.json().get("items", []):
                sid = s.get("id", "")
                if sid:
                    id_to_label[sid] = s.get("label", sid).strip().lower()
            print(f"   Batch {batch_num}/{total_batches}: resolved so far: {len(id_to_label)}")
        except Exception as e:
            print(f"   ⚠️ Batch {batch_num} failed: {e}")
    matched = sum(1 for u in unique_uris if u in id_to_label)
    print(f"🔗 Matched: {matched}/{len(unique_uris)} URIs")
    return id_to_label


def _auto_paginate(endpoint: str, headers: dict, form_builder_fn,
                   page_size: int = 100, timeout: int = 180,
                   max_retries: int = 3, backoff: int = 10) -> tuple:
    """
    Probe page 1 to get total count, then fetch all pages with retry.
    Returns (items_list, total_count).
    """
    def fetch_page(page_num):
        url = f"{os.getenv('TRACKER_API')}/{endpoint}?page={page_num}&page_size={page_size}"
        for attempt in range(1, max_retries + 1):
            try:
                print(f"   ↪ Attempt {attempt}/{max_retries} for page {page_num}...")
                r = requests.post(url, headers=headers, data=form_builder_fn(), timeout=timeout)
                if r.status_code != 200:
                    print(f"   ⚠️ HTTP {r.status_code}: {r.text[:200]}")
                    return {}
                return r.json()
            except requests.exceptions.ReadTimeout:
                print(f"   ⏱️ Timeout page {page_num}, attempt {attempt}.")
                if attempt < max_retries:
                    time.sleep(backoff)
                else:
                    return {}
            except Exception as ex:
                print(f"   ❌ {type(ex).__name__}: {ex}")
                return {}

    print(f"🔍 Probing page 1 of /{endpoint}...")
    probe = fetch_page(1)
    if not probe:
        return [], 0

    total_count = probe.get("count", 0)
    total_pages = math.ceil(total_count / page_size) if total_count > 0 else 1
    print(f"📊 Total: {total_count} records → {total_pages} page(s)")

    all_items = list(probe.get("items", []))
    print(f"📦 Page 1/{total_pages}: {len(all_items)} items")

    for page in range(2, total_pages + 1):
        print(f"📄 Fetching page {page}/{total_pages}...")
        data = fetch_page(page)
        if not data:
            print(f"⚠️ Page {page} failed — stopping early.")
            break
        items = data.get("items", [])
        print(f"📦 Page {page}/{total_pages}: {len(items)} items (running total: {len(all_items) + len(items)})")
        if not items:
            break
        all_items.extend(items)
        if len(items) < page_size:
            print("✅ Last page reached.")
            break

    print(f"🎯 Retrieved: {len(all_items)} / {total_count}")
    return all_items, total_count


AI_SKILLS_EXTENDED = [
    # === Foundational AI/ML ===
    "machine learning", "deep learning", "neural networks", "artificial intelligence",
    "supervised learning", "unsupervised learning", "reinforcement learning",
    "semi-supervised learning", "self-supervised learning", "transfer learning",
    "few-shot learning", "zero-shot learning", "meta-learning", "federated learning",
    "continual learning", "online learning", "active learning", "curriculum learning",
    # === Large Language Models & Generative AI ===
    "large language models", "llm", "gpt", "gpt-4", "gpt-4o", "claude", "gemini",
    "llama", "mistral", "falcon", "phi", "qwen", "chatgpt", "openai api",
    "prompt engineering", "prompt tuning", "instruction tuning", "rlhf",
    "retrieval augmented generation", "rag", "chain of thought", "few-shot prompting",
    "agentic ai", "ai agents", "autonomous agents", "multi-agent systems",
    "langchain", "langgraph", "llamaindex", "autogen", "crewai",
    "vector databases", "embedding models", "sentence transformers",
    "semantic search", "hybrid search",
    # === Model Architectures ===
    "transformers", "attention mechanisms", "bert", "roberta", "t5", "gpt-2",
    "convolutional neural networks", "cnn", "recurrent neural networks", "rnn",
    "lstm", "gru", "graph neural networks", "gnn", "diffusion models",
    "variational autoencoders", "vae", "generative adversarial networks", "gan",
    "stable diffusion", "dall-e", "midjourney", "controlnet",
    "mixture of experts", "moe", "sparse transformers", "state space models",
    # === MLOps & Infrastructure ===
    "mlops", "model deployment", "model serving", "model monitoring",
    "model versioning", "experiment tracking", "mlflow", "weights & biases",
    "kubeflow", "airflow", "feature stores", "data pipelines for ml",
    "model compression", "quantization", "pruning", "knowledge distillation",
    "onnx", "tensorrt", "triton inference server", "bentoml", "ray serve",
    # === AI Frameworks ===
    "pytorch", "tensorflow", "keras", "jax", "hugging face", "transformers library",
    "scikit-learn", "xgboost", "lightgbm", "catboost", "fastai",
    "opencv", "spacy", "nltk", "gensim", "detectron2",
    # === NLP ===
    "natural language processing", "nlp", "text classification", "named entity recognition",
    "sentiment analysis", "text summarization", "machine translation",
    "question answering", "information extraction", "coreference resolution",
    "dependency parsing", "pos tagging", "tokenization", "word embeddings",
    "word2vec", "glove", "fasttext", "text generation", "dialogue systems",
    "natural language understanding", "natural language generation",
    # === Computer Vision ===
    "computer vision", "object detection", "image segmentation", "image classification",
    "image generation", "video analysis", "pose estimation", "optical character recognition",
    "ocr", "face recognition", "depth estimation", "3d reconstruction",
    "point cloud processing", "medical image analysis", "satellite image analysis",
    # === Data Science & Analytics ===
    "data science", "data analysis", "exploratory data analysis", "feature engineering",
    "feature selection", "dimensionality reduction", "pca", "t-sne", "umap",
    "clustering", "k-means", "dbscan", "hierarchical clustering",
    "anomaly detection", "time series forecasting", "causal inference",
    "bayesian inference", "probabilistic programming", "statistical modelling",
    # === AI Ethics & Safety ===
    "ai ethics", "responsible ai", "ai safety", "ai alignment", "bias detection",
    "fairness in ml", "explainable ai", "xai", "interpretable ml",
    "shap", "lime", "counterfactual explanations", "model cards",
    "ai governance", "ai regulation", "gdpr compliance for ai",
    # === Emerging AI ===
    "multimodal ai", "vision language models", "vlm", "gpt-4 vision",
    "speech recognition", "text to speech", "audio generation",
    "music generation", "code generation", "ai for code", "github copilot",
    "ai pair programming", "neuro-symbolic ai", "quantum machine learning",
    "ai for drug discovery", "ai for genomics", "digital twins ai",
    "ai in robotics", "embodied ai", "world models",
]

GREEN_SKILLS_EXTENDED = [
    # === Core Sustainability ===
    "sustainability", "sustainable development", "circular economy", "green economy",
    "environmental management", "ecological footprint", "carbon footprint",
    "life cycle assessment", "lca", "environmental impact assessment",
    "sustainability reporting", "esg reporting", "gri standards",
    "sustainable finance", "green bonds", "impact investing",
    # === Climate & Energy ===
    "climate change mitigation", "climate change adaptation", "net zero",
    "carbon neutrality", "carbon offsetting", "carbon capture",
    "greenhouse gas emissions", "scope 1 2 3 emissions", "emissions trading",
    "renewable energy", "solar energy", "wind energy", "offshore wind",
    "hydropower", "geothermal energy", "tidal energy", "biomass energy",
    "energy storage", "battery technology", "hydrogen energy", "green hydrogen",
    "energy efficiency", "building energy efficiency", "smart grids",
    "demand response", "power purchase agreements", "ppa",
    # === Circular Economy & Waste ===
    "waste management", "recycling", "upcycling", "material recovery",
    "industrial symbiosis", "product lifecycle management",
    "eco-design", "design for disassembly", "cradle to cradle",
    "plastic reduction", "zero waste", "composting", "bioeconomy",
    # === Biodiversity & Land Use ===
    "biodiversity conservation", "ecosystem services", "nature-based solutions",
    "reforestation", "afforestation", "sustainable land management",
    "soil health", "regenerative agriculture", "agroecology",
    "precision agriculture", "sustainable farming", "organic farming",
    "water management", "water conservation", "watershed management",
    "marine conservation", "blue economy",
    # === Green Building & Cities ===
    "green building", "leed certification", "breeam", "passive house",
    "sustainable urban planning", "smart cities sustainability",
    "urban heat island mitigation", "green infrastructure",
    "sustainable transport", "electric vehicles", "ev charging infrastructure",
    "public transport decarbonisation", "cycling infrastructure",
    # === Supply Chain & Industry ===
    "sustainable supply chain", "green procurement", "sustainable sourcing",
    "corporate social responsibility", "csr", "supplier sustainability audits",
    "low carbon manufacturing", "industrial decarbonisation",
    "carbon border adjustment", "sustainable logistics",
    "environmental product declaration", "epd",
    # === Policy & Standards ===
    "eu green deal", "taxonomy regulation", "sfdr", "csrd",
    "paris agreement", "sdgs", "sustainable development goals",
    "environmental compliance", "iso 14001", "iso 50001",
    "science based targets", "sbti", "task force on climate disclosures", "tcfd",
    # === Green Data & Tech ===
    "green it", "sustainable computing", "energy efficient algorithms",
    "ai for sustainability", "climate data analysis", "carbon accounting software",
    "environmental monitoring systems", "remote sensing for environment",
    "iot for environmental monitoring", "digital sustainability",
]


def _load_non_esco_skills() -> tuple:
    """
    Load non-ESCO skills from THREE sources and return merged list + source map.
    Sources:
      1. Technology Skills CSV
      2. AI_SKILLS_EXTENDED (hardcoded)
      3. GREEN_SKILLS_EXTENDED (hardcoded)
    Returns:
      all_skills: sorted deduplicated list of all non-ESCO skill strings
      skill_source: dict mapping skill_string -> source label
    """
    skill_source = {}

    # 1. CSV
    csv_path = r"C:\Users\USER\PycharmProjects\pythonProject_pdf_parser\Technology Skills(Technology Skills) (1).csv"
    csv_skills = []
    try:
        tech_df = pd.read_csv(csv_path, sep=None, engine='python', on_bad_lines='skip')
        if "Example" not in tech_df.columns:
            print("⚠️ CSV missing 'Example' column — skipping CSV source.")
        else:
            for row in tech_df["Example"].dropna().astype(str):
                for s in row.replace(";", ",").split(","):
                    s = s.strip().lower()
                    if s:
                        csv_skills.append(s)
            print(f"📄 CSV: loaded {len(set(csv_skills))} technology skills.")
    except Exception as e:
        print(f"⚠️ Could not load CSV ({e}) — continuing without it.")

    for s in set(csv_skills):
        skill_source[s] = "technology_csv"

    # 2. AI skills
    for s in AI_SKILLS_EXTENDED:
        s = s.strip().lower()
        if s not in skill_source:
            skill_source[s] = "ai_extended"
        # if already from CSV, keep CSV label but note it's also AI
    print(f"🤖 AI extended: {len(AI_SKILLS_EXTENDED)} skills.")

    # 3. Green skills
    for s in GREEN_SKILLS_EXTENDED:
        s = s.strip().lower()
        if s not in skill_source:
            skill_source[s] = "green_extended"
    print(f"🌿 Green extended: {len(GREEN_SKILLS_EXTENDED)} skills.")

    all_skills = sorted(skill_source.keys())
    print(f"✅ Total non-ESCO skill pool: {len(all_skills)} unique skills (CSV + AI + Green).")
    return all_skills, skill_source


def _load_tech_csv() -> list:
    """Backwards-compatible wrapper — returns merged skill list only."""
    skills, _ = _load_non_esco_skills()
    return skills


def _compute_similarity(esco_labels: list, non_esco_skills: list,
                         similarity_threshold: float, confidence_threshold: float,
                         freq_map: dict, skill_source: dict = None) -> tuple:
    """
    TF-IDF cosine similarity + confidence. Returns (matches, high_conf_skills).
    Each match is tagged with its source: technology_csv | ai_extended | green_extended.
    """
    from collections import Counter as _Counter
    print(f"🔍 TF-IDF similarity: {len(esco_labels)} ESCO x {len(non_esco_skills)} non-ESCO skills...")
    corpus = esco_labels + non_esco_skills
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4))
    tfidf = vec.fit_transform(corpus)
    sim = cosine_similarity(tfidf[:len(esco_labels)], tfidf[len(esco_labels):])

    matches = []
    for i, esco_skill in enumerate(esco_labels):
        j = int(np.argmax(sim[i]))
        sc = float(sim[i][j])
        if sc >= similarity_threshold:
            matched_skill = non_esco_skills[j]
            source = (skill_source or {}).get(matched_skill, "technology_csv")
            matches.append({
                "ESCO_skill": esco_skill,
                "non_ESCO_skill": matched_skill,
                "similarity": round(sc, 3),
                "source": source,
            })

    for m in matches:
        f = freq_map.get(m["ESCO_skill"], 1)
        m["confidence"] = round(float(m["similarity"] * (1 + np.log1p(f) / 10)), 3)

    high_conf = [m for m in matches if m["confidence"] >= confidence_threshold]
    src_counts = dict(_Counter(m["source"] for m in high_conf))
    print(f"✅ {len(matches)} matches -> {len(high_conf)} high-confidence | by source: {src_counts}")
    return matches, high_conf


def _build_network(high_conf_skills: list) -> tuple:
    """Build NetworkX graph and compute metrics. Returns (G, nodes, edges, stats)."""
    G = nx.Graph()
    for m in high_conf_skills:
        e, n = m["ESCO_skill"], m["non_ESCO_skill"]
        G.add_node(e, group="ESCO_skill")
        G.add_node(n, group="non_ESCO_skill")
        G.add_edge(e, n, similarity=m["similarity"], confidence=m["confidence"])

    if G.number_of_edges() > 0:
        raw_degree = {k: int(v) for k, v in dict(G.degree()).items()}
        degree_centrality = {k: round(float(v), 3) for k, v in nx.degree_centrality(G).items()}
        avg_similarity = float(np.mean([d["similarity"] for _, _, d in G.edges(data=True)]))
        clustering = round(float(nx.average_clustering(G)), 3)
        components = [len(c) for c in nx.connected_components(G)]
        largest_component = int(max(components))
    else:
        raw_degree = degree_centrality = {}
        avg_similarity = clustering = largest_component = 0

    nodes = [{"id": str(n), "group": G.nodes[n]["group"],
               "degree": int(raw_degree.get(n, 0)),
               "degree_centrality": float(degree_centrality.get(n, 0))} for n in G.nodes()]
    edges = [{"source": str(u), "target": str(v),
               "similarity": float(d.get("similarity", 0)),
               "confidence": float(d.get("confidence", 0))} for u, v, d in G.edges(data=True)]

    esco_deg = [nd["degree"] for nd in nodes if nd["group"] == "ESCO_skill"]
    non_deg = [nd["degree"] for nd in nodes if nd["group"] == "non_ESCO_skill"]

    stats = {
        "nodes": int(G.number_of_nodes()), "edges": int(G.number_of_edges()),
        "avg_similarity": round(avg_similarity, 3),
        "avg_clustering": clustering,
        "largest_component_size": largest_component,
        "avg_degree": round(float(np.mean(list(degree_centrality.values()))), 3) if degree_centrality else 0,
        "ESCO_avg_degree": round(float(np.mean(esco_deg)), 3) if esco_deg else 0,
        "non_ESCO_avg_degree": round(float(np.mean(non_deg)), 3) if non_deg else 0,
    }
    return G, nodes, edges, stats


def _explainability_metrics(high_conf_skills: list) -> dict:
    if not high_conf_skills:
        return {"avg_similarity": 0, "avg_confidence": 0,
                "similarity_distribution": {}, "confidence_distribution": {}}
    sims = [m["similarity"] for m in high_conf_skills]
    confs = [m["confidence"] for m in high_conf_skills]
    return {
        "avg_similarity": round(float(np.mean(sims)), 3),
        "avg_confidence": round(float(np.mean(confs)), 3),
        "similarity_distribution": {
            "0.6-0.7": sum(0.6 <= s < 0.7 for s in sims),
            "0.7-0.8": sum(0.7 <= s < 0.8 for s in sims),
            "0.8-1.0": sum(s >= 0.8 for s in sims),
        },
        "confidence_distribution": {
            "0.6-0.7": sum(0.6 <= c < 0.7 for c in confs),
            "0.7-0.8": sum(0.7 <= c < 0.8 for c in confs),
            "0.8-1.0": sum(c >= 0.8 for c in confs),
        }
    }


def _occ_code(occ: str) -> str:
    match = re.search(r'C\d+$', occ)
    return match.group(0) if match else occ.replace('/', '_').replace(':', '').replace('.', '')


# ============================================================
#  ANALYSIS: /law-policies_extend_esco  (unchanged logic)
# ============================================================

@analysis_router.get("/law-policies_extend_esco")
def law_policies_extend_esco(
    keywords: str = Query(..., description="Comma-separated keywords (e.g. data,ai,green)"),
    source: str = Query("eur_lex", description="Source of the policies"),
    similarity_threshold: float = Query(0.8),
    confidence_threshold: float = Query(0.6)
):
    try:
        print("🔐 Authenticating...")
        token = _get_token()
        headers = {"Authorization": f"Bearer {token}"}

        keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
        payload = {"keywords": keywords_list, "keywords_logic": "or", "sources": [source]}
        all_docs = []
        for page in range(1, 51):
            url = f"{os.getenv('TRACKER_API')}/law-policies?page={page}&page_size=100"
            res = requests.post(url, headers=headers, data=payload, timeout=60)
            if res.status_code != 200:
                break
            data = res.json()
            items = data.get("items", [])
            if not items:
                break
            all_docs.extend(items)
            if len(items) < 100:
                break
        print(f"📄 Retrieved {len(all_docs)} policy documents.")

        skill_uris = sorted(set(s for d in all_docs for s in d.get("skills", []) if isinstance(s, str) and s.startswith("http")))
        id_to_label = _batch_resolve_skills(headers, skill_uris)
        ESCO_skill_labels = sorted({id_to_label.get(u, u) for u in skill_uris})
        print(f"🧠 {len(ESCO_skill_labels)} unique ESCO skills mapped.")

        non_ESCO_skills, skill_source = _load_non_esco_skills()

        freq = defaultdict(int)
        for d in all_docs:
            for s in d.get("skills", []):
                if s in id_to_label:
                    freq[id_to_label[s]] += 1

        matches, high_conf = _compute_similarity(ESCO_skill_labels, non_ESCO_skills, similarity_threshold, confidence_threshold, freq, skill_source)
        proposed = sorted({m["non_ESCO_skill"] for m in high_conf})
        G, nodes, edges, net_stats = _build_network(high_conf)

        pd.DataFrame(high_conf).to_csv("ESCOplus_Extended_from_Policies.csv", index=False)
        print(f"💾 Saved {len(high_conf)} new ESCO+ skills.")

        return {
            "message": "✅ ESCOPlus taxonomy extended with network metrics.",
            "summary": {
                "Policies processed": len(all_docs),
                "Mapped ESCO skills": len(ESCO_skill_labels),
                "Non-ESCO skill pool (CSV+AI+Green)": len(non_ESCO_skills),
                "Matches found": len(matches),
                "Proposed ESCO+ extensions": len(high_conf)
            },
            "associations": [{"ESCO_skill": m["ESCO_skill"], "non_ESCO_skill": m["non_ESCO_skill"],
                               "similarity": m["similarity"], "confidence": m["confidence"]} for m in high_conf[:50]],
            "network": {"nodes": nodes, "edges": edges},
            "network_stats": net_stats
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


# ============================================================
#  ANALYSIS: /profiles_extend_esco  (unchanged logic)
# ============================================================

@analysis_router.get("/profiles_extend_esco")
def profiles_extend_esco(
    keywords: str = Query(...),
    source: str = Query(None),
    similarity_threshold: float = Query(0.8),
    confidence_threshold: float = Query(0.6),
    max_pages: int = Query(10)
):
    try:
        print("🔐 Authenticating...")
        token = _get_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"
        }

        keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
        all_profiles = []
        for page in range(1, max_pages + 1):
            form_data = [("keywords_logic", "or"), ("skill_ids_logic", "or")]
            for kw in keywords_list:
                form_data.append(("keywords", kw))
            if source:
                form_data.append(("sources", source))
            url = f"{os.getenv('TRACKER_API')}/profiles?page={page}&page_size=100"
            res = requests.post(url, headers=headers, data=form_data, timeout=90)
            if res.status_code != 200:
                break
            items = res.json().get("items", [])
            if not items:
                break
            all_profiles.extend(items)
            if len(items) < 100:
                break
        print(f"🎯 Total profiles: {len(all_profiles)}")
        if not all_profiles:
            return {"error": "No profiles found."}

        skill_uris = sorted(set(s for p in all_profiles for s in p.get("skills", []) if isinstance(s, str) and s.startswith("http")))
        id_to_label = _batch_resolve_skills(headers, skill_uris)
        ESCO_skill_labels = sorted({id_to_label.get(u, u) for u in skill_uris})

        non_ESCO_skills, skill_source = _load_non_esco_skills()

        freq = defaultdict(int)
        for p in all_profiles:
            for s in p.get("skills", []):
                if s in id_to_label:
                    freq[id_to_label[s]] += 1

        matches, high_conf = _compute_similarity(ESCO_skill_labels, non_ESCO_skills, similarity_threshold, confidence_threshold, freq, skill_source)
        proposed = sorted({m["non_ESCO_skill"] for m in high_conf})
        G, nodes, edges, net_stats = _build_network(high_conf)

        output_path = "ESCOplus_Extended_from_Profiles.csv"
        pd.DataFrame(high_conf).to_csv(output_path, index=False)

        return {
            "message": "✅ ESCOPlus taxonomy extended from profiles.",
            "summary": {
                "Profiles processed": len(all_profiles),
                "ESCO skills mapped": len(ESCO_skill_labels),
                "Non-ESCO skill pool (CSV+AI+Green)": len(non_ESCO_skills),
                "Matches found": len(matches),
                "High-confidence new skills": len(high_conf)
            },
            "proposed_extensions": proposed[:100],
            "explainability_metrics": _explainability_metrics(high_conf),
            "new_skills_preview": high_conf[:20],
            "output_file": output_path,
            "network": {"nodes": nodes, "edges": edges, "stats": net_stats}
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


# ============================================================
#  ANALYSIS: /jobs_ultra
#  Fetching logic mirrors /jobsd-forecast (cache, auto-pagination,
#  occupation_ids, retry). Analysis logic: similarity + confidence
#  + network metrics + explainability — unchanged.
# ============================================================

@analysis_router.get("/jobs_ultra")
def jobs_extend_esco_ultra(
    keywords: Optional[str] = Query(None, description="Comma-separated keywords (e.g. AI, data, software)"),
    occupation_ids: Optional[str] = Query(None, description="Comma-separated occupation IDs (e.g. http://data.europa.eu/esco/isco/C2153)"),
    source: Optional[str] = Query(None, description="Optional source filter (e.g. linkedin, indeed)"),
    min_upload_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    max_upload_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    similarity_threshold: float = Query(0.8, description="TF-IDF cosine similarity threshold"),
    confidence_threshold: float = Query(0.6, description="Confidence cutoff for adding new skills"),
):
    """
    Fetch ALL job postings (auto-paginated, retry, occupation_ids support),
    extend ESCO taxonomy via TF-IDF similarity against CSV + AI + Green skill pools,
    build a skill co-occurrence network, compute explainability metrics.
    Results cached in Completed_Analyses/.
    """
    try:

        # ================================================================
        # 📁 CACHE SETUP — same pattern as /jobsd-forecast
        # ================================================================
        folder = _ensure_cache_folder()
        keywords_list = [k.strip() for k in keywords.split(",") if k.strip()] if keywords else []
        occ_ids_list  = [o.strip() for o in occupation_ids.split(",") if o.strip()] if occupation_ids else []

        filename = "completed_analysis_jobs_ultra_esco"
        for kw in keywords_list:
            filename += f"_{kw}"
        for occ in occ_ids_list:
            filename += f"_{_occ_code(occ)}"
        if source:
            filename += f"_{source}"
        if min_upload_date:
            filename += f"_from{min_upload_date}"
        if max_upload_date:
            filename += f"_to{max_upload_date}"
        filename += f"_sim{similarity_threshold}_conf{confidence_threshold}.json"

        file_path = folder / filename
        print(f"🗂️  Cache path: {file_path}")

        # --- 1. Check Status ---
        state, data = _get_analysis_state(file_path)
        if state == "completed":
            print(f"✅ Cache hit — returning results.")
            return data
        if state == "busy":
            return data

        # --- 2. Create Lock ---
        print(f"🌐 Starting new analysis. Locking {file_path}")
        _set_analysis_state(file_path, "in_progress")

        # ================================================================
        # 1️⃣  AUTHENTICATE
        # ================================================================
        print("🌐 No cache found — running full analysis...")
        print("🔐 Authenticating with Tracker...")
        token = _get_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"
        }
        print("✅ Authenticated successfully.")
        print(f"📡 Keywords     : {keywords_list or '(none)'}")
        print(f"🏢 OccupationIDs: {occ_ids_list  or '(none)'}")
        print(f"🗃️  Source       : {source or '(none)'}")
        print(f"📅 Date range   : {min_upload_date or '*'} → {max_upload_date or '*'}")
        print(f"⚙️  Thresholds   : similarity={similarity_threshold}, confidence={confidence_threshold}")

        # ================================================================
        # 2️⃣  AUTO-PAGINATE ALL JOB PAGES WITH RETRY
        #     Identical to /jobsd-forecast
        # ================================================================
        def build_form():
            fd = [("keywords_logic", "or"), ("skill_ids_logic", "or"), ("occupation_ids_logic", "or")]
            for kw in keywords_list:
                fd.append(("keywords", kw))
            for occ in occ_ids_list:
                fd.append(("occupation_ids", occ))
            if source:
                fd.append(("sources", source))
            if min_upload_date:
                fd.append(("min_upload_date", min_upload_date))
            if max_upload_date:
                fd.append(("max_upload_date", max_upload_date))
            return fd

        all_jobs, total_count = _auto_paginate("jobs", headers, build_form)

        if not all_jobs:
            return {"error": "No job postings found for the given filters."}

        print(f"✅ {len(all_jobs)} / {total_count} total jobs retrieved.")

        # ================================================================
        # 3️⃣  EXTRACT SKILL URIs + BATCH RESOLVE TO LABELS
        # ================================================================
        print("🧩 Extracting ESCO skill URIs from jobs...")
        skill_uris = sorted(set(
            s for j in all_jobs
            for s in j.get("skills", [])
            if isinstance(s, str) and s.startswith("http")
        ))
        print(f"🔗 Found {len(skill_uris)} unique skill URIs.")

        id_to_label = _batch_resolve_skills(headers, skill_uris)
        ESCO_skill_labels = sorted({id_to_label.get(u, u) for u in skill_uris})
        print(f"🧠 {len(ESCO_skill_labels)} unique ESCO skill labels mapped from jobs.")

        if not ESCO_skill_labels:
            return {"error": "No ESCO skills could be resolved from the fetched jobs."}

        # ================================================================
        # 4️⃣  LOAD NON-ESCO SKILL POOL (CSV + AI + GREEN)
        # ================================================================
        non_ESCO_skills, skill_source = _load_non_esco_skills()
        print(f"📦 Non-ESCO skill pool: {len(non_ESCO_skills)} skills (CSV + AI extended + Green extended).")

        # ================================================================
        # 5️⃣  BUILD ESCO SKILL FREQUENCY MAP (confidence weighting)
        # ================================================================
        print("📊 Building ESCO skill frequency map across all jobs...")
        freq = defaultdict(int)
        for j in all_jobs:
            for s in j.get("skills", []):
                if s in id_to_label:
                    freq[id_to_label[s]] += 1

        top_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"   Top 10 ESCO skills by frequency: {top_freq}")

        # ================================================================
        # 6️⃣  COMPUTE TF-IDF SIMILARITY + CONFIDENCE
        # ================================================================
        matches, high_conf = _compute_similarity(
            ESCO_skill_labels, non_ESCO_skills,
            similarity_threshold, confidence_threshold,
            freq, skill_source
        )
        proposed = sorted({m["non_ESCO_skill"] for m in high_conf})

        print(f"🔬 {len(matches)} total matches above similarity threshold.")
        print(f"🚀 {len(high_conf)} high-confidence ESCO+ extensions proposed.")
        print(f"   Source breakdown: { {s: sum(1 for m in high_conf if m['source']==s) for s in {'technology_csv','ai_extended','green_extended'}} }")

        # ================================================================
        # 7️⃣  BUILD SKILL NETWORK (ESCO ↔ non-ESCO)
        # ================================================================
        print("🌐 Building skill extension network (ESCO ↔ non-ESCO)...")
        G, nodes, edges, net_stats = _build_network(high_conf)
        print(f"   Network: {net_stats['nodes']} nodes, {net_stats['edges']} edges, "
              f"avg_similarity={net_stats['avg_similarity']}, "
              f"largest_component={net_stats['largest_component_size']}")

        # ================================================================
        # 8️⃣  EXPLAINABILITY METRICS
        # ================================================================
        print("📈 Computing explainability metrics...")
        expl = _explainability_metrics(high_conf)
        print(f"   avg_similarity={expl['avg_similarity']}, avg_confidence={expl['avg_confidence']}")

        # ================================================================
        # 9️⃣  SAVE LOCAL CSV
        # ================================================================
        output_csv = "ESCOplus_Extended_from_Jobs.csv"
        pd.DataFrame(high_conf).to_csv(output_csv, index=False)
        print(f"💾 Extended taxonomy CSV saved → '{output_csv}' ({len(high_conf)} rows).")

        # ================================================================
        # 🔟  BUILD RESULT
        # ================================================================
        result = {
            "message": "✅ ESCOPlus taxonomy extended from job postings.",
            "filters_used": {
                "keywords":        keywords_list or None,
                "occupation_ids":  occ_ids_list  or None,
                "source":          source,
                "min_upload_date": min_upload_date,
                "max_upload_date": max_upload_date,
            },
            "summary": {
                "Job postings processed":          len(all_jobs),
                "Total jobs available":            total_count,
                "ESCO skills mapped":              len(ESCO_skill_labels),
                "Non-ESCO skill pool (CSV+AI+Green)": len(non_ESCO_skills),
                "Matches found":                   len(matches),
                "High-confidence new skills":      len(high_conf),
                "Source breakdown": {
                    "technology_csv":  sum(1 for m in high_conf if m["source"] == "technology_csv"),
                    "ai_extended":     sum(1 for m in high_conf if m["source"] == "ai_extended"),
                    "green_extended":  sum(1 for m in high_conf if m["source"] == "green_extended"),
                }
            },
            "proposed_extensions":    proposed[:100],
            "explainability_metrics": expl,
            "new_skills_preview":     high_conf[:20],
            "output_file":            output_csv,
            "network": {"nodes": nodes, "edges": edges, "stats": net_stats},
        }

        # ================================================================
        # 💾  CACHE
        # ================================================================
        _set_analysis_state(file_path, "completed", result)
        print(f"✅ Analysis finished and saved to cache.")
        return result

    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        traceback.print_exc()
        return {"error": str(e)}


# ============================================================
#  ANALYSIS: /courses_ultra  (unchanged logic)
# ============================================================

@analysis_router.get("/courses_ultra")
def courses_extend_esco(
    keywords: str = Query(...),
    source: str = Query(None, description="Optional source filter (e.g. Udacity, europass)"),
    similarity_threshold: float = Query(0.8),
    confidence_threshold: float = Query(0.6)
):
    try:
        print("🔐 Authenticating...")
        token = _get_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"
        }

        keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
        all_courses = []
        for page in range(1, 51):
            form_data = [("keywords_logic", "or"), ("skill_ids_logic", "or"), ("sources", source)]
            for kw in keywords_list:
                form_data.append(("keywords", kw))
            url = f"{os.getenv('TRACKER_API')}/courses?page={page}&page_size=100"
            res = requests.post(url, headers=headers, data=form_data, timeout=60)
            if res.status_code != 200:
                break
            items = res.json().get("items", [])
            if not items:
                break
            all_courses.extend(items)
            if len(items) < 100:
                break
        print(f"📄 Retrieved {len(all_courses)} courses.")
        if not all_courses:
            return {"error": "No courses found."}

        skill_uris = sorted(set(s for c in all_courses for s in c.get("skills", []) if isinstance(s, str) and s.startswith("http")))
        id_to_label = _batch_resolve_skills(headers, skill_uris)
        ESCO_skill_labels = sorted({id_to_label.get(u, u) for u in skill_uris})

        non_ESCO_skills, skill_source = _load_non_esco_skills()

        freq = defaultdict(int)
        for c in all_courses:
            for s in c.get("skills", []):
                if s in id_to_label:
                    freq[id_to_label[s]] += 1

        matches, high_conf = _compute_similarity(ESCO_skill_labels, non_ESCO_skills, similarity_threshold, confidence_threshold, freq, skill_source)
        proposed = sorted({m["non_ESCO_skill"] for m in high_conf})
        G, nodes, edges, net_stats = _build_network(high_conf)

        output_path = "ESCOplus_Extended_from_Courses.csv"
        pd.DataFrame(high_conf).to_csv(output_path, index=False)

        return {
            "message": "✅ ESCOPlus taxonomy extended from courses.",
            "summary": {
                "Courses processed": len(all_courses),
                "ESCO skills mapped": len(ESCO_skill_labels),
                "Non-ESCO skill pool (CSV+AI+Green)": len(non_ESCO_skills),
                "Matches found": len(matches),
                "High-confidence new skills": len(high_conf)
            },
            "proposed_extensions": proposed[:100],
            "new_skills_preview": high_conf[:20],
            "output_file": output_path,
            "network": {"nodes": nodes, "edges": edges, "stats": net_stats}
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


# ============================================================
#  FORECASTING: /profiles  (unchanged logic)
# ============================================================

@forecast_router.get("/profiles")
def profiles_link_prediction(
    keywords: str = Query(...),
    source: str = Query(None),
    similarity_threshold: float = Query(0.7),
    top_k: int = Query(30),
    method: str = Query("adamic_adar")
):
    try:
        print("🔐 Authenticating...")
        token = _get_token()
        headers = {"Authorization": f"Bearer {token}"}

        keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
        payload = {"keywords": keywords_list, "keywords_logic": "or"}
        if source:
            payload["sources"] = [source]

        all_profiles = []
        for page in range(1, 51):
            url = f"{os.getenv('TRACKER_API')}/profiles?page={page}&page_size=100"
            res = requests.post(url, headers=headers, data=payload, timeout=60)
            if res.status_code != 200:
                break
            items = res.json().get("items", [])
            if not items:
                break
            all_profiles.extend(items)
            if len(items) < 100:
                break
        print(f"📄 Retrieved {len(all_profiles)} profiles.")
        if not all_profiles:
            return {"error": "No profiles found."}

        skill_uris = sorted(set(s for d in all_profiles for s in d.get("skills", []) if isinstance(s, str) and s.startswith("http")))
        id_to_label = _batch_resolve_skills(headers, skill_uris)
        ESCO_skill_labels = sorted({id_to_label.get(u, u) for u in skill_uris})

        non_ESCO_skills, skill_source = _load_non_esco_skills()

        corpus = list(ESCO_skill_labels) + list(non_ESCO_skills)
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4))
        tfidf = vectorizer.fit_transform(corpus)
        sim_matrix = cosine_similarity(tfidf[:len(ESCO_skill_labels)], tfidf[len(ESCO_skill_labels):])

        G = nx.Graph()
        for i, esco_skill in enumerate(ESCO_skill_labels):
            for j, non_esco_skill in enumerate(non_ESCO_skills):
                sim = sim_matrix[i, j]
                if sim >= similarity_threshold:
                    G.add_edge(esco_skill, non_esco_skill, weight=sim)

        print(f"🕸️ Graph: {len(G.nodes())} nodes, {len(G.edges())} edges.")
        if G.number_of_edges() == 0:
            return {"error": "No edges found. Try lowering similarity_threshold."}

        if method == "adamic_adar":
            preds = nx.adamic_adar_index(G)
        elif method == "resource_allocation":
            preds = nx.resource_allocation_index(G)
        else:
            preds = nx.jaccard_coefficient(G)

        preds_sorted = sorted(preds, key=lambda x: x[2], reverse=True)[:top_k]
        candidate_links = []
        for u, v, score in preds_sorted:
            common = list(nx.common_neighbors(G, u, v))
            ws = np.mean([G[u][n]["weight"] * G[v][n]["weight"] for n in common]) if common else 0
            combined = round((score + ws) / 2, 3)
            emoji, level = ("🟢", "High confidence") if combined >= 0.8 else (("🟡", "Medium confidence") if combined >= 0.6 else ("🔴", "Low confidence"))
            candidate_links.append({"source": u, "target": v, "predicted_score": combined, "confidence_level": level, "emoji": emoji})

        summary_counts = {
            "high": sum(1 for c in candidate_links if c["predicted_score"] >= 0.8),
            "medium": sum(1 for c in candidate_links if 0.6 <= c["predicted_score"] < 0.8),
            "low": sum(1 for c in candidate_links if c["predicted_score"] < 0.6)
        }

        return {
            "message": "✅ ESCOPlus classical link prediction completed.",
            "summary": {
                "Profiles processed": len(all_profiles),
                "Mapped ESCO skills": len(ESCO_skill_labels),
                "Non-ESCO skill pool (CSV+AI+Green)": len(non_ESCO_skills),
                "Observed edges": len(G.edges()),
                "Predicted new links": len(candidate_links),
                "Method used": method,
                "Confidence distribution": summary_counts
            },
            "predicted_links": candidate_links
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


# ============================================================
#  FORECASTING: /jobsd-forecast
#  ✅ MODIFIED: occupation_ids filter + auto-pagination + cache
# ============================================================

@forecast_router.get("/jobsd-forecast")
def jobs_link_prediction(
    keywords: Optional[str] = Query(None, description="Comma-separated keywords"),
    occupation_ids: Optional[str] = Query(None, description="Comma-separated occupation IDs"),
    source: Optional[str] = Query(None),
    min_upload_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    max_upload_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    similarity_threshold: float = Query(0.7),
    top_k: int = Query(30),
    method: str = Query("adamic_adar"),
):
    """
    Predict new ESCO ↔ non-ESCO connections from job postings.
    Supports occupation_ids filtering, auto-paginates all pages with retry,
    results cached in Completed_Analyses/.
    """
    try:

        # === Cache ===
        folder = _ensure_cache_folder()
        keywords_list = [k.strip() for k in keywords.split(",") if k.strip()] if keywords else []
        occ_ids_list = [o.strip() for o in occupation_ids.split(",") if o.strip()] if occupation_ids else []

        filename = f"completed_analysis_jobs_forecast_{method}"
        for kw in keywords_list:
            filename += f"_{kw}"
        for occ in occ_ids_list:
            filename += f"_{_occ_code(occ)}"
        if source:
            filename += f"_{source}"
        if min_upload_date:
            filename += f"_from{min_upload_date}"
        if max_upload_date:
            filename += f"_to{max_upload_date}"
        filename += f"_sim{similarity_threshold}_topk{top_k}.json"

        file_path = folder / filename
        print(f"🗂️ Cache path: {file_path}")
        
        # --- 1. Check Status ---
        state, data = _get_analysis_state(file_path)
        if state == "completed":
            return data
        if state == "busy":
            return data

        # --- 2. Create Lock ---
        _set_analysis_state(file_path, "in_progress")

        print("🌐 No cache — running full analysis...")
        print("🔐 Authenticating...")
        token = _get_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"
        }
        print("✅ Authenticated.")
        print(f"📡 Keywords: {keywords_list or '(none)'} | OccupationIDs: {occ_ids_list or '(none)'}")

        def build_form():
            fd = [("keywords_logic", "or"), ("skill_ids_logic", "or"), ("occupation_ids_logic", "or")]
            for kw in keywords_list:
                fd.append(("keywords", kw))
            for occ in occ_ids_list:
                fd.append(("occupation_ids", occ))
            if source:
                fd.append(("sources", source))
            if min_upload_date:
                fd.append(("min_upload_date", min_upload_date))
            if max_upload_date:
                fd.append(("max_upload_date", max_upload_date))
            return fd

        all_jobs, total_count = _auto_paginate("jobs", headers, build_form)
        if not all_jobs:
            return {"error": "No job postings found for the given filters."}

        skill_uris = sorted(set(s for j in all_jobs for s in j.get("skills", []) if isinstance(s, str) and s.startswith("http")))
        id_to_label = _batch_resolve_skills(headers, skill_uris)
        ESCO_skill_labels = sorted({id_to_label.get(u, u) for u in skill_uris})
        print(f"🧠 {len(ESCO_skill_labels)} ESCO skills from jobs.")

        non_ESCO_skills, skill_source = _load_non_esco_skills()
        print(f"💾 {len(non_ESCO_skills)} non-ESCO skills.")

        corpus = list(ESCO_skill_labels) + list(non_ESCO_skills)
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4))
        tfidf = vectorizer.fit_transform(corpus)
        sim_matrix = cosine_similarity(tfidf[:len(ESCO_skill_labels)], tfidf[len(ESCO_skill_labels):])

        print(f"🕸️ Building similarity graph (threshold={similarity_threshold})...")
        G = nx.Graph()
        for i, esco_skill in enumerate(ESCO_skill_labels):
            for j, tech_skill in enumerate(non_ESCO_skills):
                sim = sim_matrix[i, j]
                if sim >= similarity_threshold:
                    G.add_edge(esco_skill, tech_skill, weight=sim)

        print(f"🕸️ Graph: {len(G.nodes())} nodes, {len(G.edges())} edges.")
        if G.number_of_edges() == 0:
            return {"error": "No edges found. Try lowering similarity_threshold."}

        print(f"🔮 Running link prediction: {method}")
        if method == "adamic_adar":
            preds = nx.adamic_adar_index(G)
        elif method == "resource_allocation":
            preds = nx.resource_allocation_index(G)
        else:
            preds = nx.jaccard_coefficient(G)

        preds_sorted = sorted(preds, key=lambda x: x[2], reverse=True)[:top_k]
        candidate_links = []
        for u, v, score in preds_sorted:
            common = list(nx.common_neighbors(G, u, v))
            ws = np.mean([G[u][n]["weight"] * G[v][n]["weight"] for n in common]) if common else 0
            combined = round((score + ws) / 2, 3)
            emoji, level = ("🟢", "High confidence") if combined >= 0.8 else (("🟡", "Medium confidence") if combined >= 0.6 else ("🔴", "Low confidence"))
            candidate_links.append({"source": u, "target": v, "predicted_score": combined, "confidence_level": level, "emoji": emoji})

        summary_counts = {
            "high": sum(1 for c in candidate_links if c["predicted_score"] >= 0.8),
            "medium": sum(1 for c in candidate_links if 0.6 <= c["predicted_score"] < 0.8),
            "low": sum(1 for c in candidate_links if c["predicted_score"] < 0.6)
        }

        result = {
            "message": "✅ ESCOPlus job-based link prediction completed.",
            "filters_used": {
                "keywords": keywords_list or None,
                "occupation_ids": occ_ids_list or None,
                "source": source,
                "min_upload_date": min_upload_date,
                "max_upload_date": max_upload_date,
            },
            "summary": {
                "Jobs processed": len(all_jobs),
                "Total jobs available": total_count,
                "Mapped ESCO skills": len(ESCO_skill_labels),
                "Non-ESCO skill pool (CSV+AI+Green)": len(non_ESCO_skills),
                "Observed edges": len(G.edges()),
                "Predicted new links": len(candidate_links),
                "Method used": method,
                "Confidence distribution": summary_counts
            },
            "predicted_links": candidate_links
        }

        _set_analysis_state(file_path, "completed", result)
        return result
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        traceback.print_exc()
        return {"error": str(e)}


# ============================================================
#  FORECASTING: /courses  (unchanged logic)
# ============================================================

@forecast_router.get("/courses")
def courses_link_prediction(
    keywords: str = Query(...),
    source: str = Query("coursera"),
    similarity_threshold: float = Query(0.7),
    top_k: int = Query(30),
    method: str = Query("adamic_adar")
):
    try:
        print("🔐 Authenticating...")
        token = _get_token()
        headers = {"Authorization": f"Bearer {token}"}

        keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
        payload = {"keywords": keywords_list, "keywords_logic": "or", "sources": [source]}
        all_courses = []
        for page in range(1, 51):
            url = f"{os.getenv('TRACKER_API')}/courses?page={page}&page_size=100"
            res = requests.post(url, headers=headers, data=payload, timeout=60)
            if res.status_code != 200:
                break
            items = res.json().get("items", [])
            if not items:
                break
            all_courses.extend(items)
            if len(items) < 100:
                break
        print(f"📄 Retrieved {len(all_courses)} courses.")
        if not all_courses:
            return {"error": "No courses found."}

        skill_uris = sorted(set(s for d in all_courses for s in d.get("skills", []) if isinstance(s, str) and s.startswith("http")))
        id_to_label = _batch_resolve_skills(headers, skill_uris)
        ESCO_skill_labels = sorted({id_to_label.get(u, u) for u in skill_uris})

        non_ESCO_skills, skill_source = _load_non_esco_skills()

        corpus = list(ESCO_skill_labels) + list(non_ESCO_skills)
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4))
        tfidf = vectorizer.fit_transform(corpus)
        sim_matrix = cosine_similarity(tfidf[:len(ESCO_skill_labels)], tfidf[len(ESCO_skill_labels):])

        G = nx.Graph()
        for i, esco_skill in enumerate(ESCO_skill_labels):
            for j, tech_skill in enumerate(non_ESCO_skills):
                sim = sim_matrix[i, j]
                if sim >= similarity_threshold:
                    G.add_edge(esco_skill, tech_skill, weight=sim)

        print(f"🕸️ Graph: {len(G.nodes())} nodes, {len(G.edges())} edges.")
        if G.number_of_edges() == 0:
            return {"error": "No edges found."}

        if method == "adamic_adar":
            preds = nx.adamic_adar_index(G)
        elif method == "resource_allocation":
            preds = nx.resource_allocation_index(G)
        else:
            preds = nx.jaccard_coefficient(G)

        preds_sorted = sorted(preds, key=lambda x: x[2], reverse=True)[:top_k]
        candidate_links = []
        for u, v, score in preds_sorted:
            common = list(nx.common_neighbors(G, u, v))
            ws = np.mean([G[u][n]["weight"] * G[v][n]["weight"] for n in common]) if common else 0
            combined = round((score + ws) / 2, 3)
            emoji, level = ("🟢", "High confidence") if combined >= 0.8 else (("🟡", "Medium confidence") if combined >= 0.6 else ("🔴", "Low confidence"))
            candidate_links.append({"source": u, "target": v, "predicted_score": combined, "confidence_level": level, "emoji": emoji})

        summary_counts = {
            "high": sum(1 for c in candidate_links if c["predicted_score"] >= 0.8),
            "medium": sum(1 for c in candidate_links if 0.6 <= c["predicted_score"] < 0.8),
            "low": sum(1 for c in candidate_links if c["predicted_score"] < 0.6)
        }

        return {
            "message": "✅ ESCOPlus course-based link prediction completed.",
            "summary": {
                "Courses processed": len(all_courses), "Mapped ESCO skills": len(ESCO_skill_labels),
                "Non-ESCO skill pool (CSV+AI+Green)": len(non_ESCO_skills), "Observed edges": len(G.edges()),
                "Predicted new links": len(candidate_links), "Method used": method,
                "Confidence distribution": summary_counts
            },
            "predicted_links": candidate_links
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


# ============================================================
#  FORECASTING: /law_predict  (unchanged logic)
# ============================================================

@forecast_router.get("/law_predict")
def law_policies_link_prediction(
    keywords: str = Query(...),
    source: str = Query("eur_lex"),
    similarity_threshold: float = Query(0.7),
    top_k: int = Query(30),
    method: str = Query("jaccard")
):
    try:
        print("🔐 Authenticating...")
        token = _get_token()
        headers = {"Authorization": f"Bearer {token}"}

        keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
        payload = {"keywords": keywords_list, "keywords_logic": "or", "sources": [source]}
        all_docs = []
        for page in range(1, 51):
            url = f"{os.getenv('TRACKER_API')}/law-policies?page={page}&page_size=100"
            res = requests.post(url, headers=headers, data=payload, timeout=60)
            if res.status_code != 200:
                break
            items = res.json().get("items", [])
            if not items:
                break
            all_docs.extend(items)
            if len(items) < 100:
                break
        print(f"📄 Retrieved {len(all_docs)} policy documents.")

        skill_uris = sorted(set(s for d in all_docs for s in d.get("skills", []) if isinstance(s, str) and s.startswith("http")))
        id_to_label = _batch_resolve_skills(headers, skill_uris)
        ESCO_skill_labels = sorted({id_to_label.get(u, u) for u in skill_uris})
        print(f"🧠 {len(ESCO_skill_labels)} ESCO skills.")

        non_ESCO_skills, skill_source = _load_non_esco_skills()

        corpus = list(ESCO_skill_labels) + list(non_ESCO_skills)
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4))
        tfidf = vectorizer.fit_transform(corpus)
        sim_matrix = cosine_similarity(tfidf[:len(ESCO_skill_labels)], tfidf[len(ESCO_skill_labels):])

        G = nx.Graph()
        for i, esco in enumerate(ESCO_skill_labels):
            for j, non_esco in enumerate(non_ESCO_skills):
                sim = sim_matrix[i, j]
                if sim >= similarity_threshold:
                    G.add_edge(esco, non_esco, weight=sim)

        print(f"🕸️ Graph: {len(G.nodes())} nodes, {len(G.edges())} edges.")

        if method == "adamic_adar":
            preds = nx.adamic_adar_index(G)
        elif method == "resource_allocation":
            preds = nx.resource_allocation_index(G)
        else:
            preds = nx.jaccard_coefficient(G)

        adjusted_preds = []
        for u, v, score in preds:
            common = list(nx.common_neighbors(G, u, v))
            ws = np.mean([G[u][n]["weight"] * G[v][n]["weight"] for n in common if G.has_edge(u, n) and G.has_edge(v, n)]) if common else 0
            adjusted_preds.append((u, v, (score + ws) / 2))

        preds_sorted = sorted(adjusted_preds, key=lambda x: x[2], reverse=True)[:top_k]
        candidate_links = []
        for u, v, score in preds_sorted:
            s = round(score, 3)
            emoji, level = ("🟢", "High confidence") if s >= 0.8 else (("🟠", "Medium confidence") if s >= 0.6 else ("🔴", "Low confidence"))
            candidate_links.append({"source": u, "target": v, "predicted_score": s, "confidence_level": level, "emoji": emoji})

        print(f"🔗 {len(candidate_links)} predicted links.")

        return {
            "message": "✅ ESCOPlus classical link prediction completed.",
            "summary": {
                "Policies processed": len(all_docs), "Mapped ESCO skills": len(ESCO_skill_labels),
                "Non-ESCO skill pool (CSV+AI+Green)": len(non_ESCO_skills), "Observed edges": len(G.edges()),
                "Predicted new links": len(candidate_links), "Method used": method
            },
            "predicted_links": candidate_links
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


# ============================================================
#  REGISTER ROUTERS
# ============================================================
app.include_router(analysis_router)
app.include_router(forecast_router)
