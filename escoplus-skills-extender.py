from fastapi import FastAPI, APIRouter
# === FILE: esco_extension_service.py ===
from fastapi import FastAPI, Query


app = FastAPI(title="SKILLAB ESCOPlus Skills Extender API",
              root_path="/escoplus-skills-extender")

# Create sub-routers
analysis_router = APIRouter(prefix="/api/analysis", tags=["Analysis"])
forecast_router = APIRouter(prefix="/api/forecasting", tags=["Forecasting"])

# === ANALYSIS ENDPOINTS ===
@analysis_router.get("/law-policies_extend_esco")
def law_policies_extend_esco(
    keywords: str = Query(..., description="Comma-separated keywords (e.g. data,ai,green)"),
    source: str = Query("eur_lex", description="Source of the policies"),
    similarity_threshold: float = Query(0.8, description="TF-IDF similarity threshold"),
    confidence_threshold: float = Query(0.6, description="Confidence cutoff for adding new skills")
):
    """
    Extend the ESCO taxonomy by matching policy-linked ESCO skills with non-ESCO technology skills.
    Computes network structure + centrality metrics for deeper analysis.
    """
    import pandas as pd, requests, os, numpy as np, networkx as nx
    from dotenv import load_dotenv
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from collections import defaultdict
    import traceback

    try:
        # === Authenticate ===
        load_dotenv()
        API = os.getenv("TRACKER_API", "https://skillab-tracker.csd.auth.gr/api")
        USERNAME = os.getenv("TRACKER_USERNAME", "")
        PASSWORD = os.getenv("TRACKER_PASSWORD", "")

        print("🔐 Authenticating with Tracker...")
        res = requests.post(f"{API}/login", json={"username": USERNAME, "password": PASSWORD}, timeout=15)
        res.raise_for_status()
        token = res.text.replace('"', "")
        headers = {"Authorization": f"Bearer {token}"}

        # === Retrieve Law/Policy Documents ===
        keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
        payload = {"keywords": keywords_list, "keywords_logic": "or", "sources": [source]}
        all_docs = []
        for page in range(1, 51):
            url = f"{API}/law-policies?page={page}&page_size=100"
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

        # === Extract Skill URIs ===
        skill_uris = [s for d in all_docs for s in d.get("skills", []) if isinstance(s, str) and s.startswith("http")]
        unique_uris = sorted(set(skill_uris))

        # === Map URIs → ESCO Labels ===
        print("🗺️ Mapping ESCO URIs to skill labels...")
        id_to_label = {}
        all_esco = []
        for page in range(1, 51):
            r = requests.post(f"{API}/skills?page={page}&page_size=100", headers=headers, timeout=30)
            if r.status_code != 200:
                break
            items = r.json().get("items", [])
            if not items:
                break
            all_esco.extend(items)
            if len(items) < 100:
                break
        for s in all_esco:
            sid = s.get("id")
            label = s.get("label", "").strip().lower()
            if sid and label:
                id_to_label[sid] = label
        ESCO_skill_labels = sorted({id_to_label.get(u, u) for u in unique_uris})
        print(f"🧠 {len(ESCO_skill_labels)} unique ESCO skills mapped.")

        # === Load Technology Skills (non-ESCO) ===
        csv_path = "technology_skills.csv"
        tech_df = pd.read_csv(csv_path, sep=None, engine='python', on_bad_lines='skip')
        tech_examples = []
        for row in tech_df["Example"].dropna().astype(str):
            tech_examples.extend([s.strip().lower() for s in row.replace(";", ",").split(",") if s.strip()])
        non_ESCO_skills = sorted(set(tech_examples))

        # === Similarity Computation ===
        corpus = ESCO_skill_labels + non_ESCO_skills
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4))
        tfidf = vec.fit_transform(corpus)
        esco_emb = tfidf[:len(ESCO_skill_labels)]
        non_emb = tfidf[len(ESCO_skill_labels):]
        sim = cosine_similarity(esco_emb, non_emb)

        matches = []
        for i, esco_skill in enumerate(ESCO_skill_labels):
            scores = sim[i]
            j = np.argmax(scores)
            sc = scores[j]
            if sc >= similarity_threshold:
                matches.append({"ESCO_skill": esco_skill, "non_ESCO_skill": non_ESCO_skills[j], "similarity": round(float(sc), 3)})

        # === Confidence & Filtering ===
        freq = defaultdict(int)
        for d in all_docs:
            for s in d.get("skills", []):
                if s in id_to_label:
                    freq[id_to_label[s]] += 1
        for m in matches:
            f = freq.get(m["ESCO_skill"], 1)
            m["confidence"] = round(m["similarity"] * (1 + np.log1p(f) / 10), 3)
        new_skills = [m for m in matches if m["non_ESCO_skill"] not in id_to_label.values()]
        high_conf = [m for m in new_skills if m["confidence"] >= confidence_threshold]
        proposed = sorted({m["non_ESCO_skill"] for m in high_conf})

        # === Prepare Association Rule–style Output ===
        associations = [
            {
                "ESCO_skill": m["ESCO_skill"],
                "non_ESCO_skill": m["non_ESCO_skill"],
                "similarity": m["similarity"],
                "confidence": m["confidence"]
            }
            for m in high_conf
        ]

        # === Build Network ===
        print("🌐 Building skill network...")
        G = nx.Graph()
        for m in high_conf:
            e, n = m["ESCO_skill"], m["non_ESCO_skill"]
            G.add_node(e, group="ESCO_skill")
            G.add_node(n, group="non_ESCO_skill")
            G.add_edge(e, n, similarity=m["similarity"], confidence=m["confidence"])

        # === Compute Network Metrics ===
        if G.number_of_edges() > 0:
            raw_degree = dict(G.degree())
            degree_centrality = {k: round(v, 3) for k, v in nx.degree_centrality(G).items()}
            avg_similarity = np.mean([d["similarity"] for _, _, d in G.edges(data=True)])
            clustering = round(nx.average_clustering(G), 3)
            components = [len(c) for c in nx.connected_components(G)]
            largest_component = max(components) if components else 0
        else:
            raw_degree, degree_centrality, avg_similarity, clustering, largest_component = {}, {}, 0, 0, 0

        # === Nodes with both raw and normalized degree ===
        nodes = [
            {
                "id": n,
                "group": G.nodes[n]["group"],
                "degree": raw_degree.get(n, 0),
                "degree_centrality": degree_centrality.get(n, 0)
            }
            for n in G.nodes()
        ]

        # === Edges for association rule–style graph ===
        edges = [
            {"source": u, "target": v, **d}
            for u, v, d in G.edges(data=True)
        ]

        # === Compute per-group degree averages ===
        esco_degrees = [n["degree"] for n in nodes if n["group"] == "ESCO_skill"]
        non_esco_degrees = [n["degree"] for n in nodes if n["group"] == "non_ESCO_skill"]
        group_degree_stats = {
            "ESCO_avg_degree": round(np.mean(esco_degrees), 3) if esco_degrees else 0,
            "non_ESCO_avg_degree": round(np.mean(non_esco_degrees), 3) if non_esco_degrees else 0
        }

        # === Save & Return Final Output ===
        pd.DataFrame(high_conf).to_csv("ESCOplus_Extended_from_Policies.csv", index=False)
        print(f"💾 Saved {len(high_conf)} new ESCO+ skills and {len(edges)} edges.")



        return {
            "message": "✅ ESCOPlus taxonomy extended with network metrics.",
            "summary": {
                "Policies processed": len(all_docs),
                "Mapped ESCO skills": len(ESCO_skill_labels),
                "Non-ESCO skills": len(non_ESCO_skills),
                "Matches found": len(matches),
                "Proposed ESCO+ extensions": len(high_conf)
            },
            "associations": associations[:50],  # direct skill pairs
            "network": {
                "nodes": nodes,
                "edges": edges
            },
            "network_stats": {
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "avg_similarity": round(float(avg_similarity), 3),
                "avg_clustering": clustering,
                "largest_component_size": largest_component,
                "avg_degree": round(np.mean(list(degree_centrality.values())), 3) if degree_centrality else 0,
                **group_degree_stats
            }
        }


    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

@analysis_router.get("/profiles_extend_esco")
def profiles_extend_esco(
    keywords: str = Query(..., description="Comma-separated keywords (e.g. AI, data, education)"),
    source: str = Query(None, description="Optional source filter for profiles (e.g. linkedin, eurofound)"),
    similarity_threshold: float = Query(0.8, description="Cosine similarity threshold for alternative label detection"),
    confidence_threshold: float = Query(0.6, description="Confidence threshold for including new skills"),
    max_pages: int = Query(10, description="Maximum number of pages to fetch (each page = 100 profiles)")
):
    """
    Fetch filtered profiles from Tracker API using keywords and optional filters.
    Extract ESCO skill labels, compare them with Technology Skills CSV,
    and identify new ESCO+ skills using similarity and confidence metrics.
    """
    import os, requests, numpy as np, pandas as pd
    from dotenv import load_dotenv
    from collections import defaultdict
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import traceback

    try:
        # === 1️⃣ Authenticate ===
        load_dotenv()
        API = os.getenv("TRACKER_API", "https://skillab-tracker.csd.auth.gr/api")
        USERNAME = os.getenv("TRACKER_USERNAME", "")
        PASSWORD = os.getenv("TRACKER_PASSWORD", "")

        print("🔐 Authenticating with Tracker...")
        res = requests.post(f"{API}/login", json={"username": USERNAME, "password": PASSWORD}, timeout=15)
        res.raise_for_status()
        token = res.text.replace('"', "")
        headers = {"Authorization": f"Bearer {token}"}
        print("✅ Authenticated successfully.")

        # === 2️⃣ Fetch Profiles ===
        keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
        print(f"📡 Fetching profiles matching: {keywords_list}")
        if source:
            print(f"🗂️ Source filter applied: {source}")
        else:
            print("🗂️ No source filter applied.")

        all_profiles = []
        for page in range(1, max_pages + 1):
            form_data = [
                ("keywords_logic", "or"),
                ("skill_ids_logic", "or"),
            ]
            for kw in keywords_list:
                form_data.append(("keywords", kw))
            if source:
                form_data.append(("sources", source))

            url = f"{API}/profiles?page={page}&page_size=100"
            print(f"📄 Fetching page {page}/{max_pages}...")

            res = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json"
                },
                data=form_data,
                timeout=90
            )
            if res.status_code != 200:
                print(f"⚠️ Page {page}: HTTP {res.status_code}")
                break

            data = res.json()
            items = data.get("items", [])
            if not items:
                print("✅ No more results — stopping.")
                break

            all_profiles.extend(items)
            if len(items) < 100:
                print("✅ Last page reached.")
                break

        print(f"🎯 Total profiles retrieved: {len(all_profiles)}")
        if not all_profiles:
            return {"error": "No profiles found for the given filters."}

        # === 3️⃣ Extract skill URIs and map to labels ===
        print("🧩 Extracting ESCO skill URIs from profiles...")
        skill_uris = []
        for profile in all_profiles:
            skill_uris.extend([s for s in profile.get("skills", []) if isinstance(s, str) and s.startswith("http")])

        unique_uris = sorted(set(skill_uris))
        print(f"🔗 Found {len(unique_uris)} unique skill URIs.")

        id_to_label = {}
        if unique_uris:
            print("📚 Fetching ESCO labels for skills...")
            all_esco = []
            for page in range(1, 51):
                r = requests.post(f"{API}/skills?page={page}&page_size=100", headers=headers, timeout=30)
                if r.status_code != 200:
                    break
                data = r.json()
                items = data.get("items", [])
                if not items:
                    break
                all_esco.extend(items)
                if len(items) < 100:
                    break
            id_to_label = {s["id"]: s.get("label", "").strip().lower() for s in all_esco if "id" in s}
            print(f"✅ Retrieved {len(id_to_label)} ESCO labels.")

        ESCO_skill_labels = [id_to_label.get(uri, uri) for uri in unique_uris]
        ESCO_skill_labels = sorted(set(ESCO_skill_labels))
        print(f"🧠 Mapped {len(ESCO_skill_labels)} ESCO skills from profiles.")

        # === 4️⃣ Load Technology Skills CSV ===
        csv_path = "technology_skills.csv"
        tech_df = pd.read_csv(csv_path, sep=None, engine='python', on_bad_lines='skip')
        if "Example" not in tech_df.columns:
            return {"error": "CSV must contain an 'Example' column."}

        tech_examples = []
        for row in tech_df["Example"].dropna().astype(str):
            tech_examples.extend([s.strip().lower() for s in row.replace(";", ",").split(",") if s.strip()])
        non_ESCO_skills = sorted(set(tech_examples))
        print(f"💾 Loaded {len(non_ESCO_skills)} non-ESCO technology skills.")

        # === 5️⃣ Compute Similarity ===
        print("🔍 Computing similarity between ESCO and non-ESCO skills...")
        corpus = list(ESCO_skill_labels) + list(non_ESCO_skills)
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4))
        tfidf = vectorizer.fit_transform(corpus)
        esco_emb = tfidf[:len(ESCO_skill_labels)]
        non_esco_emb = tfidf[len(ESCO_skill_labels):]
        sim_matrix = cosine_similarity(esco_emb, non_esco_emb)

        matches = []
        for i, esco_skill in enumerate(ESCO_skill_labels):
            sim_scores = sim_matrix[i]
            best_idx = np.argmax(sim_scores)
            best_score = sim_scores[best_idx]
            if best_score >= similarity_threshold:
                non_esco_match = non_ESCO_skills[best_idx]
                matches.append({
                    "ESCO_skill": esco_skill,
                    "non_ESCO_skill": non_esco_match,
                    "similarity": round(float(best_score), 3)
                })
                if i < 3:
                    print(f"✅ {esco_skill} ↔ {non_esco_match} ({best_score:.3f})")

        print(f"🔗 Found {len(matches)} ESCO ↔ non-ESCO matches.")

        # === 6️⃣ Compute Confidence + Identify new ESCO+ skills ===
        esco_labels = set(id_to_label.values())
        new_skills = [m for m in matches if m["non_ESCO_skill"] not in esco_labels]

        # Frequency weighting (how often ESCO skill appears in profiles)
        profile_freq = defaultdict(int)
        for profile in all_profiles:
            for s in profile.get("skills", []):
                if s in id_to_label:
                    label = id_to_label[s]
                    profile_freq[label] += 1

        for m in new_skills:
            freq = profile_freq.get(m["ESCO_skill"], 1)
            m["confidence"] = round(float(m["similarity"] * (1 + np.log1p(freq) / 10)), 3)

        high_conf_skills = [m for m in new_skills if m["confidence"] >= confidence_threshold]
        proposed_extensions = sorted(set([m["non_ESCO_skill"] for m in high_conf_skills]))

        print(f"🚀 {len(high_conf_skills)} high-confidence new ESCO+ skills proposed.")

        # === 7️⃣ Save Extended Taxonomy ===
        output_path = "ESCOplus_Extended_from_Profiles.csv"
        pd.DataFrame(high_conf_skills).to_csv(output_path, index=False)
        print(f"💾 Extended taxonomy saved to {output_path}")

        # === 🌐 Build Skill Network (ESCO ↔ non-ESCO) ===
        import networkx as nx

        print("🌐 Building skill network for visualization and metrics...")
        G = nx.Graph()

        for m in high_conf_skills:
            e, n = m["ESCO_skill"], m["non_ESCO_skill"]
            G.add_node(e, group="ESCO_skill")
            G.add_node(n, group="non_ESCO_skill")
            G.add_edge(e, n, similarity=m["similarity"], confidence=m["confidence"])

        # === 🧮 Compute Network Metrics ===
        if G.number_of_edges() > 0:
            raw_degree = dict(G.degree())
            degree_centrality = {k: round(v, 3) for k, v in nx.degree_centrality(G).items()}
            avg_similarity = np.mean([d["similarity"] for _, _, d in G.edges(data=True)])
            clustering = round(nx.average_clustering(G), 3)
            components = [len(c) for c in nx.connected_components(G)]
            largest_component = max(components) if components else 0
        else:
            raw_degree, degree_centrality, avg_similarity, clustering, largest_component = {}, {}, 0, 0, 0

        # === 🧩 Prepare Nodes ===
        nodes = [
            {
                "id": n,
                "group": G.nodes[n]["group"],
                "degree": raw_degree.get(n, 0),
                "degree_centrality": degree_centrality.get(n, 0)
            }
            for n in G.nodes()
        ]

        # === 🔗 Prepare Edges (Association-style) ===
        edges = [
            {"source": u, "target": v, **d}
            for u, v, d in G.edges(data=True)
        ]

        # === 📊 Per-Group Degree Stats ===
        esco_degrees = [n["degree"] for n in nodes if n["group"] == "ESCO_skill"]
        non_esco_degrees = [n["degree"] for n in nodes if n["group"] == "non_ESCO_skill"]
        group_degree_stats = {
            "ESCO_avg_degree": round(np.mean(esco_degrees), 3) if esco_degrees else 0,
            "non_ESCO_avg_degree": round(np.mean(non_esco_degrees), 3) if non_esco_degrees else 0
        }

        # === 🧠 Attach to the Output JSON ===
        network_data = {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "avg_similarity": round(float(avg_similarity), 3),
                "avg_clustering": clustering,
                "largest_component_size": largest_component,
                "avg_degree": round(np.mean(list(degree_centrality.values())), 3) if degree_centrality else 0,
                **group_degree_stats
            }
        }

        # === 7️⃣ Explainability Metrics ===
        print("📊 Computing explainability metrics (similarity + confidence)...")

        if high_conf_skills:
            similarities = [m["similarity"] for m in high_conf_skills]
            confidences = [m["confidence"] for m in high_conf_skills]

            explainability_metrics = {
                "avg_similarity": round(float(np.mean(similarities)), 3),
                "avg_confidence": round(float(np.mean(confidences)), 3),
                "similarity_distribution": {
                    "0.6-0.7": sum(0.6 <= s < 0.7 for s in similarities),
                    "0.7-0.8": sum(0.7 <= s < 0.8 for s in similarities),
                    "0.8-1.0": sum(s >= 0.8 for s in similarities),
                },
                "confidence_distribution": {
                    "0.6-0.7": sum(0.6 <= c < 0.7 for c in confidences),
                    "0.7-0.8": sum(0.7 <= c < 0.8 for c in confidences),
                    "0.8-1.0": sum(c >= 0.8 for c in confidences),
                }
            }
        else:
            explainability_metrics = {
                "avg_similarity": 0,
                "avg_confidence": 0,
                "similarity_distribution": {},
                "confidence_distribution": {}
            }

        # === ✅ Return Results ===
        return {
            "message": "✅ ESCOPlus taxonomy successfully extended using profile-derived skills.",
            "summary": {
                "Profiles processed": len(all_profiles),
                "ESCO skills mapped": len(ESCO_skill_labels),
                "Non-ESCO skills (CSV)": len(non_ESCO_skills),
                "Matches found": len(matches),
                "High-confidence new skills": len(high_conf_skills)
            },
            "proposed_extensions": proposed_extensions[:100],
            "explainability_metrics": explainability_metrics,
            "new_skills_preview": high_conf_skills[:20],
            "output_file": output_path,
            "network": network_data
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

@analysis_router.get("/jobs_ultra")
def jobs_extend_esco_ultra(
    keywords: str = Query(..., description="Comma-separated keywords (e.g. AI, data, software)"),
    source: str = Query(None, description="Optional source filter (e.g. linkedin, indeed)"),
    min_upload_date: str = Query(None, description="Minimum upload date (YYYY-MM-DD)"),
    max_upload_date: str = Query(None, description="Maximum upload date (YYYY-MM-DD)"),
    similarity_threshold: float = Query(0.8, description="Cosine similarity threshold for alternative label detection"),
    confidence_threshold: float = Query(0.6, description="Confidence threshold for including new skills"),
    max_pages: int = Query(10, description="Maximum number of pages to fetch (each page = 100 jobs)")
):
    """
    Fetch filtered job postings from the Tracker API using keywords and optional filters.
    Extract ESCO skill labels, compare them with Technology Skills CSV,
    and identify new ESCO+ skills using similarity and confidence metrics.
    """
    import os, requests, numpy as np, pandas as pd
    from dotenv import load_dotenv
    from collections import defaultdict
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import traceback

    try:
        # === 1️⃣ Authenticate with Tracker ===
        load_dotenv()
        API = os.getenv("TRACKER_API", "https://skillab-tracker.csd.auth.gr/api")
        USERNAME = os.getenv("TRACKER_USERNAME", "")
        PASSWORD = os.getenv("TRACKER_PASSWORD", "")

        print("🔐 Authenticating with Tracker...")
        res = requests.post(f"{API}/login", json={"username": USERNAME, "password": PASSWORD}, timeout=15)
        res.raise_for_status()
        token = res.text.replace('"', "")
        headers = {"Authorization": f"Bearer {token}"}
        print("✅ Authenticated successfully.")

        # === 2️⃣ Fetch Job Postings ===
        keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
        print(f"📡 Fetching jobs for keywords: {keywords_list}")

        all_jobs = []
        for page in range(1, max_pages + 1):
            form_data = [
                ("keywords_logic", "or"),
                ("skill_ids_logic", "or"),
                ("occupation_ids_logic", "or")
            ]
            for kw in keywords_list:
                form_data.append(("keywords", kw))
            if source:
                form_data.append(("sources", source))
            if min_upload_date:
                form_data.append(("min_upload_date", min_upload_date))
            if max_upload_date:
                form_data.append(("max_upload_date", max_upload_date))

            url = f"{API}/jobs?page={page}&page_size=100"
            res = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json"
                },
                data=form_data,
                timeout=90
            )
            if res.status_code != 200:
                print(f"⚠️ Page {page}: HTTP {res.status_code}")
                break

            data = res.json()
            items = data.get("items", [])
            if not items:
                break

            all_jobs.extend(items)
            if len(items) < 100:
                break

        print(f"🎯 Total job postings retrieved: {len(all_jobs)}")
        if not all_jobs:
            return {"error": "No job postings found for the given filters."}

        # === 3️⃣ Extract skill URIs and map to labels ===
        print("🧩 Extracting ESCO skill URIs from jobs...")
        skill_uris = []
        for job in all_jobs:
            skill_uris.extend([s for s in job.get("skills", []) if isinstance(s, str) and s.startswith("http")])

        unique_uris = sorted(set(skill_uris))
        print(f"🔗 Found {len(unique_uris)} unique skill URIs.")

        id_to_label = {}
        if unique_uris:
            print("📚 Fetching ESCO labels for skills...")
            all_esco = []
            for page in range(1, 51):
                r = requests.post(f"{API}/skills?page={page}&page_size=100", headers=headers, timeout=30)
                if r.status_code != 200:
                    break
                data = r.json()
                items = data.get("items", [])
                if not items:
                    break
                all_esco.extend(items)
                if len(items) < 100:
                    break
            id_to_label = {s["id"]: s.get("label", "").strip().lower() for s in all_esco if "id" in s}
            print(f"✅ Retrieved {len(id_to_label)} ESCO labels.")

        ESCO_skill_labels = [id_to_label.get(uri, uri) for uri in unique_uris]
        ESCO_skill_labels = sorted(set(ESCO_skill_labels))
        print(f"🧠 Mapped {len(ESCO_skill_labels)} ESCO skills from jobs.")

        # === 4️⃣ Load Technology Skills CSV ===
        csv_path = "technology_skills.csv"
        tech_df = pd.read_csv(csv_path, sep=None, engine='python', on_bad_lines='skip')
        if "Example" not in tech_df.columns:
            return {"error": "CSV must contain an 'Example' column."}

        tech_examples = []
        for row in tech_df["Example"].dropna().astype(str):
            tech_examples.extend([s.strip().lower() for s in row.replace(";", ",").split(",") if s.strip()])
        non_ESCO_skills = sorted(set(tech_examples))
        print(f"💾 Loaded {len(non_ESCO_skills)} non-ESCO technology skills.")

        # === 5️⃣ Compute Similarity ===
        print("🔍 Computing similarity between ESCO and non-ESCO skills...")
        corpus = list(ESCO_skill_labels) + list(non_ESCO_skills)
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4))
        tfidf = vectorizer.fit_transform(corpus)
        esco_emb = tfidf[:len(ESCO_skill_labels)]
        non_esco_emb = tfidf[len(ESCO_skill_labels):]
        sim_matrix = cosine_similarity(esco_emb, non_esco_emb)

        matches = []
        for i, esco_skill in enumerate(ESCO_skill_labels):
            sim_scores = sim_matrix[i]
            best_idx = np.argmax(sim_scores)
            best_score = sim_scores[best_idx]
            if best_score >= similarity_threshold:
                non_esco_match = non_ESCO_skills[best_idx]
                matches.append({
                    "ESCO_skill": esco_skill,
                    "non_ESCO_skill": non_esco_match,
                    "similarity": round(float(best_score), 3)
                })
                if i < 3:
                    print(f"✅ {esco_skill} ↔ {non_esco_match} ({best_score:.3f})")

        print(f"🔗 Found {len(matches)} ESCO ↔ non-ESCO matches.")

        # === 6️⃣ Compute Confidence + Identify new ESCO+ skills ===
        esco_labels = set(id_to_label.values())
        new_skills = [m for m in matches if m["non_ESCO_skill"] not in esco_labels]

        # Frequency weighting (how often ESCO skill appears in jobs)
        job_freq = defaultdict(int)
        for job in all_jobs:
            for s in job.get("skills", []):
                if s in id_to_label:
                    label = id_to_label[s]
                    job_freq[label] += 1

        for m in new_skills:
            freq = job_freq.get(m["ESCO_skill"], 1)
            m["confidence"] = round(float(m["similarity"] * (1 + np.log1p(freq) / 10)), 3)

        high_conf_skills = [m for m in new_skills if m["confidence"] >= confidence_threshold]
        proposed_extensions = sorted(set([m["non_ESCO_skill"] for m in high_conf_skills]))

        print(f"🚀 {len(high_conf_skills)} high-confidence new ESCO+ skills proposed.")

        # === 7️⃣ Save Extended Taxonomy ===
        output_path = "ESCOplus_Extended_from_Jobs.csv"
        pd.DataFrame(high_conf_skills).to_csv(output_path, index=False)
        print(f"💾 Extended taxonomy saved to {output_path}")

        # === 🌐 Build Skill Network (ESCO ↔ non-ESCO from Jobs) ===
        import networkx as nx

        print("🌐 Building skill network from job-based matches...")
        G = nx.Graph()

        for m in high_conf_skills:
            e, n = m["ESCO_skill"], m["non_ESCO_skill"]
            G.add_node(e, group="ESCO_skill")
            G.add_node(n, group="non_ESCO_skill")
            G.add_edge(e, n, similarity=m["similarity"], confidence=m["confidence"])

        # === 🧮 Compute Network Metrics ===
        if G.number_of_edges() > 0:
            raw_degree = dict(G.degree())
            degree_centrality = {k: round(v, 3) for k, v in nx.degree_centrality(G).items()}
            avg_similarity = np.mean([d["similarity"] for _, _, d in G.edges(data=True)])
            clustering = round(nx.average_clustering(G), 3)
            components = [len(c) for c in nx.connected_components(G)]
            largest_component = max(components) if components else 0
        else:
            raw_degree, degree_centrality, avg_similarity, clustering, largest_component = {}, {}, 0, 0, 0

        # === 🧩 Nodes and Edges ===
        nodes = [
            {
                "id": n,
                "group": G.nodes[n]["group"],
                "degree": raw_degree.get(n, 0),
                "degree_centrality": degree_centrality.get(n, 0)
            }
            for n in G.nodes()
        ]

        edges = [
            {"source": u, "target": v, **d}
            for u, v, d in G.edges(data=True)
        ]

        # === 📊 Group Degree Stats ===
        esco_degrees = [n["degree"] for n in nodes if n["group"] == "ESCO_skill"]
        non_esco_degrees = [n["degree"] for n in nodes if n["group"] == "non_ESCO_skill"]
        group_degree_stats = {
            "ESCO_avg_degree": round(np.mean(esco_degrees), 3) if esco_degrees else 0,
            "non_ESCO_avg_degree": round(np.mean(non_esco_degrees), 3) if non_esco_degrees else 0
        }

        # === 🧠 Final Network Data Object ===
        network_data = {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "avg_similarity": round(float(avg_similarity), 3),
                "avg_clustering": clustering,
                "largest_component_size": largest_component,
                "avg_degree": round(np.mean(list(degree_centrality.values())), 3) if degree_centrality else 0,
                **group_degree_stats
            }
        }

        # === 7️⃣ Explainability Metrics ===
        print("📊 Computing explainability metrics (similarity + confidence)...")

        if high_conf_skills:
            similarities = [m["similarity"] for m in high_conf_skills]
            confidences = [m["confidence"] for m in high_conf_skills]

            explainability_metrics = {
                "avg_similarity": round(float(np.mean(similarities)), 3),
                "avg_confidence": round(float(np.mean(confidences)), 3),
                "similarity_distribution": {
                    "0.6-0.7": sum(0.6 <= s < 0.7 for s in similarities),
                    "0.7-0.8": sum(0.7 <= s < 0.8 for s in similarities),
                    "0.8-1.0": sum(s >= 0.8 for s in similarities),
                },
                "confidence_distribution": {
                    "0.6-0.7": sum(0.6 <= c < 0.7 for c in confidences),
                    "0.7-0.8": sum(0.7 <= c < 0.8 for c in confidences),
                    "0.8-1.0": sum(c >= 0.8 for c in confidences),
                }
            }
        else:
            explainability_metrics = {
                "avg_similarity": 0,
                "avg_confidence": 0,
                "similarity_distribution": {},
                "confidence_distribution": {}
            }

        # === ✅ Return Results ===
        return {
            "message": "✅ ESCOPlus taxonomy successfully extended using job-derived skills.",
            "summary": {
                "Job postings processed": len(all_jobs),
                "ESCO skills mapped": len(ESCO_skill_labels),
                "Non-ESCO skills (CSV)": len(non_ESCO_skills),
                "Matches found": len(matches),
                "High-confidence new skills": len(high_conf_skills)
            },
            "proposed_extensions": proposed_extensions[:100],
            "explainability_metrics": explainability_metrics,
            "new_skills_preview": high_conf_skills[:20],
            "output_file": output_path,
            "network": network_data
        }


    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

@analysis_router.get("/courses_ultra")
def courses_extend_esco(
    keywords: str = Query(..., description="Comma-separated keywords (e.g. data, ai, green)"),
    source: str = Query(None, description="Optional source filter (e.g. Udacity, europass)"),
    similarity_threshold: float = Query(0.8, description="Cosine similarity threshold for alternative label detection"),
    confidence_threshold: float = Query(0.6, description="Confidence threshold for including new skills")
):
    """
    Fetch filtered courses from Tracker,
    extract ESCO skill labels, compare them with Technology Skills CSV,
    and identify new ESCO+ skills using similarity and confidence metrics.
    """
    import os, requests, numpy as np, pandas as pd
    from dotenv import load_dotenv
    from collections import defaultdict
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import traceback

    try:
        # === 1️⃣ Authenticate with Tracker ===
        load_dotenv()
        API = os.getenv("TRACKER_API", "https://skillab-tracker.csd.auth.gr/api")
        USERNAME = os.getenv("TRACKER_USERNAME", "")
        PASSWORD = os.getenv("TRACKER_PASSWORD", "")

        print("🔐 Authenticating with Tracker...")
        res = requests.post(f"{API}/login", json={"username": USERNAME, "password": PASSWORD}, timeout=15)
        res.raise_for_status()
        token = res.text.replace('"', "")
        headers = {"Authorization": f"Bearer {token}"}
        print("✅ Authenticated successfully.")

        # === 2️⃣ Fetch Courses from Tracker ===
        keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
        print(f"📡 Querying /courses for {keywords_list}...")

        all_courses, max_pages = [], 50
        for page in range(1, max_pages + 1):
            form_data = [
                ("keywords_logic", "or"),
                ("skill_ids_logic", "or"),
                ("sources", source),
            ]
            for kw in keywords_list:
                form_data.append(("keywords", kw))

            url = f"{API}/courses?page={page}&page_size=100"
            res = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json"
                },
                data=form_data,
                timeout=60
            )
            if res.status_code != 200:
                print(f"⚠️ Page {page}: HTTP {res.status_code}")
                break

            data = res.json()
            items = data.get("items", [])
            if not items:
                break
            all_courses.extend(items)
            if len(items) < 100:
                break

        print(f"📄 Retrieved {len(all_courses)} courses.")

        if not all_courses:
            return {"error": "No courses found for the given keywords/source."}

        # === 3️⃣ Extract skill URIs and map to labels ===
        print("🧩 Extracting ESCO skill URIs from courses...")
        skill_uris = []
        for course in all_courses:
            skill_uris.extend([s for s in course.get("skills", []) if isinstance(s, str) and s.startswith("http")])

        unique_uris = sorted(set(skill_uris))
        print(f"🔗 Found {len(unique_uris)} unique skill URIs.")

        id_to_label = {}
        if unique_uris:
            print("📚 Fetching ESCO labels for skills...")
            all_esco = []
            for page in range(1, 51):
                r = requests.post(f"{API}/skills?page={page}&page_size=100", headers=headers, timeout=30)
                if r.status_code != 200:
                    break
                data = r.json()
                items = data.get("items", [])
                if not items:
                    break
                all_esco.extend(items)
                if len(items) < 100:
                    break
            id_to_label = {s["id"]: s.get("label", "").strip().lower() for s in all_esco if "id" in s}
            print(f"✅ Retrieved {len(id_to_label)} skill labels.")

        ESCO_skill_labels = [id_to_label.get(uri, uri) for uri in unique_uris]
        ESCO_skill_labels = sorted(set(ESCO_skill_labels))
        print(f"🧠 Mapped {len(ESCO_skill_labels)} ESCO skills from courses.")

        # === 4️⃣ Load Technology Skills CSV ===
        csv_path = "technology_skills.csv"
        tech_df = pd.read_csv(csv_path, sep=None, engine='python', on_bad_lines='skip')
        if "Example" not in tech_df.columns:
            return {"error": "CSV must contain an 'Example' column."}

        tech_examples = []
        for row in tech_df["Example"].dropna().astype(str):
            tech_examples.extend([s.strip().lower() for s in row.replace(";", ",").split(",") if s.strip()])
        non_ESCO_skills = sorted(set(tech_examples))
        print(f"💾 Loaded {len(non_ESCO_skills)} technology (non-ESCO) skills.")

        # === 5️⃣ Compute Similarity between course ESCO and non-ESCO ===
        print("🔍 Computing similarity...")
        corpus = list(ESCO_skill_labels) + list(non_ESCO_skills)
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4))
        tfidf = vectorizer.fit_transform(corpus)
        esco_emb = tfidf[:len(ESCO_skill_labels)]
        non_esco_emb = tfidf[len(ESCO_skill_labels):]
        sim_matrix = cosine_similarity(esco_emb, non_esco_emb)

        matches = []
        for i, esco_skill in enumerate(ESCO_skill_labels):
            sim_scores = sim_matrix[i]
            best_idx = np.argmax(sim_scores)
            best_score = sim_scores[best_idx]
            if best_score >= similarity_threshold:
                non_esco_match = non_ESCO_skills[best_idx]
                matches.append({
                    "ESCO_skill": esco_skill,
                    "non_ESCO_skill": non_esco_match,
                    "similarity": round(float(best_score), 3)
                })
                if i < 3:
                    print(f"✅ {esco_skill} ↔ {non_esco_match} ({best_score:.3f})")

        print(f"🔗 Found {len(matches)} ESCO ↔ non-ESCO matches.")

        # === 6️⃣ Compute Confidence + Identify new ESCO+ skills ===
        print("📈 Computing confidence and selecting new skills...")
        esco_labels = set(id_to_label.values())
        new_skills = [m for m in matches if m["non_ESCO_skill"] not in esco_labels]

        # Frequency weighting by course occurrence
        policy_freq = defaultdict(int)
        for course in all_courses:
            for s in course.get("skills", []):
                if s in id_to_label:
                    label = id_to_label[s]
                    policy_freq[label] += 1

        for m in new_skills:
            freq = policy_freq.get(m["ESCO_skill"], 1)
            m["confidence"] = round(float(m["similarity"] * (1 + np.log1p(freq) / 10)), 3)

        high_conf_skills = [m for m in new_skills if m["confidence"] >= confidence_threshold]
        proposed_extensions = sorted(set([m["non_ESCO_skill"] for m in high_conf_skills]))

        print(f"🚀 {len(high_conf_skills)} high-confidence new skills proposed for ESCO+ extension.")

        # === 7️⃣ Save Extended Taxonomy ===
        output_path = "ESCOplus_Extended_from_Courses.csv"
        pd.DataFrame(high_conf_skills).to_csv(output_path, index=False)
        print(f"💾 Extended taxonomy saved at {output_path}")

        # === 🌐 Build Skill Network (ESCO ↔ non-ESCO from Jobs) ===
        import networkx as nx

        print("🌐 Building skill network from job-based matches...")
        G = nx.Graph()

        for m in high_conf_skills:
            e, n = m["ESCO_skill"], m["non_ESCO_skill"]
            G.add_node(e, group="ESCO_skill")
            G.add_node(n, group="non_ESCO_skill")
            G.add_edge(e, n, similarity=m["similarity"], confidence=m["confidence"])

        # === 🧮 Compute Network Metrics ===
        if G.number_of_edges() > 0:
            raw_degree = dict(G.degree())
            degree_centrality = {k: round(v, 3) for k, v in nx.degree_centrality(G).items()}
            avg_similarity = np.mean([d["similarity"] for _, _, d in G.edges(data=True)])
            clustering = round(nx.average_clustering(G), 3)
            components = [len(c) for c in nx.connected_components(G)]
            largest_component = max(components) if components else 0
        else:
            raw_degree, degree_centrality, avg_similarity, clustering, largest_component = {}, {}, 0, 0, 0

        # === 🧩 Nodes and Edges ===
        nodes = [
            {
                "id": n,
                "group": G.nodes[n]["group"],
                "degree": raw_degree.get(n, 0),
                "degree_centrality": degree_centrality.get(n, 0)
            }
            for n in G.nodes()
        ]

        edges = [
            {"source": u, "target": v, **d}
            for u, v, d in G.edges(data=True)
        ]

        # === 📊 Group Degree Stats ===
        esco_degrees = [n["degree"] for n in nodes if n["group"] == "ESCO_skill"]
        non_esco_degrees = [n["degree"] for n in nodes if n["group"] == "non_ESCO_skill"]
        group_degree_stats = {
            "ESCO_avg_degree": round(np.mean(esco_degrees), 3) if esco_degrees else 0,
            "non_ESCO_avg_degree": round(np.mean(non_esco_degrees), 3) if non_esco_degrees else 0
        }

        # === 🧠 Final Network Data Object ===
        network_data = {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "avg_similarity": round(float(avg_similarity), 3),
                "avg_clustering": clustering,
                "largest_component_size": largest_component,
                "avg_degree": round(np.mean(list(degree_centrality.values())), 3) if degree_centrality else 0,
                **group_degree_stats
            }
        }

        # === ✅ Return results ===
        return {
            "message": "✅ ESCOPlus taxonomy successfully extended using course-derived skills.",
            "summary": {
                "Courses processed": len(all_courses),
                "ESCO skills mapped": len(ESCO_skill_labels),
                "Non-ESCO tech skills": len(non_ESCO_skills),
                "Matches found": len(matches),
                "High-confidence new skills": len(high_conf_skills)
            },
            "proposed_extensions": proposed_extensions[:100],
            "new_skills_preview": high_conf_skills[:20],
            "output_file": output_path,
            "network": network_data
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}
# === FORECASTING ENDPOINTS ===
@forecast_router.get("/profiles")
def profiles_link_prediction(
    keywords: str = Query(..., description="Comma-separated keywords (e.g. AI, data, education)"),
    source: str = Query(None, description="Optional source filter for profiles (e.g. linkedin, eurofound)"),
    similarity_threshold: float = Query(0.7, description="Minimum similarity to consider existing edges"),
    top_k: int = Query(30, description="Number of top predicted links to return"),
    method: str = Query("adamic_adar", description="Link prediction method: adamic_adar, resource_allocation, or jaccard")
):
    """
    Predict new potential ESCO ↔ non-ESCO connections from user profiles using classical link prediction methods.
    """
    import os, requests, numpy as np, pandas as pd, traceback, networkx as nx
    from dotenv import load_dotenv
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    try:
        # === 1️⃣ Authenticate ===
        load_dotenv()
        API = os.getenv("TRACKER_API", "https://skillab-tracker.csd.auth.gr/api")
        USERNAME = os.getenv("TRACKER_USERNAME", "")
        PASSWORD = os.getenv("TRACKER_PASSWORD", "")

        print("🔐 Authenticating with Tracker...")
        res = requests.post(f"{API}/login", json={"username": USERNAME, "password": PASSWORD}, timeout=15)
        res.raise_for_status()
        token = res.text.replace('"', "")
        headers = {"Authorization": f"Bearer {token}"}

        # === 2️⃣ Retrieve Profiles ===
        keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
        payload = {"keywords": keywords_list, "keywords_logic": "or"}
        if source:
            payload["sources"] = [source]

        all_profiles = []
        for page in range(1, 51):
            url = f"{API}/profiles?page={page}&page_size=100"
            res = requests.post(url, headers=headers, data=payload, timeout=60)
            if res.status_code != 200:
                break
            data = res.json()
            items = data.get("items", [])
            if not items:
                break
            all_profiles.extend(items)
            if len(items) < 100:
                break
        print(f"📄 Retrieved {len(all_profiles)} profiles.")

        if not all_profiles:
            return {"error": "No profiles found for given filters."}

        # === 3️⃣ Extract ESCO Skills ===
        skill_uris = [s for d in all_profiles for s in d.get("skills", []) if isinstance(s, str) and s.startswith("http")]
        unique_uris = sorted(set(skill_uris))

        # Map URIs → ESCO Labels
        id_to_label = {}
        all_esco = []
        for page in range(1, 51):
            r = requests.post(f"{API}/skills?page={page}&page_size=100", headers=headers, timeout=30)
            if r.status_code != 200:
                break
            items = r.json().get("items", [])
            if not items:
                break
            all_esco.extend(items)
            if len(items) < 100:
                break
        for s in all_esco:
            sid = s.get("id")
            label = s.get("label", "").strip().lower()
            if sid and label:
                id_to_label[sid] = label
        ESCO_skill_labels = sorted({id_to_label.get(u, u) for u in unique_uris})
        print(f"🧠 {len(ESCO_skill_labels)} unique ESCO skills mapped.")

        # === 4️⃣ Load Non-ESCO Skills CSV ===
        csv_path = "technology_skills.csv"
        tech_df = pd.read_csv(csv_path, sep=None, engine="python", on_bad_lines="skip")
        if "Example" not in tech_df.columns:
            return {"error": "CSV must contain an 'Example' column."}
        tech_examples = []
        for row in tech_df["Example"].dropna().astype(str):
            tech_examples.extend([s.strip().lower() for s in row.replace(";", ",").split(",") if s.strip()])
        non_ESCO_skills = sorted(set(tech_examples))
        print(f"💾 Loaded {len(non_ESCO_skills)} technology (non-ESCO) skills.")

        # === 5️⃣ Build Graph Based on TF-IDF Similarity ===
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

        print(f"🕸️ Graph built with {len(G.nodes())} nodes and {len(G.edges())} edges.")

        if G.number_of_edges() == 0:
            return {"error": "No edges found. Try lowering similarity_threshold."}

        # === 6️⃣ Link Prediction Method ===
        if method == "adamic_adar":
            preds = nx.adamic_adar_index(G)
        elif method == "resource_allocation":
            preds = nx.resource_allocation_index(G)
        else:
            preds = nx.jaccard_coefficient(G)

        preds_sorted = sorted(preds, key=lambda x: x[2], reverse=True)[:top_k]

        # === 7️⃣ Add Weighted Adjustment + Emoji Confidence ===
        candidate_links = []
        for u, v, score in preds_sorted:
            # Weighted adjustment using existing edge weights
            common_neighbors = list(nx.common_neighbors(G, u, v))
            weighted_score = np.mean([G[u][n]["weight"] * G[v][n]["weight"] for n in common_neighbors]) if common_neighbors else 0
            combined_score = round((score + weighted_score) / 2, 3)

            # Emoji and confidence level
            if combined_score >= 0.8:
                emoji = "🟢"
                level = "High confidence"
            elif combined_score >= 0.6:
                emoji = "🟡"
                level = "Medium confidence"
            else:
                emoji = "🔴"
                level = "Low confidence"

            candidate_links.append({
                "source": u,
                "target": v,
                "predicted_score": combined_score,
                "confidence_level": level,
                "emoji": emoji
            })

        # === 8️⃣ Build Summary ===
        summary_counts = {
            "high": sum(1 for c in candidate_links if c["predicted_score"] >= 0.8),
            "medium": sum(1 for c in candidate_links if 0.6 <= c["predicted_score"] < 0.8),
            "low": sum(1 for c in candidate_links if c["predicted_score"] < 0.6)
        }

        # === 9️⃣ Return Response ===
        return {
            "message": "✅ ESCOPlus classical link prediction completed.",
            "summary": {
                "Profiles processed": len(all_profiles),
                "Mapped ESCO skills": len(ESCO_skill_labels),
                "Non-ESCO skills": len(non_ESCO_skills),
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



@forecast_router.get("/jobsd-forecast")
def jobs_link_prediction(
    keywords: str = Query(..., description="Comma-separated keywords (e.g. AI, data, software)"),
    source: str = Query(None, description="Optional source filter (e.g. linkedin, indeed)"),
    min_upload_date: str = Query(None, description="Minimum upload date (YYYY-MM-DD)"),
    max_upload_date: str = Query(None, description="Maximum upload date (YYYY-MM-DD)"),
    similarity_threshold: float = Query(0.7, description="Minimum similarity to consider edges"),
    top_k: int = Query(30, description="Number of top predicted links to return"),
    method: str = Query("adamic_adar", description="Link prediction method: adamic_adar, resource_allocation, or jaccard"),
    max_pages: int = Query(10, description="Maximum number of pages to fetch (each page = 100 jobs)")
):
    """
    Predict new potential ESCO ↔ non-ESCO connections using job-related ESCO skills
    and technology skills from CSV via classical link prediction methods.
    """
    import os, requests, numpy as np, pandas as pd, traceback, networkx as nx
    from dotenv import load_dotenv
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    try:
        # === 1️⃣ Authenticate ===
        load_dotenv()
        API = os.getenv("TRACKER_API", "https://skillab-tracker.csd.auth.gr/api")
        USERNAME = os.getenv("TRACKER_USERNAME", "")
        PASSWORD = os.getenv("TRACKER_PASSWORD", "")

        print("🔐 Authenticating with Tracker...")
        res = requests.post(f"{API}/login", json={"username": USERNAME, "password": PASSWORD}, timeout=15)
        res.raise_for_status()
        token = res.text.replace('"', "")
        headers = {"Authorization": f"Bearer {token}"}

        # === 2️⃣ Retrieve Job Postings ===
        keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
        all_jobs = []
        for page in range(1, max_pages + 1):
            form_data = [
                ("keywords_logic", "or"),
                ("skill_ids_logic", "or"),
                ("occupation_ids_logic", "or")
            ]
            for kw in keywords_list:
                form_data.append(("keywords", kw))
            if source:
                form_data.append(("sources", source))
            if min_upload_date:
                form_data.append(("min_upload_date", min_upload_date))
            if max_upload_date:
                form_data.append(("max_upload_date", max_upload_date))

            url = f"{API}/jobs?page={page}&page_size=100"
            res = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json"
                },
                data=form_data,
                timeout=90
            )
            if res.status_code != 200:
                print(f"⚠️ Page {page}: HTTP {res.status_code}")
                break
            data = res.json()
            items = data.get("items", [])
            if not items:
                break
            all_jobs.extend(items)
            if len(items) < 100:
                break

        print(f"🎯 Total job postings retrieved: {len(all_jobs)}")
        if not all_jobs:
            return {"error": "No job postings found for the given filters."}

        # === 3️⃣ Extract ESCO Skills ===
        print("🧩 Extracting ESCO skill URIs from jobs...")
        skill_uris = []
        for job in all_jobs:
            skill_uris.extend([s for s in job.get("skills", []) if isinstance(s, str) and s.startswith("http")])
        unique_uris = sorted(set(skill_uris))
        print(f"🔗 Found {len(unique_uris)} unique skill URIs.")

        # Map URIs → ESCO Labels
        id_to_label = {}
        all_esco = []
        for page in range(1, 51):
            r = requests.post(f"{API}/skills?page={page}&page_size=100", headers=headers, timeout=30)
            if r.status_code != 200:
                break
            items = r.json().get("items", [])
            if not items:
                break
            all_esco.extend(items)
            if len(items) < 100:
                break

        for s in all_esco:
            sid = s.get("id")
            label = s.get("label", "").strip().lower()
            if sid and label:
                id_to_label[sid] = label

        ESCO_skill_labels = sorted({id_to_label.get(u, u) for u in unique_uris})
        print(f"🧠 {len(ESCO_skill_labels)} unique ESCO skills mapped from jobs.")

        # === 4️⃣ Load Non-ESCO (Tech) Skills CSV ===
        csv_path = "technology_skills.csv"
        tech_df = pd.read_csv(csv_path, sep=None, engine="python", on_bad_lines="skip")
        if "Example" not in tech_df.columns:
            return {"error": "CSV must contain an 'Example' column."}
        tech_examples = []
        for row in tech_df["Example"].dropna().astype(str):
            tech_examples.extend([s.strip().lower() for s in row.replace(";", ",").split(",") if s.strip()])
        non_ESCO_skills = sorted(set(tech_examples))
        print(f"💾 Loaded {len(non_ESCO_skills)} technology (non-ESCO) skills.")

        # === 5️⃣ Build Graph via TF-IDF Similarity ===
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

        print(f"🕸️ Graph built with {len(G.nodes())} nodes and {len(G.edges())} edges.")
        if G.number_of_edges() == 0:
            return {"error": "No edges found. Try lowering similarity_threshold."}

        # === 6️⃣ Classical Link Prediction ===
        if method == "adamic_adar":
            preds = nx.adamic_adar_index(G)
        elif method == "resource_allocation":
            preds = nx.resource_allocation_index(G)
        else:
            preds = nx.jaccard_coefficient(G)

        preds_sorted = sorted(preds, key=lambda x: x[2], reverse=True)[:top_k]

        # === 7️⃣ Combine Scores + Emoji Confidence ===
        candidate_links = []
        for u, v, score in preds_sorted:
            common_neighbors = list(nx.common_neighbors(G, u, v))
            weighted_score = np.mean([G[u][n]["weight"] * G[v][n]["weight"] for n in common_neighbors]) if common_neighbors else 0
            combined_score = round((score + weighted_score) / 2, 3)

            if combined_score >= 0.8:
                emoji = "🟢"
                level = "High confidence"
            elif combined_score >= 0.6:
                emoji = "🟡"
                level = "Medium confidence"
            else:
                emoji = "🔴"
                level = "Low confidence"

            candidate_links.append({
                "source": u,
                "target": v,
                "predicted_score": combined_score,
                "confidence_level": level,
                "emoji": emoji
            })

        # === 8️⃣ Summary ===
        summary_counts = {
            "high": sum(1 for c in candidate_links if c["predicted_score"] >= 0.8),
            "medium": sum(1 for c in candidate_links if 0.6 <= c["predicted_score"] < 0.8),
            "low": sum(1 for c in candidate_links if c["predicted_score"] < 0.6)
        }

        # === 9️⃣ Return Response ===
        return {
            "message": "✅ ESCOPlus job-based link prediction completed.",
            "summary": {
                "Jobs processed": len(all_jobs),
                "Mapped ESCO skills": len(ESCO_skill_labels),
                "Non-ESCO skills": len(non_ESCO_skills),
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

@forecast_router.get("/courses")
def courses_link_prediction(
    keywords: str = Query(..., description="Comma-separated keywords (e.g. data, ai, green)"),
    source: str = Query("coursera", description="Source of the courses"),
    similarity_threshold: float = Query(0.7, description="Minimum similarity to consider edges"),
    top_k: int = Query(30, description="Number of top predicted links to return"),
    method: str = Query("adamic_adar", description="Link prediction method: adamic_adar, resource_allocation, or jaccard")
):
    """
    Predict new potential ESCO ↔ non-ESCO connections using course-related ESCO skills
    and technology skills from CSV, via classical link prediction methods.
    """
    import os, requests, numpy as np, pandas as pd, traceback, networkx as nx
    from dotenv import load_dotenv
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    try:
        # === 1️⃣ Authenticate ===
        load_dotenv()
        API = os.getenv("TRACKER_API", "https://skillab-tracker.csd.auth.gr/api")
        USERNAME = os.getenv("TRACKER_USERNAME", "")
        PASSWORD = os.getenv("TRACKER_PASSWORD", "")

        print("🔐 Authenticating with Tracker...")
        res = requests.post(f"{API}/login", json={"username": USERNAME, "password": PASSWORD}, timeout=15)
        res.raise_for_status()
        token = res.text.replace('"', "")
        headers = {"Authorization": f"Bearer {token}"}

        # === 2️⃣ Retrieve Courses ===
        keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
        payload = {"keywords": keywords_list, "keywords_logic": "or", "sources": [source]}
        all_courses = []

        for page in range(1, 51):
            url = f"{API}/courses?page={page}&page_size=100"
            res = requests.post(url, headers=headers, data=payload, timeout=60)
            if res.status_code != 200:
                break
            data = res.json()
            items = data.get("items", [])
            if not items:
                break
            all_courses.extend(items)
            if len(items) < 100:
                break

        print(f"📄 Retrieved {len(all_courses)} courses.")
        if not all_courses:
            return {"error": "No courses found for given filters."}

        # === 3️⃣ Extract ESCO Skills ===
        skill_uris = [s for d in all_courses for s in d.get("skills", []) if isinstance(s, str) and s.startswith("http")]
        unique_uris = sorted(set(skill_uris))

        # Map URIs → ESCO Labels
        id_to_label = {}
        all_esco = []
        for page in range(1, 51):
            r = requests.post(f"{API}/skills?page={page}&page_size=100", headers=headers, timeout=30)
            if r.status_code != 200:
                break
            items = r.json().get("items", [])
            if not items:
                break
            all_esco.extend(items)
            if len(items) < 100:
                break

        for s in all_esco:
            sid = s.get("id")
            label = s.get("label", "").strip().lower()
            if sid and label:
                id_to_label[sid] = label

        ESCO_skill_labels = sorted({id_to_label.get(u, u) for u in unique_uris})
        print(f"🧠 {len(ESCO_skill_labels)} unique ESCO skills mapped from courses.")

        # === 4️⃣ Load Non-ESCO (Tech) Skills ===
        csv_path = "technology_skills.csv"
        tech_df = pd.read_csv(csv_path, sep=None, engine="python", on_bad_lines="skip")
        if "Example" not in tech_df.columns:
            return {"error": "CSV must contain an 'Example' column."}
        tech_examples = []
        for row in tech_df["Example"].dropna().astype(str):
            tech_examples.extend([s.strip().lower() for s in row.replace(";", ",").split(",") if s.strip()])
        non_ESCO_skills = sorted(set(tech_examples))
        print(f"💾 Loaded {len(non_ESCO_skills)} non-ESCO (technology) skills.")

        # === 5️⃣ Build Graph via TF-IDF Similarity ===
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

        print(f"🕸️ Graph built with {len(G.nodes())} nodes and {len(G.edges())} edges.")

        if G.number_of_edges() == 0:
            return {"error": "No edges found. Try lowering similarity_threshold."}

        # === 6️⃣ Classical Link Prediction ===
        if method == "adamic_adar":
            preds = nx.adamic_adar_index(G)
        elif method == "resource_allocation":
            preds = nx.resource_allocation_index(G)
        else:
            preds = nx.jaccard_coefficient(G)

        preds_sorted = sorted(preds, key=lambda x: x[2], reverse=True)[:top_k]

        # === 7️⃣ Combine Scores + Add Emoji Confidence ===
        candidate_links = []
        for u, v, score in preds_sorted:
            common_neighbors = list(nx.common_neighbors(G, u, v))
            weighted_score = np.mean([G[u][n]["weight"] * G[v][n]["weight"] for n in common_neighbors]) if common_neighbors else 0
            combined_score = round((score + weighted_score) / 2, 3)

            if combined_score >= 0.8:
                emoji = "🟢"
                level = "High confidence"
            elif combined_score >= 0.6:
                emoji = "🟡"
                level = "Medium confidence"
            else:
                emoji = "🔴"
                level = "Low confidence"

            candidate_links.append({
                "source": u,
                "target": v,
                "predicted_score": combined_score,
                "confidence_level": level,
                "emoji": emoji
            })

        # === 8️⃣ Summary ===
        summary_counts = {
            "high": sum(1 for c in candidate_links if c["predicted_score"] >= 0.8),
            "medium": sum(1 for c in candidate_links if 0.6 <= c["predicted_score"] < 0.8),
            "low": sum(1 for c in candidate_links if c["predicted_score"] < 0.6)
        }

        # === 9️⃣ Return Response ===
        return {
            "message": "✅ ESCOPlus course-based link prediction completed.",
            "summary": {
                "Courses processed": len(all_courses),
                "Mapped ESCO skills": len(ESCO_skill_labels),
                "Non-ESCO skills": len(non_ESCO_skills),
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

@forecast_router.get("/law_predict")
def law_policies_link_prediction(
    keywords: str = Query(..., description="Comma-separated keywords (e.g. data,ai,green)"),
    source: str = Query("eur_lex", description="Source of the policies"),
    similarity_threshold: float = Query(0.7, description="Minimum similarity to consider edges"),
    top_k: int = Query(30, description="Number of top predicted links to return"),
    method: str = Query("jaccard", description="Link prediction method: jaccard | adamic_adar | resource_allocation")
):
    """
    Extend the ESCO taxonomy using policy-linked ESCO skills and non-ESCO technology skills.
    Predict new potential ESCO ↔ non-ESCO connections using classical link prediction methods.
    """

    import os, requests, pandas as pd, numpy as np, networkx as nx, traceback
    from dotenv import load_dotenv
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    try:
        # === 1️⃣ Authenticate ===
        load_dotenv()
        API = os.getenv("TRACKER_API", "https://skillab-tracker.csd.auth.gr/api")
        USERNAME = os.getenv("TRACKER_USERNAME", "")
        PASSWORD = os.getenv("TRACKER_PASSWORD", "")

        print("🔐 Authenticating with Tracker...")
        res = requests.post(f"{API}/login", json={"username": USERNAME, "password": PASSWORD}, timeout=15)
        res.raise_for_status()
        token = res.text.replace('"', "")
        headers = {"Authorization": f"Bearer {token}"}
        print("✅ Authenticated.")

        # === 2️⃣ Retrieve Policies ===
        keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
        payload = {"keywords": keywords_list, "keywords_logic": "or", "sources": [source]}
        all_docs = []
        for page in range(1, 51):
            url = f"{API}/law-policies?page={page}&page_size=100"
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

        # === 3️⃣ Extract and Map ESCO Skills ===
        skill_uris = [s for d in all_docs for s in d.get("skills", []) if isinstance(s, str) and s.startswith("http")]
        unique_uris = sorted(set(skill_uris))

        id_to_label = {}
        all_esco = []
        for page in range(1, 51):
            r = requests.post(f"{API}/skills?page={page}&page_size=100", headers=headers, timeout=30)
            if r.status_code != 200:
                break
            items = r.json().get("items", [])
            if not items:
                break
            all_esco.extend(items)
            if len(items) < 100:
                break
        for s in all_esco:
            sid = s.get("id")
            label = s.get("label", "").strip().lower()
            if sid and label:
                id_to_label[sid] = label

        ESCO_skill_labels = sorted({id_to_label.get(u, u) for u in unique_uris})
        print(f"🧠 {len(ESCO_skill_labels)} ESCO skills mapped.")

        # === 4️⃣ Load Non-ESCO Skills ===
        csv_path = "technology_skills.csv"
        tech_df = pd.read_csv(csv_path, sep=None, engine="python", on_bad_lines="skip")

        if "Example" not in tech_df.columns:
            return {"error": "CSV must contain an 'Example' column."}

        tech_examples = []
        for row in tech_df["Example"].dropna().astype(str):
            tech_examples.extend([s.strip().lower() for s in row.replace(";", ",").split(",") if s.strip()])
        non_ESCO_skills = sorted(set(tech_examples))
        print(f"💾 Loaded {len(non_ESCO_skills)} non-ESCO (technology) skills.")

        # === 5️⃣ Build Graph by Similarity ===
        corpus = list(ESCO_skill_labels) + list(non_ESCO_skills)
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4))
        tfidf = vectorizer.fit_transform(corpus)
        sim_matrix = cosine_similarity(tfidf[:len(ESCO_skill_labels)], tfidf[len(ESCO_skill_labels):])

        edges, weights = [], []
        for i, esco in enumerate(ESCO_skill_labels):
            for j, non_esco in enumerate(non_ESCO_skills):
                sim = sim_matrix[i, j]
                if sim >= similarity_threshold:
                    edges.append((esco, non_esco))
                    weights.append(sim)

        G = nx.Graph()
        for (s, t), w in zip(edges, weights):
            G.add_edge(s, t, weight=w)
        print(f"🕸️ Graph built: {len(G.nodes())} nodes, {len(G.edges())} edges")

        # === 6️⃣ Predict New Links with Weighted Adjustment ===
        print(f"🔮 Using link prediction method: {method}")
        if method == "adamic_adar":
            preds = nx.adamic_adar_index(G)
        elif method == "resource_allocation":
            preds = nx.resource_allocation_index(G)
        else:  # default = jaccard
            preds = nx.jaccard_coefficient(G)

        adjusted_preds = []
        for u, v, score in preds:
            common_neighbors = list(nx.common_neighbors(G, u, v))
            if common_neighbors:
                weighted_score = np.mean([
                    G[u][n]["weight"] * G[v][n]["weight"]
                    for n in common_neighbors
                    if G.has_edge(u, n) and G.has_edge(v, n)
                ])
            else:
                weighted_score = 0
            combined_score = (score + weighted_score) / 2
            adjusted_preds.append((u, v, combined_score))

        # sort & take top_k
        preds_sorted = sorted(adjusted_preds, key=lambda x: x[2], reverse=True)[:top_k]
        candidate_links = []
        for u, v, score in preds_sorted:
            score_rounded = round(score, 3)

            # Assign color & emoji based on confidence level
            if score_rounded >= 0.8:
                emoji = "🟢"
                level = "High confidence"
            elif score_rounded >= 0.6:
                emoji = "🟠"
                level = "Medium confidence"
            else:
                emoji = "🔴"
                level = "Low confidence"

            candidate_links.append({
                "source": u,
                "target": v,
                "predicted_score": score_rounded,
                "confidence_level": level,
                "emoji": emoji
            })

        print(f"🔗 Found {len(candidate_links)} potential new links.")

        # === 7️⃣ Return Output JSON ===
        return {
            "message": "✅ ESCOPlus classical link prediction completed.",
            "summary": {
                "Policies processed": len(all_docs),
                "Mapped ESCO skills": len(ESCO_skill_labels),
                "Non-ESCO skills": len(non_ESCO_skills),
                "Observed edges": len(G.edges()),
                "Predicted new links": len(candidate_links),
                "Method used": method
            },
            "predicted_links": candidate_links
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

# === Register routers with main app ===
app.include_router(analysis_router)
app.include_router(forecast_router)
