#
# from datetime import datetime
# from fastapi import FastAPI, Query
# import pandas as pd
# import numpy as np
# import requests
# from datetime import datetime
# from collections import defaultdict
# from itertools import combinations
# from sklearn.linear_model import LinearRegression
# import json
# import uuid
# from pathlib import Path
# import os
# from dotenv import load_dotenv
#
# def run_skill_analysis_from_list(job_list):
#
#
#     skill_occurrences = defaultdict(list)
#     for job in job_list:
#         try:
#             date_str = job.get("upload_date")
#             if not date_str:
#                 continue
#             date = datetime.strptime(date_str, "%Y-%m-%d")
#             skills = job.get("skills", [])
#             if not isinstance(skills, list):
#                 continue
#             for skill in skills:
#                 skill_occurrences[skill].append(date)
#         except:
#             continue
#
#     biology_summary = []
#     combined_index = pd.date_range(start="2020-01-01", end="2025-12-31", freq="M")
#
#     def get_slope(ts):
#         if len(ts) < 3:
#             return 0
#         X = np.arange(len(ts)).reshape(-1, 1)
#         y = ts.values.reshape(-1, 1)
#         model = LinearRegression().fit(X, y)
#         return model.coef_[0][0]
#
#     for skill, dates in skill_occurrences.items():
#         df = pd.DataFrame(dates, columns=["date"])
#         df["year_month"] = df["date"].dt.to_period("M")
#         birth = df["date"].min()
#         peak = df["year_month"].value_counts().idxmax()
#         total_jobs = len(df)
#         recent_use = df["date"].max().year > 2022
#         immunity = "High" if total_jobs > 20 and recent_use else "Low"
#
#         # === New: Compute slope (trend) ===
#         s = pd.Series(1, index=pd.to_datetime(dates))
#         s = s.resample("M").sum().reindex(combined_index, fill_value=0)
#         slope = get_slope(s)
#
#         if slope < -0.01:
#             trend = "Declining"
#         elif slope > 0.01:
#             trend = "Rising"
#         else:
#             trend = "Stable"
#
#         biology_summary.append({
#             "Skill": skill,
#             "Date of Birth": birth.strftime("%Y-%m-%d"),
#             "Peak Activity Date": str(peak),
#             "Total Jobs": total_jobs,
#             "Immunity Score": immunity,
#             "Trend": trend,                 # 👈 NEW FIELD
#             "Slope": round(slope, 4)        # 👈 Optional numeric slope
#         })
#
#
#     # === Time Series Construction ===
#     tag_series = {}
#     combined_index = pd.date_range(start="2020-01-01", end="2025-12-31", freq="M")
#     for skill, dates in skill_occurrences.items():
#         s = pd.Series(1, index=pd.to_datetime(dates))
#         s = s.resample("M").sum().reindex(combined_index, fill_value=0)
#         tag_series[skill] = s
#
#     all_tags_df = pd.DataFrame(tag_series).fillna(0)
#     filtered_tags = [tag for tag in all_tags_df.columns if all_tags_df[tag].sum() >= 10]
#
#     # === Competing Skills (Negative Correlation) ===
#     competing_results = []
#     for tag1, tag2 in combinations(filtered_tags, 2):
#         s1, s2 = all_tags_df[tag1], all_tags_df[tag2]
#         overlap = (s1 > 0) & (s2 > 0)
#         if overlap.sum() < 5:
#             continue
#         corr = s1[overlap].corr(s2[overlap])
#         if pd.notna(corr) and corr < -0.5:
#             competing_results.append({
#                 "Skill A": tag1,
#                 "Skill B": tag2,
#                 "Correlation": round(corr, 3)
#             })
#
#     # === Inverse Trends ===
#     def get_slope(ts):
#         X = np.arange(len(ts)).reshape(-1, 1)
#         y = ts.values.reshape(-1, 1)
#         return LinearRegression().fit(X, y).coef_[0][0]
#
#     inverse_results = []
#     top_skills = sorted(filtered_tags, key=lambda x: all_tags_df[x].sum(), reverse=True)[:100]
#     for tag1, tag2 in combinations(top_skills, 2):
#         s1, s2 = all_tags_df[tag1], all_tags_df[tag2]
#         mask = (s1 > 0) & (s2 > 0)
#         if mask.sum() < 6:
#             continue
#         slope1, slope2 = get_slope(s1[mask]), get_slope(s2[mask])
#         if slope1 < -0.005 and slope2 > 0.005:
#             inverse_results.append({
#                 "Declining Skill": tag1,
#                 "Competing Skill": tag2,
#                 "Slope A": round(slope1, 4),
#                 "Slope B": round(slope2, 4),
#                 "Overlapping Months": int(mask.sum())
#             })
#
#     # === Rapid Obsolescence Detection ===
#     rapid_drops = []
#     for tag, series in tag_series.items():
#         if (series > 0).sum() < 12:
#             continue
#         peak_value = series.max()
#         if peak_value < 5:
#             continue
#         peak_idx = series.idxmax()
#         peak_loc = series.index.get_loc(peak_idx)
#         post_peak = series.iloc[peak_loc:peak_loc + 7]
#         drop_ratio = (peak_value - post_peak.min()) / peak_value
#         if drop_ratio >= 0.3:
#             rapid_drops.append({
#                 "Skill": tag,
#                 "Peak Month": peak_idx.strftime("%Y-%m"),
#                 "Peak Value": int(peak_value),
#                 "Min Value After Peak": int(post_peak.min()),
#                 "Drop %": round(drop_ratio * 100, 2)
#             })
#
#     # === Epidemiological Metrics ===
#     epi_metrics = []
#     shock_start, shock_end = pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31")
#     old_start, old_end = pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")
#
#     for tag, series in tag_series.items():
#         series = series.fillna(0)
#         total_jobs = series.sum()
#         if total_jobs == 0:
#             continue
#
#         incidence = series.loc[shock_start:shock_end].sum()
#         old_incidence = series.loc[old_start:old_end].sum()
#         if old_incidence > 0:
#             pct_change = 100 * (incidence - old_incidence) / old_incidence
#         elif incidence > 0:
#             pct_change = 999
#         else:
#             pct_change = 0
#
#         ip_ratio = incidence / total_jobs if total_jobs else 0
#         recent_activity = series[series.index >= datetime(2023, 7, 1)].sum()
#         is_dead = recent_activity == 0
#         revival = "Yes" if old_incidence < incidence and old_incidence > 0 else "No"
#         mortality_ratio = incidence / (total_jobs - incidence) if (total_jobs - incidence) > 0 else 999
#         was_active = old_incidence > 0 or incidence > 0
#         cfr = 1.0 if (was_active and is_dead) else 0.0
#         attack_rate = (series > 0).sum() / len(series)
#
#         epi_metrics.append({
#             "Skill": tag,
#             "Total Jobs": int(total_jobs),
#             "Incidence (2023)": int(incidence),
#             "Incidence (2022)": int(old_incidence),
#             "% Change in Incidence": round(pct_change, 2),
#             "Incidence : Prevalence": round(ip_ratio, 4),
#             "Mortality Risk": "☠️" if is_dead else "🟢",
#             "Revived?": revival,
#             "Incidence : Mortality Ratio": round(mortality_ratio, 2),
#             "CFR": round(cfr, 2),
#             "Attack Rate": round(attack_rate, 4)
#         })
#
#     # === Save results ===
#     output = {
#         "skill_biology_summary": biology_summary,
#         "competing_skills": competing_results,
#         "inverse_trends": inverse_results,
#         "rapid_obsolescence": rapid_drops,
#         "epidemiological_metrics": epi_metrics,
#         "total_jobs_analyzed": len(job_list)
#     }
#
#     Path("results").mkdir(parents=True, exist_ok=True)
#     filename = f"results/ku_skill_analysis_{uuid.uuid4().hex[:6]}.json"
#     with open(filename, "w", encoding="utf-8") as f:
#         json.dump(output, f, ensure_ascii=False, indent=2)
#
#     return {
#         "message": "✅ KU Skill analysis complete",
#         "file_saved": filename,
#         "summary": {
#             "Total Skills Found": len(biology_summary),
#             "Competing Skill Pairs": len(competing_results),
#             "Inverse Trend Pairs": len(inverse_results),
#             "Rapidly Obsolete Skills": len(rapid_drops),
#             "Epidemiological Metrics": len(epi_metrics),
#         },
#         "data": output
#     }

# analysis.py
import numpy as np
import networkx as nx
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_skill_similarity(
    esco_skills,
    non_esco_skills,
    similarity_threshold=0.8
):
    """
    Compute TF-IDF character n-gram similarity between ESCO and non-ESCO skills.
    """
    corpus = list(esco_skills) + list(non_esco_skills)

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4))
    tfidf = vectorizer.fit_transform(corpus)

    esco_emb = tfidf[:len(esco_skills)]
    non_esco_emb = tfidf[len(esco_skills):]

    sim_matrix = cosine_similarity(esco_emb, non_esco_emb)

    matches = []
    for i, esco in enumerate(esco_skills):
        j = np.argmax(sim_matrix[i])
        score = sim_matrix[i, j]
        if score >= similarity_threshold:
            matches.append({
                "ESCO_skill": esco,
                "non_ESCO_skill": non_esco_skills[j],
                "similarity": round(float(score), 3)
            })

    return matches


def compute_confidence(matches, frequency_dict):
    """
    Apply frequency-weighted confidence score.
    """
    for m in matches:
        freq = frequency_dict.get(m["ESCO_skill"], 1)
        m["confidence"] = round(
            m["similarity"] * (1 + np.log1p(freq) / 10), 3
        )
    return matches


def build_skill_network(matches):
    """
    Build bipartite ESCO ↔ non-ESCO graph.
    """
    G = nx.Graph()
    for m in matches:
        e = m["ESCO_skill"]
        n = m["non_ESCO_skill"]
        G.add_node(e, group="ESCO_skill")
        G.add_node(n, group="non_ESCO_skill")
        G.add_edge(e, n, similarity=m["similarity"], confidence=m.get("confidence", 0))
    return G


def compute_network_metrics(G):
    """
    Compute global and node-level network metrics.
    """
    if G.number_of_edges() == 0:
        return {}

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "avg_similarity": round(
            np.mean([d["similarity"] for _, _, d in G.edges(data=True)]), 3
        ),
        "avg_clustering": round(nx.average_clustering(G), 3),
        "largest_component": len(max(nx.connected_components(G), key=len)),
    }
