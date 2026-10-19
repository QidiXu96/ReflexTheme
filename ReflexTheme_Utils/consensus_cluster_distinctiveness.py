import json, re, os, glob, time
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

# Build one agent's co-assignment matrix
def build_coassignment_matrix(code_ids, themes):
    """
    code_ids = ["C1", "C2", "C3", "C4"]
    themes = [{
        "theme_name": "Communication",
        "related_initial_codes": ["C1", "C2"]
    }]
    """
    index = {code_id: i for i, code_id in enumerate(code_ids)} # code id: index
    M = np.eye(len(code_ids), dtype=float) # diagonal value = 1
    for theme in themes:
        theme_code_ids = [str(x).strip() for x in theme.get("related_initial_codes", []) if str(x).strip() in index]
        idxs = [index[code_id] for code_id in theme_code_ids]
        if idxs:
            M[np.ix_(idxs, idxs)] = 1.0 
    return M

def get_mean_theme_count_for_cluster(
    cluster_id,
    final_agent_results
):
    theme_counts = {}
    for agent_name, agent_data in final_agent_results.items():
        cluster_result = next((x for x in agent_data["final_results"] if int(x["cluster_id"]) == int(cluster_id)), None)
        if cluster_result is None:
            continue
        themes = (cluster_result.get("codebook", {}).get("candidate_themes", [])) or []
        theme_counts[agent_name] = len(themes)

    if not theme_counts:
        raise ValueError(f"No agent themes found for cluster {cluster_id}")

    counts = list(theme_counts.values())
    mean_value = float(np.mean(counts))
    k_consensus = int(np.floor(mean_value + 0.5))
    k_consensus = max(1, k_consensus)
    return k_consensus, theme_counts, mean_value

# Convert average co-assignment matrix into consensus groups
def build_consensus_code_groups(
    code_ids,
    consensus_matrix,
    n_clusters
):
    """
    Convert the average co-assignment matrix into consensus code groups.

    Uses:
        distance = 1 - co-assignment similarity
        average linkage
        fixed number of clusters = mean number of agent themes
    """
    if len(code_ids) == 0:
        return []
    if len(code_ids) == 1:
        return [{
            "group_id": 1,
            "code_ids": [code_ids[0]]
        }]

    n_clusters = max(1,min(int(n_clusters), len(code_ids)))
    distance_matrix = (1.0 - np.asarray(consensus_matrix,dtype=float))
    np.fill_diagonal(distance_matrix,0.0)
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="average",
        metric="precomputed"
    )
    labels = clustering.fit_predict(distance_matrix)
    groups = defaultdict(list)
    for code_id, label in zip(code_ids, labels):
        groups[int(label)].append(code_id)
    group_values = sorted(groups.values(), key=lambda x: (-len(x), x[0]))
    return [
        {
            "group_id": i,
            "code_ids": group_codes
        }
        for i, group_codes in enumerate(group_values, start=1)
    ]

# Compute how well one agent theme matches one consensus group.
def _theme_group_code_fit(group_codes, theme_codes):
    group_codes = set(group_codes)
    theme_codes = set(theme_codes)

    if not group_codes or not theme_codes:
        return None

    overlap = group_codes & theme_codes
    if not overlap:
        return None

    n_overlap = len(overlap)
    group_coverage = n_overlap / len(group_codes)
    theme_coverage = n_overlap / len(theme_codes)
    code_f1 = 2 * n_overlap / (len(group_codes) + len(theme_codes))

    return {
        "n_overlap": n_overlap,
        "group_coverage": float(group_coverage),
        "theme_coverage": float(theme_coverage),
        "code_f1": float(code_f1)
    }

# Final theme decision — many-to-one + distinctiveness-aware joint selection
def _candidate_key(candidate):
    """Unique source-theme key within one original code cluster."""
    return (candidate["agent"], int(candidate["theme_id"]))

def _theme_text(candidate):
    return f'Theme: {candidate["theme_name"]}. Definition: {candidate["definition"]}'

def _normalized_redundancy(similarity, similarity_threshold=0.75):
    """
    Convert cosine similarity into a [0, 1] redundancy penalty.

    similarity <= tau -> 0 penalty
    similarity == 1   -> 1 penalty
    """
    tau = float(similarity_threshold)
    if not 0.0 <= tau < 1.0:
        raise ValueError("similarity_threshold must be in [0, 1).")

    s = float(np.clip(similarity, -1.0, 1.0))
    if s <= tau:
        return 0.0
    return float((s - tau) / (1.0 - tau))

def _build_candidate_similarity_lookup(group_infos, semantic_model):
    """Embed each unique eligible source theme and calculate pairwise similarity.
       Work on all generated code groups.
    """
    unique_candidates = {} # key is (agent name, theme id), value is theme name, definition

    for info in group_infos:
        for candidate in info.get("eligible", []):
            unique_candidates.setdefault(_candidate_key(candidate), candidate)

    keys = list(unique_candidates.keys())
    if not keys:
        return {}, unique_candidates

    texts = [_theme_text(unique_candidates[key]) for key in keys]
    embeddings = semantic_model.encode(texts, normalize_embeddings=True)
    sim_matrix = cosine_similarity(embeddings)

    lookup = {
        key_i: {key_j: float(sim_matrix[i, j]) for j, key_j in enumerate(keys)}
        for i, key_i in enumerate(keys)
    }

    return lookup, unique_candidates

def _score_selected_candidates(
    selected_candidates,
    similarity_lookup,
    similarity_threshold=0.75,
    lambda_distinctiveness=0.10
):
    """
    Quality = mean code F1 across selected consensus groups.
    Redundancy = for each UNIQUE selected source theme, find its largest normalized redundancy with another unique selected source theme, then average those maxima.
    Score = Quality - lambda_distinctiveness * Redundancy
    """
    if not selected_candidates:
        return {
            "quality": 0.0,
            "redundancy": 0.0,
            "score": 0.0,
            "per_theme_redundancy": {}
        }

    quality = float(np.mean([x["code_f1"] for x in selected_candidates]))

    unique_selected = {}
    for candidate in selected_candidates:
        unique_selected.setdefault(_candidate_key(candidate), candidate)

    keys = list(unique_selected.keys())
    per_theme_redundancy = {}

    if len(keys) < 2:
        redundancy = 0.0
        for key in keys:
            per_theme_redundancy[key] = {
                "max_redundancy": 0.0,
                "closest_theme_key": None,
                "closest_similarity": None
            }
    else:
        max_penalties = []

        for key_i in keys:
            best_penalty = -1.0
            best_key = None
            best_similarity = None

            for key_j in keys:
                if key_i == key_j:
                    continue

                similarity = similarity_lookup[key_i][key_j]
                penalty = _normalized_redundancy(
                    similarity,
                    similarity_threshold=similarity_threshold
                )

                if penalty > best_penalty:
                    best_penalty = penalty
                    best_key = key_j
                    best_similarity = similarity

            best_penalty = max(0.0, float(best_penalty))
            max_penalties.append(best_penalty)
            per_theme_redundancy[key_i] = {
                "max_redundancy": best_penalty,
                "closest_theme_key": best_key,
                "closest_similarity": (
                    float(best_similarity)
                    if best_similarity is not None
                    else None
                )
            }

        redundancy = float(np.mean(max_penalties))

    score = float(quality - float(lambda_distinctiveness) * redundancy)

    return {
        "quality": quality,
        "redundancy": redundancy,
        "score": score,
        "per_theme_redundancy": per_theme_redundancy
    }

def _beam_search_joint_selection(
    group_infos,
    similarity_lookup,
    similarity_threshold=0.75,
    lambda_distinctiveness=0.10,
    beam_width=100
):
    selectable_groups = [x for x in group_infos if x.get("eligible")]

    if not selectable_groups:
        return {}, {
            "n_selectable_groups": 0,
            "quality": None,
            "redundancy": None,
            "global_score": None,
            "beam_width": int(beam_width)
        }

    beam_width = max(1, int(beam_width))
    beam = [{
        "selections": [],
        "metrics": {
            "quality": 0.0,
            "redundancy": 0.0,
            "score": 0.0,
            "per_theme_redundancy": {}
        }
    }]

    for info in selectable_groups:
        expanded = []
        for state in beam:
            for candidate in info["eligible"]:
                new_selections = state["selections"] + [candidate]
                metrics = _score_selected_candidates(
                    new_selections,
                    similarity_lookup=similarity_lookup,
                    similarity_threshold=similarity_threshold,
                    lambda_distinctiveness=lambda_distinctiveness
                )

                # Structural tie-breakers preserve deterministic preference for better code fit when global scores are effectively equal.
                mean_group_coverage = float(np.mean([x["group_coverage"] for x in new_selections]))
                mean_theme_coverage = float(np.mean([x["theme_coverage"] for x in new_selections]))

                expanded.append({
                    "selections": new_selections,
                    "metrics": metrics,
                    "mean_group_coverage": mean_group_coverage,
                    "mean_theme_coverage": mean_theme_coverage
                })

        expanded.sort(
            key=lambda state: (
                state["metrics"]["score"],
                state["metrics"]["quality"],
                -state["metrics"]["redundancy"],
                state["mean_group_coverage"],
                state["mean_theme_coverage"]
            ),
            reverse=True
        )
        beam = expanded[:beam_width]

    best_state = beam[0]
    selected_by_group = {info["group_id"]: candidate for info, candidate in zip(selectable_groups, best_state["selections"])}

    metrics = best_state["metrics"]
    summary = {
        "n_selectable_groups": len(selectable_groups),
        "n_unique_selected_source_themes": len({
            _candidate_key(x) for x in best_state["selections"]
        }),
        "quality": float(metrics["quality"]),
        "redundancy": float(metrics["redundancy"]),
        "global_score": float(metrics["score"]),
        "similarity_threshold": float(similarity_threshold),
        "lambda_distinctiveness": float(lambda_distinctiveness),
        "beam_width": int(beam_width)
    }

    return selected_by_group, summary

def select_final_themes_many_to_one(
    cluster_id,
    consensus_groups,
    final_agent_results,
    semantic_model,
    f1_tolerance=0.10,
    similarity_threshold=0.75,
    lambda_distinctiveness=0.10,
    beam_width=100
):
    """
    Select final themes for all consensus groups in one original code cluster.

    Step 1: For each group, keep the best matching theme from each agent.
    Step 2: Remove candidates whose code F1 falls outside f1_tolerance.
    Step 3: Jointly select one candidate per group with beam search to maximize mean F1 - lambda * nearest-neighbor redundancy.
    Step 4: Preserve many-to-one behavior by merging consensus groups that selected the exact same source agent theme.
    """
    # 1. Collect themes by agent for this cluster.
    candidates_by_agent = {}
    for agent_name, agent_data in final_agent_results.items():
        cluster_result = next((x for x in agent_data.get("final_results", []) if int(x.get("cluster_id")) == int(cluster_id)), None)
        if cluster_result is None:
            continue
        themes = (cluster_result.get("codebook", {}).get("candidate_themes", [])) or []

        agent_candidates = []
        for theme_id, theme in enumerate(themes, start=1):
            code_ids = [str(x).strip() for x in theme.get("related_initial_codes", []) if str(x).strip()]
            agent_candidates.append({
                "agent": agent_name,
                "theme_id": theme_id,
                "theme_name": str(theme.get("theme_name", "")).strip(),
                "definition": str(theme.get("definition", "")).strip(),
                "representative_quotes": (theme.get("representative_quotes", []) or []),
                "code_ids": code_ids
            })

        candidates_by_agent[agent_name] = agent_candidates

    # 2. Build all per-group candidate pools.
    group_infos = []

    for group in consensus_groups:
        group_id = group.get("group_id")
        group_codes = list(group.get("code_ids", []))
        per_agent_best = []

        for agent_name, agent_candidates in candidates_by_agent.items():
            best = None
            best_score = None

            for theme in agent_candidates:
                fit = _theme_group_code_fit(group_codes, theme["code_ids"])
                if fit is None:
                    continue
                score = (
                    fit["code_f1"],
                    fit["group_coverage"],
                    fit["theme_coverage"]
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best = {**theme, **fit}
            if best is not None:
                per_agent_best.append(best)
        if not per_agent_best:
            group_infos.append({
                "group_id": group_id,
                "code_ids": group_codes,
                "status": "no_matching_agent_theme",
                "max_f1": None,
                "min_eligible_f1": None,
                "per_agent_best": [],
                "eligible": []
            })
            continue
        max_f1 = max(x["code_f1"] for x in per_agent_best)
        min_eligible_f1 = max_f1 - float(f1_tolerance)
        eligible = [x for x in per_agent_best if x["code_f1"] >= min_eligible_f1]

        group_infos.append({
            "group_id": group_id,
            "code_ids": group_codes,
            "status": "eligible_candidates_found",
            "max_f1": float(max_f1),
            "min_eligible_f1": float(min_eligible_f1),
            "per_agent_best": per_agent_best,
            "eligible": eligible
        })

    # 3. Embed all unique eligible candidates once.
    similarity_lookup, unique_candidates = _build_candidate_similarity_lookup(
        group_infos=group_infos,
        semantic_model=semantic_model
    )

    # 4. Jointly optimize all consensus groups in this cluster.
    selected_by_group, optimization_summary = _beam_search_joint_selection(
        group_infos=group_infos,
        similarity_lookup=similarity_lookup,
        similarity_threshold=similarity_threshold,
        lambda_distinctiveness=lambda_distinctiveness,
        beam_width=beam_width
    )

    # Re-score the final chosen set so we can attach closest-theme diagnostics.
    selected_candidates = [
        selected_by_group[info["group_id"]]
        for info in group_infos
        if info["group_id"] in selected_by_group
    ]
    final_metrics = _score_selected_candidates(
        selected_candidates,
        similarity_lookup=similarity_lookup,
        similarity_threshold=similarity_threshold,
        lambda_distinctiveness=lambda_distinctiveness
    )

    # 5. Build per-group selection details.
    group_selection_details = []

    for info in group_infos:
        group_id = info["group_id"]

        if group_id not in selected_by_group:
            group_selection_details.append({
                "group_id": group_id,
                "code_ids": info["code_ids"],
                "status": info["status"],
                "per_agent_best": [],
                "eligible_candidates": [],
                "selected_theme": None
            })
            continue

        selected = selected_by_group[group_id]
        selected_key = _candidate_key(selected)
        redundancy_info = final_metrics["per_theme_redundancy"].get(
            selected_key,
            {
                "max_redundancy": 0.0,
                "closest_theme_key": None,
                "closest_similarity": None
            }
        )

        closest_key = redundancy_info["closest_theme_key"]
        closest_candidate = (
            unique_candidates.get(closest_key)
            if closest_key is not None
            else None
        )

        group_selection_details.append({
            "group_id": group_id,
            "code_ids": info["code_ids"],
            "max_code_f1": float(info["max_f1"]),
            "f1_tolerance": float(f1_tolerance),
            "minimum_eligible_f1": float(info["min_eligible_f1"]),
            "per_agent_best": [
                {
                    "agent": x["agent"],
                    "theme_id": x["theme_id"],
                    "theme_name": x["theme_name"],
                    "definition": x["definition"],
                    "theme_code_ids": x["code_ids"],
                    "code_f1": x["code_f1"],
                    "group_coverage": x["group_coverage"],
                    "theme_coverage": x["theme_coverage"]
                }
                for x in info["per_agent_best"]
            ],
            "eligible_candidates": [
                {
                    "agent": x["agent"],
                    "theme_id": x["theme_id"],
                    "theme_name": x["theme_name"],
                    "definition": x["definition"],
                    "code_f1": x["code_f1"],
                    "group_coverage": x["group_coverage"],
                    "theme_coverage": x["theme_coverage"]
                }
                for x in info["eligible"]
            ],
            "selected_theme": {
                "agent": selected["agent"],
                "theme_id": selected["theme_id"],
                "theme_name": selected["theme_name"],
                "code_f1": selected["code_f1"],
                "group_coverage": selected["group_coverage"],
                "theme_coverage": selected["theme_coverage"],
                "max_redundancy": float(
                    redundancy_info["max_redundancy"]
                ),
                "closest_selected_theme": (
                    {
                        "agent": closest_candidate["agent"],
                        "theme_id": closest_candidate["theme_id"],
                        "theme_name": closest_candidate["theme_name"],
                        "cosine_similarity": float(
                            redundancy_info["closest_similarity"]
                        )
                    }
                    if closest_candidate is not None
                    else None
                )
            }
        })

    # 6. Many-to-one: merge groups selecting the exact same source agent theme.
    merged = {}

    for info in group_infos:
        group_id = info["group_id"]
        if group_id not in selected_by_group:
            continue

        selected = selected_by_group[group_id]
        key = _candidate_key(selected)

        if key not in merged:
            merged[key] = {
                "theme_name": selected["theme_name"],
                "definition": selected["definition"],
                "representative_quotes": selected["representative_quotes"],
                "source_agent": selected["agent"],
                "source_theme_id": selected["theme_id"],
                "source_code_ids": selected["code_ids"],
                "consensus_group_ids": [],
                "code_ids": []
            }

        merged[key]["consensus_group_ids"].append(group_id)
        merged[key]["code_ids"].extend(info["code_ids"])

    # 7. Final themes remain exact agent-generated name/definition/quotes.
    final_themes = []

    for final_theme_id, item in enumerate(merged.values(), start=1):
        code_ids = list(dict.fromkeys(item["code_ids"]))
        key = (item["source_agent"], int(item["source_theme_id"]))
        redundancy_info = final_metrics["per_theme_redundancy"].get(
            key,
            {
                "max_redundancy": 0.0,
                "closest_theme_key": None,
                "closest_similarity": None
            }
        )

        final_themes.append({
            "final_theme_id": final_theme_id,
            "theme_name": item["theme_name"],
            "definition": item["definition"],
            "code_ids": code_ids,
            "n_codes": len(code_ids),
            "representative_quotes": item["representative_quotes"],
            "source_agent": item["source_agent"],
            "source_theme_id": item["source_theme_id"],
            "source_code_ids": item["source_code_ids"],
            "consensus_group_ids": item["consensus_group_ids"],
            "max_redundancy_with_selected_theme": float(
                redundancy_info["max_redundancy"]
            )
        })

    return final_themes, group_selection_details, optimization_summary


def run_final_theme_consensus(
    final_clusters_path,
    final_agent_results,
    output_dir,
    selection_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    f1_tolerance=0.10,
    similarity_threshold=0.75,
    lambda_distinctiveness=0.10,
    beam_width=100
):
    if float(f1_tolerance) < 0:
        raise ValueError("f1_tolerance must be >= 0.")
    if not 0.0 <= float(similarity_threshold) < 1.0:
        raise ValueError("similarity_threshold must be in [0, 1).")
    if float(lambda_distinctiveness) < 0:
        raise ValueError("lambda_distinctiveness must be >= 0.")
    if int(beam_width) < 1:
        raise ValueError("beam_width must be >= 1.")

    with open(final_clusters_path, "r", encoding="utf-8") as f:
        cluster_data = json.load(f)

    semantic_model = SentenceTransformer(selection_model_name)

    consensus_clusters = []

    for cluster in cluster_data["clusters"]:
        cluster_id = int(cluster["cluster_id"])
        code_ids = [str(item["code_id"]).strip() for item in cluster["items"]]

        # Agent co-assignment matrices
        agent_matrices = {}
        for agent_name, agent_data in final_agent_results.items():
            cluster_result = next((x for x in agent_data.get("final_results", []) if int(x.get("cluster_id")) == cluster_id), None)

            if cluster_result is None:
                raise ValueError(f"Missing cluster {cluster_id} for {agent_name}")

            themes = (cluster_result.get("codebook", {}).get("candidate_themes", [])) or []
            agent_matrices[agent_name] = build_coassignment_matrix(code_ids,themes)

        # Average agents
        consensus_matrix = np.mean(np.stack(list(agent_matrices.values()), axis=0), axis=0)

        # Mean K from agent theme counts for this cluster
        (
            k_consensus,
            agent_theme_counts,
            mean_theme_count
        ) = get_mean_theme_count_for_cluster(
            cluster_id=cluster_id,
            final_agent_results=final_agent_results
        )

        # Consensus code groups: fixed mean K + average linkage
        consensus_groups = build_consensus_code_groups(
            code_ids=code_ids,
            consensus_matrix=consensus_matrix,
            n_clusters=k_consensus
        )

        # joint F1 + distinctiveness-aware selection across all consensus groups in this original code cluster.
        (
            final_themes,
            group_selection_details,
            optimization_summary
        ) = select_final_themes_many_to_one(
            cluster_id=cluster_id,
            consensus_groups=consensus_groups,
            final_agent_results=final_agent_results,
            semantic_model=semantic_model,
            f1_tolerance=f1_tolerance,
            similarity_threshold=similarity_threshold,
            lambda_distinctiveness=lambda_distinctiveness,
            beam_width=beam_width
        )

        consensus_clusters.append({
            "cluster_id": cluster_id,
            "code_ids": code_ids,
            "agent_theme_counts": agent_theme_counts,
            "mean_theme_count": mean_theme_count,
            "k_consensus": k_consensus,
            "agent_coassignment_matrices": {
                agent: matrix.astype(int).tolist()
                for agent, matrix in agent_matrices.items()
            },
            "consensus_matrix": consensus_matrix.tolist(),
            "consensus_code_groups": consensus_groups,
            "selection_optimization": optimization_summary,
            "consensus_group_theme_selection": group_selection_details,
            "final_themes": final_themes
        })

    result = {
        "n_agents": len(final_agent_results),
        "agents": list(final_agent_results.keys()),
        "consensus_group_method": "mean_k_average_linkage",
        "final_theme_selection_method": (
            "per_agent_best_f1_filter_then_joint_beam_search_"
            "quality_minus_nearest_neighbor_redundancy"
        ),
        "selection_model_name": selection_model_name,
        "f1_tolerance": float(f1_tolerance),
        "similarity_threshold": float(similarity_threshold),
        "lambda_distinctiveness": float(lambda_distinctiveness),
        "beam_width": int(beam_width),
        "clusters": consensus_clusters
    }

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir,"final_theme_consensus.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result, output_path

def extract_final_theme_names(consensus_results):
    return [
        theme["theme_name"] for cluster in consensus_results["clusters"] for theme in cluster["final_themes"]if str(theme.get("theme_name", "")).strip()
    ]

def theme_similarity_to_human(
    human_themes,
    predicted_themes,
    model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
):
    human_themes = [str(x).strip()for x in human_themes if str(x).strip()]
    predicted_themes = [str(x).strip() for x in predicted_themes if str(x).strip()]
    if not human_themes or not predicted_themes:
        return None

    human_emb = model.encode(human_themes,normalize_embeddings=True)
    pred_emb = model.encode(predicted_themes,normalize_embeddings=True)
    similarity = cosine_similarity(human_emb, pred_emb)

    best_idx = similarity.argmax(axis=1)
    best_scores = similarity[np.arange(len(human_themes)),best_idx]
    details = []
    for i, j in enumerate(best_idx):
        details.append({"human_theme":human_themes[i],
                        "best_predicted_theme":predicted_themes[int(j)],
                        "similarity":round(float(best_scores[i]), 3)
        })

    return {
        "mean_similarity": round(float(np.mean(best_scores)), 3),
        "details":details
    }

def mean_pairwise_distance(
    texts,
    model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
):
    texts = [str(x).strip() for x in texts if str(x).strip()]
    if len(texts) < 2:
        return None
    embeddings = model.encode(texts, normalize_embeddings=True)
    similarity = cosine_similarity(embeddings)
    upper = np.triu_indices(len(texts),k=1)
    distances = (1.0 - similarity[upper])
    return round(float(np.mean(distances)),3)

def calculate_theme_distinctiveness(
    final_themes,
    model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
):
    return mean_pairwise_distance(final_themes, model)

def mean_pairwise_similarity(
    texts, 
    model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
):
    texts = [str(x).strip() for x in texts if str(x).strip()]
    if len(texts) < 2:
        return None
    embeddings = model.encode(texts, normalize_embeddings=True)
    similarity = cosine_similarity(embeddings)
    upper = np.triu_indices(len(texts), k=1)
    return float(np.mean(similarity[upper]))

def calculate_theme_coherence(
    consensus_results,
    final_clusters_path,
    model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
):
    # 1. Build code_id -> code_name map
    with open(final_clusters_path, "r", encoding="utf-8") as f:
        cluster_data = json.load(f)
    code_id_to_name = {}
    for cluster in cluster_data.get("clusters", []):
        for item in cluster.get("items", []):
            code_id = str(item.get("code_id", "")).strip()
            code_name = str(item.get("initial code", "")).strip()
            if code_id and code_name:
                code_id_to_name[code_id] = code_name

    # 2. S_all: average pairwise similarity among ALL generated codes
    all_code_names = list(code_id_to_name.values())
    overall_similarity = mean_pairwise_similarity(all_code_names, model)

    # 3. S_k: within-theme similarity for each final theme
    details = []
    within_similarities = []

    for cluster in consensus_results.get("clusters", []):
        cluster_id = cluster.get("cluster_id")
        for theme in cluster.get("final_themes", []):
            code_ids = [str(x).strip() for x in theme.get("code_ids", []) if str(x).strip()]
            code_names = [code_id_to_name[x] for x in code_ids if x in code_id_to_name]

            within_similarity = mean_pairwise_similarity(code_names, model)

            if within_similarity is not None:
                within_similarities.append(within_similarity)

            details.append({
                "cluster_id": cluster_id,
                "theme_name": theme.get("theme_name"),
                "n_codes": len(code_names),
                "code_names": code_names,
                "within_similarity": within_similarity
            })

    # 4. S_within: equally weighted average across themes
    mean_within_similarity = (float(np.mean(within_similarities)) if within_similarities else None)
    # 5. Normalized coherence = S_within / S_all
    normalized_coherence = None
    if (
        mean_within_similarity is not None
        and overall_similarity is not None
        and overall_similarity < 1 # and overall_similarity != 0
    ):
        #normalized_coherence = mean_within_similarity / overall_similarity
        normalized_coherence = (mean_within_similarity - overall_similarity)/(1 - overall_similarity)

    return {
        "n_final_themes": len(details),
        "overall_similarity": overall_similarity,
        "mean_within_similarity": round(mean_within_similarity, 3),
        "normalized_coherence": round(normalized_coherence, 3),
        "details": details
    }

def evaluate_final_themes(
    consensus_results,
    final_clusters_path,
    human_themes,
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
):

    model = SentenceTransformer(model_name)
    final_themes = extract_final_theme_names(consensus_results)
    human_similarity = theme_similarity_to_human(human_themes,final_themes,model)['mean_similarity']
    distinctiveness = calculate_theme_distinctiveness(final_themes,model)
    coherence = calculate_theme_coherence(consensus_results,final_clusters_path,model)['mean_within_similarity']
    n_coherence = calculate_theme_coherence(consensus_results,final_clusters_path,model)['normalized_coherence']
    return {
        "final_themes":final_themes,
        "n_final_themes":len(final_themes),
        "human_theme_similarity":human_similarity,
        "distinctiveness":distinctiveness,
        "coherence":coherence,
        "n_coherence":n_coherence
    }