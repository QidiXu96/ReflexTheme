import os, re, glob, json, time, copy
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from coder_agent import *
from AzureUtils import *
from coder_collaboration_reflection import *
from dedup import *
from coding_utils import *

# get transcript id and chunk id
pattern = re.compile(r"^Interview_(\d+)_chunk_(\d+)\.docx$", re.IGNORECASE)
def parse_ids(fp: str):
    base = os.path.basename(fp)
    m = pattern.match(base)
    if not m:
        return None, None
    return m.group(1), m.group(2) # transcript id, chunk id

# clean all raw json coding output files 
def clean_chunk_round_outputs(coding_base_dir, agents, transcript_id, round_number):
    # coding_base_dir: independent_coding
    for agent in agents:
        agent_dir = os.path.join(coding_base_dir, agent)

        json_files = [
            f for f in glob.glob(f"{agent_dir}/round_{round_number}_{transcript_id}_*.json") # all chunks for one transcript one round
            if not f.endswith("_clean.json")
        ]
        if json_files:
            process_multiple_json_files(json_files) # output is each cleaned round_chunk_transcript file
        else:
            print(f"[{agent}] No JSON files to clean (round {round_number}).")

# merge all cleaned coding output files for one transcript and one round, across all agents.
def merge_chunk_round_outputs(coding_base_dir, final_coding_base_dir, agents, transcript_id, round_number):
    for agent in agents:
        try:
            merge_agent_results(
                agent_name=agent,
                transcript_id=transcript_id,
                round_num=round_number,
                base_dir=coding_base_dir,
                output_base_dir=final_coding_base_dir,
                suffix="_final"
            ) 
        except FileNotFoundError as e:
            print(f"Skipped merge: {agent} | {e}")

def load_prev_round_coding_output(
    coding_base_dir,
    agent_name,
    previous_round,
    transcript_id,
    chunk_id,
):
    r_file_path = os.path.join(
        coding_base_dir,
        agent_name,
        f"round_{previous_round}_{transcript_id}_chunk_{chunk_id}_clean.json",
    )

    if not os.path.exists(r_file_path):
        print(f"Warning: Missing round {previous_round} chunk file for {agent_name}: {r_file_path}")
        return ""

    try:
        with open(r_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("coding_output", "")
    except Exception as e:
        print(f"Warning: Failed reading {r_file_path} ({agent_name}): {e}")
        return "" 

# clue_pool format
def parse_transcript_and_round(fname):
    base = os.path.basename(fname)

    m = re.match(r"^clue_pool_(\d{4})(?:_round_(\d+))?\.json$", base)
    if not m:
        raise ValueError(f"Unexpected filename format: {base}")

    transcript_id = m.group(1)
    round_str = m.group(2)

    round_num = int(round_str) if round_str is not None else 1
    return transcript_id, round_num

def parse_transcript_id_from_filename(fname):
    tid, _ = parse_transcript_and_round(fname)
    return tid

# Maps transcript_id -> file_path
def build_tid_to_file_map(file_paths):
    m = {}
    for p in file_paths:
        tid = parse_transcript_id_from_filename(p)
        if tid:
            m[tid] = p
    return m

# get files for a given round
def collect_round_files(clue_pool_dir, round_num):
    round_files = []
    for fname in sorted(os.listdir(clue_pool_dir)):
        if not fname.endswith(".json"):
            continue
        _, r = parse_transcript_and_round(fname)
        if r == round_num:
            round_files.append(os.path.join(clue_pool_dir, fname))
    return round_files

# get dedup agents clues
def load_dedup_clues_from_clue_pool(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    texts = []
    for clue in data["clue_pool"]:
        t = str(clue.get("text", "")).strip()
        t = re.sub(r"\s+", " ", t)
        if t:
            texts.append(t)
    return texts

def get_embeddings(embedding_model, texts):
    return embedding_model.encode(
        texts
    )

def compute_similarity(embedding_model, cur_texts, pre_texts):
    if (not cur_texts) or (not pre_texts):
        return None
    
    C = get_embeddings(embedding_model, cur_texts)
    P = get_embeddings(embedding_model, pre_texts)

    sims = cosine_similarity(C, P)  # shape: (len(C), len(P))
    return sims

def novelty_from_matrix(sims):
    if sims is None:
        return float("nan")
    mean_max_sim = float(np.mean(np.max(sims, axis=1)))
    novelty = 1 - mean_max_sim
    return novelty

def compute_round_to_round_novelty(
    clue_pool_dir,
    cur_round,
    embedding_model
):
    if cur_round <= 1:
        return None  # no previous round to compare

    cur_files = collect_round_files(clue_pool_dir, cur_round) 
    pre_files = collect_round_files(clue_pool_dir, cur_round-1)

    if not cur_files or not pre_files:
        return None

    cur_map = build_tid_to_file_map(cur_files)
    pre_map = build_tid_to_file_map(pre_files)

    results = []

    common_tids = sorted(set(cur_map.keys()) & set(pre_map.keys()))

    for tid in common_tids:
        cur_path = cur_map[tid]
        pre_path = pre_map[tid]

        cur_texts = load_dedup_clues_from_clue_pool(cur_path)
        pre_texts = load_dedup_clues_from_clue_pool(pre_path)

        sims = compute_similarity(embedding_model, cur_texts, pre_texts) if (cur_texts and pre_texts) else None
        novel = novelty_from_matrix(sims)

        results.append({
            "transcript_id": tid,
            "cur_round": cur_round,
            "pre_round": cur_round-1,
            "novelty": float(novel),
            "cur_agents_units": len(cur_texts),
            "pre_agents_units": len(pre_texts)
        })

    df = pd.DataFrame(results)
    avg_novelty = float(df["novelty"].mean()) if not df.empty else float("nan")
    return df, avg_novelty

def compute_single_transcript_round_novelty(clue_pool_dir, transcript_id, cur_round, embedding_model):
    result = compute_round_to_round_novelty(clue_pool_dir, cur_round, embedding_model)
    if result is None:
        return None

    df, _avg = result
    if df is None:
        return None
    row = df[df["transcript_id"] == transcript_id]
    if row.empty:
        return None
    return float(row["novelty"].iloc[0])

def should_stop_by_novelty(novelty_history, threshold, k):
    """
    novelty_history:
        [novelty(2 vs 1), novelty(3 vs 2), ..., novelty(r vs r-1)]
    """
    if len(novelty_history) < k:
        return False

    return all(
        v is not None and not np.isnan(v) and v < threshold
        for v in novelty_history[-k:]
    )

def extract_score_from_text(text):
    text = str(text or "")
    match = re.search(r"Importance Score:\s*(\d+)", text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

def filter_json_files_by_score(json_paths, output_dir, min_score=4.0):
    os.makedirs(output_dir, exist_ok=True)

    summary = []

    for input_path in json_paths:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        original_clues = data.get("clue_pool", [])

        filtered_clues = []
        fixed_none_scores = 0

        for clue in original_clues:
            score = clue.get("average_score", None)

            # If average_score is None, try to recover score from text
            if score is None:
                score = extract_score_from_text(clue.get("text", ""))
                if score is not None:
                    clue["average_score"] = score
                    fixed_none_scores += 1

            # If still None or invalid, treat as 0
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = 0.0

            if score >= min_score:
                # clean clue text: remove embedded Importance Score
                clue["text"] = re.sub(
                    r"\s*Importance Score:\s*\d+\s*$",
                    "",
                    str(clue.get("text", "")),
                    flags=re.IGNORECASE
                ).strip()

                filtered_clues.append(clue)

        filtered_data = {
            k: v for k, v in data.items() if k != "clue_pool"
        }
        filtered_data["clue_pool"] = filtered_clues

        filename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, filename)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)

        summary.append({
            "file": filename,
            "original_clues": len(original_clues),
            "kept_clues": len(filtered_clues),
            "removed": len(original_clues) - len(filtered_clues),
            "fixed_none_scores": fixed_none_scores
        })

    return summary

def collect_max_round_files(base_dir):
    pattern = re.compile(r"clue_pool_(\d+)_round_(\d+)\.json")
    best = {}

    for fn in os.listdir(base_dir):
        m = pattern.match(fn)
        if not m:
            continue
        tid, rnd = m.group(1), int(m.group(2))
        if tid not in best or rnd > best[tid][0]:
            best[tid] = (rnd, os.path.join(base_dir, fn))

    selected_files = [v[1] for v in best.values()]
    return selected_files

def normalize_text(s):
    return " ".join((s or "").strip().split()).lower()

def get_embeddings_sentence_transformers(texts, embedding_model):
    emb = embedding_model.encode(texts)
    return emb

def choose_one_code_per_clue(input_path, output_path, embedding_model, sim_threshold=0.80):
    # Groups similar initial code names, finds the cluster with the most total evidence support, and selects the shortest initial code name within that cluster.
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    compact = {
        "transcript_id": data.get("transcript_id"),
        "round": data.get("round"),
        "clue_pool": []
    }

    for clue in data.get("clue_pool", []):
        codes = []
        for ev in clue.get("evidence", []):
            c = ev.get("initial code")
            if c and str(c).strip():
                codes.append(str(c).strip())

        if not codes:
            compact["clue_pool"].append({
                "clue_id": clue.get("clue_id"),
                "text": clue.get("text", ""),
                "final_initial_code": None
            })
            continue

        # Normalization
        norm_map = {} # {normalized code name: list of original code name}
        for c in codes:
            norm = normalize_text(c)
            norm_map.setdefault(norm, []).append(c)

        freq = Counter(normalize_text(c) for c in codes) # {normalized code name: number of original code name}
        unique_norms = list(freq.keys()) # normalized code names

        if len(unique_norms) == 1:
            final = max(norm_map[unique_norms[0]], key=len)
            compact["clue_pool"].append({
                "clue_id": clue.get("clue_id"),
                "text": clue.get("text", ""),
                "final_initial_code": final
            })
            continue

        rep_texts = [max(norm_map[n], key=len) for n in unique_norms] # one representative code name (shortest, since some original code name has more space) per unique concept
        embeddings = get_embeddings_sentence_transformers(rep_texts, embedding_model) # (N, D)

        sim_matrix = cosine_similarity(embeddings) # (N, N)

        clusters = [] # list of list [[1,2], [0,3]]
        visited = set()

        for i in range(len(unique_norms)):
            if i in visited:
                continue

            cluster = [i]
            visited.add(i)

            for j in range(i + 1, len(unique_norms)):
                if sim_matrix[i, j] >= sim_threshold:
                    cluster.append(j)
                    visited.add(j)

            clusters.append(cluster)

        best_cluster = None
        best_score = -1

        for cluster in clusters:
            score = sum(freq[unique_norms[i]] for i in cluster)
            if score > best_score:
                best_score = score
                best_cluster = cluster

        candidates = []
        for i in best_cluster:
            candidates.extend(norm_map[unique_norms[i]])

        final = max(candidates, key=len)

        compact["clue_pool"].append({
            "clue_id": clue.get("clue_id"),
            "text": clue.get("text", ""),
            "final_initial_code": final
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(compact, f, indent=2, ensure_ascii=False)

    return compact

def generate_compact_clue_pool(
    input_dir,
    output_dir,
    embedding_model
):
    json_files = sorted(glob.glob(os.path.join(input_dir, "*.json")))

    if not json_files:
        raise ValueError(f"No JSON files found in {input_dir}")

    os.makedirs(output_dir, exist_ok=True)
    out_paths = []

    for input_path in json_files:
        base = os.path.basename(input_path).replace(".json", "")
        output_path = os.path.join(output_dir, f"{base}_compact.json")

        choose_one_code_per_clue(
            input_path=input_path,
            output_path=output_path,
            embedding_model=embedding_model
        )

        out_paths.append(output_path)
    return out_paths

# ----------- Codebook Functions ---------- #
OPTION_CONFIGS = {
    3: {
        "name": "code_clue_threshold",
        "retrieval_mode": "code_clue",
        "review_policy": "threshold"
    }
}

def get_option_codebook_path(base_codebook_dir, option_id):
    option_name = OPTION_CONFIGS[option_id]["name"]
    option_dir = os.path.join(base_codebook_dir, f"option_{option_id}_{option_name}")
    os.makedirs(option_dir, exist_ok=True)
    return os.path.join(option_dir, "codebook.json")

### Code text
def build_code_text(code_name):
    code_name = (code_name or "").strip()
    return f"Initial Code: {code_name}"

### Code + clues text
def build_code_clue_text(code_name, clues):
    code_name = (code_name or "").strip()

    if isinstance(clues, list):
        clues_text = "\n".join([str(c).strip() for c in clues if str(c).strip()])
    else:
        clues_text = str(clues).strip()

    return f"Initial Code: {code_name}\nSupporting Quotes: {clues_text}"

### Generate embedding input text
def build_embedding_text(code_name, clues=None, retrieval_mode="code_clue"):
    if retrieval_mode == "code":
        return build_code_text(code_name)

    if retrieval_mode == "code_clue":
        return build_code_clue_text(code_name, clues or [])

    raise ValueError(f"Unknown retrieval_mode: {retrieval_mode}")

### Initialize codebook based on 1st transcript coding results
def initialize_codebook_from_compact_file(
    compact_json_path, # from different coder agents
    codebook_path,
    embedding_model,
    retrieval_mode="code_clue"
):
    with open(compact_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    transcript_id = str(data.get("transcript_id", "")).strip()
    clue_pool = data.get("clue_pool", [])

    # group clues by code within this transcript
    code_to_clues = {}

    for item in clue_pool:
        code_name = (item.get("final_initial_code") or "").strip()
        clue = (item.get("text") or "").strip()

        if not code_name:
            continue

        code_to_clues.setdefault(code_name, [])
        if clue and clue not in code_to_clues[code_name]:
            code_to_clues[code_name].append(clue)

    unique_codes = list(code_to_clues.keys())

    texts_to_embed = [
        build_embedding_text(
            code_name=code_name,
            clues=code_to_clues[code_name],
            retrieval_mode=retrieval_mode
        )
        for code_name in unique_codes
    ]

    if texts_to_embed:
        embeddings = embedding_model.encode(texts_to_embed, normalize_embeddings=True)
    else:
        embeddings = []

    entries = []
    for code_name, emb in zip(unique_codes, embeddings):
        entries.append({
            "initial code": code_name,
            "embedding": np.asarray(emb, dtype=np.float32).tolist(),
            "transcript_ids": [transcript_id],
            "clues": code_to_clues[code_name]
        })

    codebook = {
        "retrieval_mode": retrieval_mode,
        "entries": entries
    }

    codebook_dir = os.path.dirname(codebook_path)
    if codebook_dir:
        os.makedirs(codebook_dir, exist_ok=True)

    with open(codebook_path, "w", encoding="utf-8") as f:
        json.dump(codebook, f, indent=2, ensure_ascii=False)

    return codebook

# Codebook compact after finishing transcript
def codebook_compact(codebook_path, embedding_model, threshold=0.8, retrieval_mode="code_clue"):
    with open(codebook_path, "r", encoding="utf-8") as f:
        entries = json.load(f).get("entries", [])

    norm = []
    for e in entries:
        code = str(e.get("initial code", "")).strip()
        if not code:
            continue

        clues = e.get("clues", [])
        if not clues and "clue" in e:
            clues = [e["clue"]]

        tids = e.get("transcript_ids", [])
        if not isinstance(tids, list):
            tids = [tids]

        embedding = e.get("embedding", None)
        if embedding is None:
            continue

        norm.append({
            "initial code": code,
            "embedding": embedding,
            "transcript_ids": list(set(str(t).strip() for t in tids if str(t).strip())),
            "clues": list(set(str(c).strip() for c in clues if str(c).strip()))
        })
    if not norm:
        codebook = {
            "retrieval_mode": retrieval_mode,
            "entries": []
        }
        return codebook

    emb_matrix = np.array(
        [e["embedding"] for e in norm if len(e["embedding"]) > 0],
        dtype=np.float32
    )
    sim_matrix = cosine_similarity(emb_matrix)

    used = set()
    merged = []

    for i in range(len(norm)):
        if i in used:
            continue

        base = {
            "initial code": norm[i]["initial code"],
            "clues": list(norm[i]["clues"]),
            "transcript_ids": list(norm[i]["transcript_ids"])
        }
        used.add(i)

        for j in range(i + 1, len(norm)):
            if j in used:
                continue

            same_name = norm[i]["initial code"].lower() == norm[j]["initial code"].lower()
            similar = sim_matrix[i, j] >= threshold

            if same_name or similar:
                candidate_clues = list(set(base["clues"] + norm[j]["clues"]))
                candidate_tids = list(set(base["transcript_ids"] + norm[j]["transcript_ids"]))

                # decide which code name to keep
                keep_j = False
                if len(norm[j]["transcript_ids"]) > len(base["transcript_ids"]):
                    keep_j = True
                elif len(norm[j]["transcript_ids"]) == len(base["transcript_ids"]):
                    if len(norm[j]["clues"]) > len(base["clues"]):
                        keep_j = True

                if keep_j:
                    base["initial code"] = norm[j]["initial code"]

                base["clues"] = candidate_clues
                base["transcript_ids"] = candidate_tids
                used.add(j)

        merged.append(base)

    texts = [
        build_embedding_text(
            code_name=e["initial code"],
            clues=e["clues"],
            retrieval_mode=retrieval_mode
        )
        for e in merged
    ]

    new_embeddings = embedding_model.encode(texts, normalize_embeddings=True)

    for e, emb in zip(merged, new_embeddings):
        e["embedding"] = np.asarray(emb, dtype=np.float32).tolist()

    codebook = {
        "retrieval_mode": retrieval_mode,
        "entries": merged
    }

    with open(codebook_path, "w", encoding="utf-8") as f:
        json.dump(codebook, f, indent=2, ensure_ascii=False)

    return codebook

def load_compact_file(compact_json_path):
    with open(compact_json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_codebook(codebook_path, retrieval_mode="code_clue"):
    if not os.path.exists(codebook_path):
        return {
            "retrieval_mode": retrieval_mode,
            "entries": []
        }

    with open(codebook_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return {
            "retrieval_mode": retrieval_mode,
            "entries": []
        }

    data = json.loads(text)

    if not isinstance(data, dict):
        return {
            "retrieval_mode": retrieval_mode,
            "entries": []
        }

    data.setdefault("retrieval_mode", retrieval_mode)
    data.setdefault("entries", [])

    return data
    
def save_codebook(codebook, path):
    codebook_dir = os.path.dirname(path)
    if codebook_dir:
        os.makedirs(codebook_dir, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(codebook, f, indent=2, ensure_ascii=False)

# New coding results for a new transcript
def prepare_new_entries_from_compact(compact_json_path, embedding_model, retrieval_mode="code_clue"):
    data = load_compact_file(compact_json_path)

    transcript_id = str(data.get("transcript_id", "")).strip()
    clue_pool = data.get("clue_pool", [])

    code_to_clues = {}

    for item in clue_pool:
        if not isinstance(item, dict):
            continue
        code_name = (item.get("final_initial_code") or "").strip()
        clue = (item.get("text") or "").strip()

        if not code_name:
            continue

        code_to_clues.setdefault(code_name, [])
        if clue and clue not in code_to_clues[code_name]:
            code_to_clues[code_name].append(clue)

    unique_codes = list(code_to_clues.keys())
    texts_to_embed = [
        build_embedding_text(
            code_name=code_name,
            clues=code_to_clues[code_name],
            retrieval_mode=retrieval_mode
        )
        for code_name in unique_codes
    ]

    if texts_to_embed:
        embeddings = embedding_model.encode(texts_to_embed, normalize_embeddings=True)
    else:
        embeddings = []

    prepared = []
    for code_name, emb in zip(unique_codes, embeddings):
        prepared.append({
            "code_name": code_name,
            "transcript_id": transcript_id,
            "embedding": np.asarray(emb, dtype=np.float32).tolist(),
            "clues": code_to_clues[code_name],
            "retrieval_mode": retrieval_mode
        })

    return prepared

# Retrieval top k codebook for a new code
def retrieve_topk_from_codebook(new_entry, codebook, top_k=3):
    emb_new = new_entry.get("embedding", None)
    if emb_new is None:
        return []
    q = np.asarray(emb_new, dtype=np.float32)
    if q.size == 0:
        return []
    q = q.reshape(1, -1)
    scored = []
    for entry in codebook.get("entries", []):
        emb = entry.get("embedding", None)
        if emb is None:
            continue
        emb = np.asarray(emb, dtype=np.float32)
        if emb.size == 0:
            continue
        emb = emb.reshape(1, -1)
        score = float(cosine_similarity(q, emb)[0, 0])
        scored.append({
            "score": score,
            "initial code": entry.get("initial code", ""),
            "clues": entry.get("clues", [])
        })
    scored.sort(key=lambda x: x["score"], reverse=True)

    return scored[:top_k]

# Retrieval top k codebook for all new codes
def retrieve_for_transcript_from_codebook(compact_json_path, codebook, embedding_model, top_k=3, high_threshold=0.80, retrieval_mode="code_clue", review_policy="threshold"):
    new_entries = prepare_new_entries_from_compact(
        compact_json_path=compact_json_path,
        embedding_model=embedding_model,
        retrieval_mode=retrieval_mode
    )

    results = []

    for new_entry in new_entries:
        retrieved = retrieve_topk_from_codebook(
            new_entry=new_entry,
            codebook=codebook,
            top_k=top_k
        )
        top_score = retrieved[0]["score"] if retrieved else 0.0
        if review_policy == "all_to_reviewer":
            decision = "reviewer_decide"

        elif review_policy == "threshold":
            if top_score >= high_threshold:
                decision = "reuse"
            else:
                decision = "reviewer_decide"

        else:
            raise ValueError(f"Unknown review_policy: {review_policy}")

        results.append({
            "code_name": new_entry["code_name"],
            "transcript_id": new_entry["transcript_id"],
            "clues": new_entry.get("clues", []),
            "top_k": retrieved,
            "top_score": top_score,
            "decision": decision,
            "retrieval_mode": retrieval_mode,
            "review_policy": review_policy
        })
    return results

def split_retrieval_results_by_decision(retrieval_results):
    reuse_results = []
    reviewer_results = []

    for item in retrieval_results:
        decision = (item.get("decision") or "").strip()

        if decision == "reuse":
            reuse_results.append(item)
        elif decision == "reviewer_decide":
            reviewer_results.append(item)

    return reuse_results, reviewer_results

# Process reuse part (based on threshold policy)
"""
New code from current transcript
→ retrieve top-k similar codebook entries
→ if top cosine similarity >= 0.80
→ directly reuse the top retrieved codebook entry
→ add the new clues + transcript_id into that existing codebook entry
"""
def merge_reuse_results_into_codebook(
    codebook,
    reuse_results,
    embedding_model,
    retrieval_mode="code_clue"
):
    n_merged = 0
    n_skipped = 0

    for item in reuse_results:
        transcript_id = str(item.get("transcript_id", "")).strip()
        new_clues = item.get("clues", [])
        top_k = item.get("top_k", [])

        if not isinstance(new_clues, list):
            new_clues = [new_clues]

        if not top_k:
            n_skipped += 1
            continue

        matched_code = (top_k[0].get("initial code") or "").strip()

        if not matched_code:
            n_skipped += 1
            continue

        matched_entry = None

        for e in codebook.get("entries", []):
            code = (e.get("initial code") or "").strip()

            if code.lower() == matched_code.lower():
                matched_entry = e
                break

        if matched_entry is None:
            n_skipped += 1
            print(f"ERROR! [WARNING] matched code not found in codebook: {matched_code}")
            continue

        # add clues
        matched_entry.setdefault("clues", [])
        for c in new_clues:
            c = str(c).strip()
            if c and c not in matched_entry["clues"]:
                matched_entry["clues"].append(c)

        # add transcript id
        matched_entry.setdefault("transcript_ids", [])
        if transcript_id and transcript_id not in matched_entry["transcript_ids"]:
            matched_entry["transcript_ids"].append(transcript_id)

        # update embedding
        text = build_embedding_text(
            code_name=matched_entry["initial code"],
            clues=matched_entry["clues"],
            retrieval_mode=retrieval_mode
        )

        emb = embedding_model.encode([text], normalize_embeddings=True)[0]
        matched_entry["embedding"] = np.asarray(emb, dtype=np.float32).tolist()

        n_merged += 1

    return {
        "n_merged": n_merged,
        "n_skipped": n_skipped
    }

# Process reviewer part
def reviewer_single(reviewer_input_text, retrieval_mode="code_clue"):
    if retrieval_mode == "code":
        retrieval_note = """
        You will see:
        - The new Initial Code and its Clues
        - Retrieved similar existing codebook entries (Code Name only)
        """
    elif retrieval_mode == "code_clue":
        retrieval_note = """
        You will see:
        - The new Initial Code and its Clues
        - Retrieved similar existing codebook entries (Code Name and Clues)
        """
    else:
        raise ValueError(f"Unknown retrieval_mode: {retrieval_mode}")

    system_instruction = f"""
    You are a review coder in the reflexive thematic analysis of interviews conducted with family caregivers of individuals with Alzheimer’s disease or related dementias.
    Your job is to review one new coding result against similar existing entries from the codebook.

    {retrieval_note}

    Your task:
    Step 1. Merge step:
    Decide whether the new code and its clues can be merged to one of the retrieved similar codebook entries.
    Step 2. New-code step:
    If no retrieved codebook entry adequately matches, keep the original Initial Code.

    Rules:
    - Always try to map/merge to an existing code first.
    - Do not create or invent a different code name.
    - If Status is EXISTING:
    Final Code must be exactly one Code Name from the retrieved similar codebook entries.
    - If Status is NEW:
    Final Code must be exactly the Initial Code.
    - Status must be exactly EXISTING or NEW.

    Output exactly:
    Final Code: [code name]
    Status: [EXISTING or NEW]
    """

    task_instruction = f"""
    {reviewer_input_text}

    ### OUTPUT ###
    Final Code: [final code name after review]
    Status: [EXISTING or NEW]
    """

    messages = [
        {"role": "system", "content": system_instruction.strip()},
        {"role": "user", "content": task_instruction.strip()}
    ]
    return messages

# format content will be sent to reviewer
def build_reviewer_input_text_single(item, retrieval_mode="code_clue"):
    code_name = (item.get("code_name") or "").strip()
    clues = item.get("clues", [])
    top_k = item.get("top_k", [])

    if not isinstance(clues, list):
        clues = [clues]

    lines = [f"New Code: {code_name}"]
    lines.append("Clues:")
    for clue in clues:
        clue = str(clue).strip()
        if clue:
            lines.append(f'- "{clue}"')

    lines.append("")
    lines.append("Retrieved similar codebook entries:")

    if not top_k:
        lines.append("(no similar codes found)")
    else:
        for idx, hit in enumerate(top_k, start=1):
            existing_code = (hit.get("initial code") or "").strip()
            lines.append(f"Code Name: {existing_code}")

            if retrieval_mode == "code_clue":
                existing_clues = hit.get("clues", [])
                if not isinstance(existing_clues, list):
                    existing_clues = [existing_clues]
                lines.append("Clues:")
                for clue in existing_clues:
                    clue = str(clue).strip()
                    if clue:
                        lines.append(f'- "{clue}"')

            lines.append("")

    return "\n".join(lines).strip()


# Reviewer output parser
_BLOCK_PATTERN_REVIEWER_SIMPLE = re.compile(
    r"Final\s*Code:\s*(.*?)\n\s*Status:\s*(.*?)(?=\n\s*Final\s*Code:|\Z)",
    flags=re.DOTALL | re.IGNORECASE,
)

def parse_reviewer_output_simple(text):
    if not text or not str(text).strip():
        return []

    out = []
    for code_raw, status_raw in _BLOCK_PATTERN_REVIEWER_SIMPLE.findall(text):
        final_code = " ".join(code_raw.strip().split())
        status = status_raw.strip().upper()

        if status in {"EXIST", "EXISTING"}:
            status = "EXISTING"
        elif status == "NEW":
            status = "NEW"
        else:
            continue

        if final_code:
            out.append({
                "final code": final_code,
                "status": status
            })

    return out

# NEW status
def add_new_entry(codebook, code, clues, transcript_id, embedding):
    codebook.setdefault("entries", []).append({
        "initial code": code,
        "embedding": np.asarray(embedding, dtype=np.float32).tolist(),
        "transcript_ids": [transcript_id],
        "clues": clues
    })

# EXISTING status
def merge_existing_reviewer_result_into_codebook(
    codebook,
    final_code,
    clues,
    transcript_id,
    embedding_model,
    retrieval_mode="code_clue",
    sim_threshold=0.95
):
    matched_entry = None
    final_code = str(final_code or "").strip()
    final_code_lower = final_code.lower()

    # Step 1: lowercase exact match
    for e in codebook.get("entries", []):
        code = (e.get("initial code") or "").strip()
        if code.lower() == final_code_lower:
            matched_entry = e
            break

    # Step 2: high code-name embedding similarity
    if matched_entry is None:
        target_text = build_code_text(final_code)
        target_emb = embedding_model.encode([target_text], normalize_embeddings=True)[0]

        best_entry = None
        best_score = -1

        for e in codebook.get("entries", []):
            code = (e.get("initial code") or "").strip()
            code_text = build_code_text(code)
            code_emb = embedding_model.encode([code_text], normalize_embeddings=True)[0]

            score = float(
                cosine_similarity(
                    np.asarray(target_emb).reshape(1, -1),
                    np.asarray(code_emb).reshape(1, -1)
                )[0, 0]
            )

            if score > best_score:
                best_score = score
                best_entry = e

        if best_entry is not None and best_score >= sim_threshold:
            matched_entry = best_entry
        else:
            print(
                f"WARNING: Reviewer EXISTING code not found in codebook: {final_code}. "
                f"Best similarity={best_score:.4f}. Adding as NEW instead."
            )

            if not isinstance(clues, list):
                clues = [clues]

            embedding_text = build_embedding_text(
                code_name=final_code,
                clues=clues,
                retrieval_mode=retrieval_mode
            )
            embedding = embedding_model.encode([embedding_text], normalize_embeddings=True)[0]

            add_new_entry(
                codebook=codebook,
                code=final_code,
                clues=clues,
                transcript_id=transcript_id,
                embedding=embedding
            )

            return "added_as_new"

    # Merge clues
    if not isinstance(clues, list):
        clues = [clues]

    matched_entry.setdefault("clues", [])
    for c in clues:
        c = str(c).strip()
        if c and c not in matched_entry["clues"]:
            matched_entry["clues"].append(c)

    # Merge transcript id
    transcript_id = str(transcript_id).strip()
    matched_entry.setdefault("transcript_ids", [])
    if transcript_id and transcript_id not in matched_entry["transcript_ids"]:
        matched_entry["transcript_ids"].append(transcript_id)

    # Update embedding
    text = build_embedding_text(
        code_name=matched_entry["initial code"],
        clues=matched_entry["clues"],
        retrieval_mode=retrieval_mode
    )

    emb = embedding_model.encode([text], normalize_embeddings=True)[0]
    matched_entry["embedding"] = np.asarray(emb, dtype=np.float32).tolist()

    return True
    
def update_codebook_from_reviewer_results(
    codebook,
    reviewer_results,
    transcript_id,
    embedding_model,
    retrieval_mode="code_clue"
):
    n_merged = 0
    n_added = 0
    n_skipped_existing = 0
    n_skipped_unknown_status = 0

    for item in reviewer_results:
        final_code = (item.get("final code") or "").strip()
        status = (item.get("status") or "").strip().upper()
        clues = item.get("clues", [])

        if not final_code:
            continue

        if not isinstance(clues, list):
            clues = [clues]

        if status == "EXISTING":
            merged = merge_existing_reviewer_result_into_codebook(
                codebook=codebook,
                final_code=final_code,
                clues=clues,
                transcript_id=transcript_id,
                embedding_model=embedding_model,
                retrieval_mode=retrieval_mode,
                sim_threshold=0.95
            )

            if merged:
                n_merged += 1
            else:
                n_skipped_existing += 1

                print(
                    f"ERROR! [SKIPPED] Reviewer marked EXISTING, but no matched codebook entry was found. "
                )

        elif status == "NEW":
            embedding_text = build_embedding_text(
                code_name=final_code,
                clues=clues,
                retrieval_mode=retrieval_mode
            )

            embedding = embedding_model.encode(
                [embedding_text],
                normalize_embeddings=True
            )[0]

            add_new_entry(
                codebook=codebook,
                code=final_code,
                clues=clues,
                transcript_id=transcript_id,
                embedding=embedding
            )

            n_added += 1

        else:
            n_skipped_unknown_status += 1
            print(f"ERROR! [WARNING] Unknown reviewer status skipped: {status} | code={final_code}")

    return {
        "n_merged": n_merged,
        "n_added": n_added,
        "n_skipped_existing": n_skipped_existing,
        "n_skipped_unknown_status": n_skipped_unknown_status
    }

### Final Function
def run_multi_transcript_dynamic_rounds(
    client,
    llm_model,
    embedding_model,
    files_list,
    agents,
    max_rounds,
    all_agent_initial_reflections,
    coding_base_dir,
    final_coding_base_dir,
    final_coding_without_score_base_dir,
    reflection_output_dir,
    agent_personas,
    codebook_path,
    review_output_dir,
    research_question,
    interviewee,
    novelty_threshold=0.05,
    k=2,
    clue_pool_dir_for_novelty=None
):
    all_novelty_logs = []  

    research_dir = os.path.dirname(coding_base_dir) # dataset folder
    output_dir_1a = os.path.join(research_dir, "clue_pool_step1a")  # merge all agents
    output_dir_1c = os.path.join(research_dir, "clue_pool_step1c")  # containment dedup
    output_dir_1d = os.path.join(research_dir, "clue_pool_step1d")  # similarity dedup
    output_dir_filtered_4 = os.path.join(research_dir, "clue_pool_filtered_4") # score >= 4
    output_dir_compact = os.path.join(research_dir, "clue_pool_compact") # select the final initial code for a clue
    os.makedirs(reflection_output_dir, exist_ok=True)

    if clue_pool_dir_for_novelty is None:
        clue_pool_dir_for_novelty = output_dir_1d
    
    os.makedirs(output_dir_1a, exist_ok=True)
    os.makedirs(output_dir_1c, exist_ok=True)
    os.makedirs(output_dir_1d, exist_ok=True)
    os.makedirs(output_dir_filtered_4, exist_ok=True)
    os.makedirs(output_dir_compact, exist_ok=True)
    os.makedirs(review_output_dir, exist_ok=True)
    os.makedirs(final_coding_base_dir, exist_ok=True)
    os.makedirs(final_coding_without_score_base_dir, exist_ok=True)

    for t_idx, selected_files in enumerate(files_list, start=1):
        if not selected_files:
            print("empty, skipping.")
            continue

        tid0, _ = parse_ids(selected_files[0]) # transcript id, chunk id
        if not tid0:
            raise ValueError(f"ERROR! Filename not matched: {selected_files[0]}")

        for fp in selected_files:
            tid, _ = parse_ids(fp)
            if tid != tid0:
                raise ValueError(f"ERROR! Mixed transcript IDs in same group: expected {tid0}, got {tid} in {fp}")

        transcript_id = tid0
        print(f"\n\n==============================")
        print(f"  TRANSCRIPT {transcript_id} (order {t_idx})")
        print(f"==============================")

        novelty_hist = []
        round_number = 1
        last_completed_round = None

        while round_number <= max_rounds:
            print(f"\n--- Transcript {transcript_id} | Round {round_number}/{max_rounds} ---")

            ### independent coding among agents###
            for fp in selected_files:
                tid, chunk_id = parse_ids(fp)
                if not tid:
                    print(f"ERROR! Skipping (filename not matched): {os.path.basename(fp)}")
                    continue
                transcript = read_docx(fp)

                for agent_name in agents:
                    if (t_idx == 1) and (round_number == 1): # first transcript under first round
                        agent_reflection = all_agent_initial_reflections[agent_name]  # ref0
                    else:
                        raw_reflection = get_latest_reflection(agent_name=agent_name, output_base_dir=reflection_output_dir)  # ref1/ref2/... global

                        if not raw_reflection:
                            print(
                                f"ERROR! Skipping {agent_name} | transcript {tid} chunk {chunk_id}: "
                                "no updated reflection found."
                            )
                            continue
                    
                        agent_reflection = copy.deepcopy(raw_reflection)
                        if round_number == 1 and isinstance(agent_reflection, dict): # other transcripts under first round
                            agent_reflection = {
                                "assumptions": agent_reflection.get("assumptions", "")
                            }

                    coding_memory_content = None
                    if round_number > 1: # coding memory
                        coding_memory_content = load_prev_round_coding_output(
                            coding_base_dir=coding_base_dir,
                            agent_name=agent_name,
                            previous_round=round_number - 1,
                            transcript_id=tid,
                            chunk_id=chunk_id
                        )

                    try:
                        result = run_independent_coding(
                            client=client,
                            model=llm_model,
                            transcript=transcript,
                            agent_name=agent_name,
                            reflection=agent_reflection,
                            transcript_id=tid,
                            chunk_id=chunk_id,
                            round_number=round_number,
                            fixed_profile=agent_personas[agent_name],
                            coding_memory_content=coding_memory_content,
                            research_question=research_question,
                            interviewee=interviewee
                        )
                        print(f"Done: {agent_name} | chunk {chunk_id} | round {round_number}")
                        save_coding_result(
                            result=result,
                            output_base_dir=coding_base_dir
                        )
                    except Exception as e:
                        print(f"Error: {agent_name} | chunk {chunk_id} | round {round_number}: {e}")

            clean_chunk_round_outputs(coding_base_dir, agents, transcript_id, round_number)
            merge_chunk_round_outputs(coding_base_dir=coding_base_dir, final_coding_base_dir=final_coding_base_dir, agents=agents, transcript_id=transcript_id, round_number=round_number) # merge each agent chunk results
            # _final.json contains score

            remove_importance_score(input_base_dir=final_coding_base_dir, output_base_dir=final_coding_without_score_base_dir) # generate _final_without_score json files

            # build clue_pool for this round (used for novelty)
            run_merge_all(final_coding_base_dir, output_dir_1a, [transcript_id], [round_number], agents)
            run_build_clue_pools_containment(output_dir_1a, output_dir_1c)
            run_build_clue_pools_similar(output_dir_1c, output_dir_1d)

            # mark this round as successfully completed
            last_completed_round = round_number

            # novelty check (round>=2)
            if round_number >= 2:
                novelty_val = compute_single_transcript_round_novelty(
                    clue_pool_dir=clue_pool_dir_for_novelty,
                    transcript_id=transcript_id,
                    cur_round=round_number,
                    embedding_model=embedding_model
                )
                all_novelty_logs.append({
                    "transcript_id": transcript_id,
                    "cur_round": round_number,
                    "pre_round": round_number - 1,
                    "novelty": novelty_val,
                })
                if novelty_val is None or (isinstance(novelty_val, float) and np.isnan(novelty_val)):
                    novelty_hist.append(None)
                    print(f"[Novelty] {transcript_id} r{round_number-1}->r{round_number}: (missing)")
                else:
                    novelty_hist.append(novelty_val)
                    print(f"[Novelty] {transcript_id} r{round_number-1}->r{round_number}: {novelty_val:.4f}")

                if should_stop_by_novelty(novelty_hist, novelty_threshold, k):
                    print(f"\n[STOP] Novelty saturated for {transcript_id}: last {k} transitions < {novelty_threshold}.")
                    break

            # collaboration (only if we will continue to another round)
            if round_number < max_rounds:
                print(f"\n--- Collaborative reflection update (creates next ref) ---")
                coding_results = {}
                for agent_name in agents:
                    fp = os.path.join(final_coding_without_score_base_dir, agent_name, f"{transcript_id}_round_{round_number}_final.json")

                    if not os.path.exists(fp):
                        print(f"[WARN] Missing coding result: {fp}")
                        continue

                    with open(fp, "r", encoding="utf-8") as f:
                        coding_results[agent_name] = json.load(f)

                for target_agent in agents:
                    try:
                        current_reflection = get_latest_reflection(
                            agent_name=target_agent,
                            output_base_dir=reflection_output_dir
                        )

                        if not current_reflection:
                            current_reflection = copy.deepcopy(
                                all_agent_initial_reflections[target_agent]
                            )

                        updated_reflection = run_reflection_update(
                            client=client,
                            model=llm_model,
                            target_agent=target_agent,
                            fixed_profile=agent_personas[target_agent],
                            current_reflection=current_reflection,
                            coding_results=coding_results,
                            agents=agents
                        )

                        save_reflection(
                            reflection=updated_reflection,
                            agent_name=target_agent,
                            output_base_dir=reflection_output_dir
                        )
                        print(f"Updated reflection: {target_agent} (after round {round_number})")
                    except Exception as e:
                        print(f"Reflection update failed: {target_agent} | {e}")

            round_number += 1

        print("\n--- Transcript finished (dynamic rounds) ---")

        # we only process coding results from final round
        if last_completed_round is None:
            print(f"ERROR! No completed rounds for transcript {transcript_id}, skipping score filter/code selection.")
            continue

        final_clue_pool_file = os.path.join(
            output_dir_1d,
            f"clue_pool_{transcript_id}_round_{last_completed_round}.json"
        )

        if not os.path.exists(final_clue_pool_file):
            print(f"Final clue pool file not found: {final_clue_pool_file}")
            continue
        
        # filter by score for this transcript only
        summary = filter_json_files_by_score([final_clue_pool_file], output_dir_filtered_4, min_score=4)
        # choose one final code per clue for this transcript only
        filtered_file = os.path.join(
            output_dir_filtered_4,
            os.path.basename(final_clue_pool_file)
        )
        compact_output_file = os.path.join(
            output_dir_compact,
            os.path.basename(filtered_file).replace(".json", "_compact.json")
        )
        choose_one_code_per_clue(
            input_path=filtered_file,
            output_path=compact_output_file,
            embedding_model=embedding_model,
            sim_threshold=0.80
        )

        # ============================================================
        # Codebook option comparison after final code-clue generation
        # ============================================================
        base_codebook_dir = os.path.dirname(codebook_path)

        for option_id, config in OPTION_CONFIGS.items():
            option_name = config["name"]
            retrieval_mode = config["retrieval_mode"]
            review_policy = config["review_policy"]

            option_codebook_path = get_option_codebook_path(
                base_codebook_dir=base_codebook_dir,
                option_id=option_id
            )

            # --------------------------------------------------------
            # initialize it from current transcript
            # --------------------------------------------------------
            if t_idx == 1 and not os.path.exists(option_codebook_path):
                initialize_codebook_from_compact_file(
                    compact_json_path=compact_output_file,
                    codebook_path=option_codebook_path,
                    embedding_model=embedding_model,
                    retrieval_mode=retrieval_mode
                )

                codebook_compact(
                    codebook_path=option_codebook_path,
                    embedding_model=embedding_model,
                    threshold=0.80,
                    retrieval_mode=retrieval_mode
                )

                codebook = load_codebook(
                    option_codebook_path,
                    retrieval_mode=retrieval_mode
                )

                versioned_path = os.path.join(
                    os.path.dirname(option_codebook_path),
                    f"codebook_after_{transcript_id}.json"
                )

                save_codebook(codebook, versioned_path)

                continue

            # --------------------------------------------------------
            # retrieve, split, reviewer/update, compact
            # --------------------------------------------------------
            codebook = load_codebook(
                option_codebook_path,
                retrieval_mode=retrieval_mode
            )

            retrieval_results = retrieve_for_transcript_from_codebook(
                compact_json_path=compact_output_file,
                codebook=codebook,
                embedding_model=embedding_model,
                top_k=3,
                high_threshold=0.80,
                retrieval_mode=retrieval_mode,
                review_policy=review_policy
            )

            reuse_results, reviewer_results = split_retrieval_results_by_decision(retrieval_results)

            if reuse_results:
                reuse_summary = merge_reuse_results_into_codebook(
                    codebook=codebook,
                    reuse_results=reuse_results,
                    embedding_model=embedding_model,
                    retrieval_mode=retrieval_mode
                )

            if reviewer_results:
                all_reviewer_parsed = []
                review_log_entries = []
                n_content_filtered = 0
                n_empty_response = 0
                n_parse_failures = 0

                for item in reviewer_results:
                    reviewer_input_text = build_reviewer_input_text_single(
                        item,
                        retrieval_mode=retrieval_mode
                    )

                    messages = reviewer_single(
                        reviewer_input_text=reviewer_input_text,
                        retrieval_mode=retrieval_mode
                    )

                    try:
                        resp = get_completion(client=client, model=llm_model, messages=messages)

                    except Exception as e:
                        print(f"ERROR! WARNING: API error for {item.get('code_name', '')}: {e}")
                        print("Adding directly to codebook as NEW (compact will merge if similar)")

                        code_name = item.get("code_name", "").strip()
                        clues = item.get("clues", [])
                        if not isinstance(clues, list):
                            clues = [clues]

                        embedding_text = build_embedding_text(
                            code_name=code_name,
                            clues=clues,
                            retrieval_mode=retrieval_mode
                        )
                        embedding = embedding_model.encode([embedding_text], normalize_embeddings=True)[0]

                        add_new_entry(
                            codebook=codebook,
                            code=code_name,
                            clues=clues,
                            transcript_id=transcript_id,
                            embedding=embedding
                        )

                        review_log_entries.append({
                            "code_name": code_name,
                            "parsed": [{"final code": code_name, "status": "API ERROR", "clues": clues}]
                        })

                        n_content_filtered += 1

                        continue

                    time.sleep(2)
                    review_content = resp.choices[0].message.content

                    if not review_content:
                        print(f"ERROR! WARNING: Empty reviewer response for: {item.get('code_name', '')}")
                        print("Adding directly to codebook as NEW (compact will merge if similar)")

                        code_name = item.get("code_name", "").strip()
                        clues = item.get("clues", [])
                        if not isinstance(clues, list):
                            clues = [clues]

                        embedding_text = build_embedding_text(
                            code_name=code_name,
                            clues=clues,
                            retrieval_mode=retrieval_mode
                        )
                        embedding = embedding_model.encode([embedding_text], normalize_embeddings=True)[0]

                        add_new_entry(
                            codebook=codebook,
                            code=code_name,
                            clues=clues,
                            transcript_id=transcript_id,
                            embedding=embedding
                        )

                        review_log_entries.append({
                            "code_name": code_name,
                            "parsed": [{"final code": code_name, "status": "Reviewer agent return empty", "clues": clues}]
                        })
                        n_empty_response += 1

                        continue

                    review_text = review_content.strip()

                    parsed = parse_reviewer_output_simple(review_text)
                    if not parsed:
                        print(f"WARNING: Failed to parse reviewer output for: {item.get('code_name', '')}")
                        print("Adding directly to codebook as NEW (compact will merge if similar)")

                        code_name = item.get("code_name", "").strip()
                        clues = item.get("clues", [])
                        if not isinstance(clues, list):
                            clues = [clues]

                        embedding_text = build_embedding_text(
                            code_name=code_name,
                            clues=clues,
                            retrieval_mode=retrieval_mode
                        )
                        embedding = embedding_model.encode([embedding_text], normalize_embeddings=True)[0]

                        add_new_entry(
                            codebook=codebook,
                            code=code_name,
                            clues=clues,
                            transcript_id=transcript_id,
                            embedding=embedding
                        )

                        review_log_entries.append({
                            "code_name": code_name,
                            "parsed": [{"final code": code_name, "status": "Fail to format Reviewer output", "clues": clues}]
                        })
                        n_parse_failures += 1
                        continue

                    for p in parsed:
                        p["clues"] = item.get("clues", [])
                        if p.get("status") == "NEW":
                            if p.get("final code", "").strip().upper() != item.get("code_name", "").strip().upper():
                                p["final code"] = item.get("code_name", "").strip()
                                print(f"ERROR! WARNING: Reviewer return code name mismatch, restored to: {p['final code']}")
                            
                    all_reviewer_parsed.extend(parsed)
                    review_log_entries.append({
                        "code_name": item.get("code_name", ""),
                        "parsed": parsed
                    })

                # save log
                review_output_path = os.path.join(
                    review_output_dir,
                    f"option_{option_id}_{option_name}_"
                    + os.path.basename(compact_output_file).replace(".json", "_reviewed.md")
                )

                with open(review_output_path, "w", encoding="utf-8") as f:
                    for entry in review_log_entries:
                        f.write(f"--- {entry['code_name']} ---\n")
                        f.write(str(entry['parsed']) + "\n\n")

                review_log_json_path = os.path.join(
                    review_output_dir,
                    f"option_{option_id}_{option_name}_"
                    + os.path.basename(compact_output_file).replace(".json", "_reviewed_log.json")
                )

                with open(review_log_json_path, "w", encoding="utf-8") as f:
                    json.dump(review_log_entries, f, indent=2, ensure_ascii=False)

                update_summary = update_codebook_from_reviewer_results(
                    codebook=codebook,
                    reviewer_results=all_reviewer_parsed,
                    transcript_id=transcript_id,
                    embedding_model=embedding_model,
                    retrieval_mode=retrieval_mode
                )

                total_accounted = (
                    update_summary["n_merged"]
                    + update_summary["n_added"]
                    + update_summary["n_skipped_existing"]
                    + update_summary["n_skipped_unknown_status"]
                    + n_content_filtered
                    + n_empty_response
                    + n_parse_failures
                )

                if total_accounted != len(reviewer_results):
                    print(f"ERROR! Reviewer accounting mismatch: {total_accounted} != {len(reviewer_results)}")

            save_codebook(codebook, option_codebook_path) # save pre-compact

            codebook_compact(
                    codebook_path=option_codebook_path,
                    embedding_model=embedding_model,
                    threshold=0.80,
                    retrieval_mode=retrieval_mode
                ) # read, compact, save

            codebook = load_codebook(
                option_codebook_path,
                retrieval_mode=retrieval_mode
            ) # load compact version

            versioned_path = os.path.join(
                os.path.dirname(option_codebook_path),
                f"codebook_after_{transcript_id}.json"
            )
            save_codebook(codebook, versioned_path)

            print(f"Finished option {option_id}: {option_name}")
