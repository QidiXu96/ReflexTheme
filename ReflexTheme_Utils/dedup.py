import re, json, os, glob
from difflib import SequenceMatcher

# process one agent result
def extract_res(json_files, field="coding_output"):
    """
    [
        {
            "clues": "...",
            "initial code": "...",
            "importance score": 4
        }
    ] readable
    """
    if isinstance(json_files, str):
        json_files = [json_files]

    block_pattern = re.compile(
        r"Initial Code:\s*(.*?)\n"
        r"\s*Clues?:\s*(.*?)\n"
        r"(?:\s*Importance Score:\s*(\d+)\s*)?"
        r"(?=\n\s*Initial Code:|\Z)",
        flags=re.DOTALL | re.IGNORECASE,
    )

    all_extracted_items = []

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict) or field not in data:
            continue

        text = data[field]
        for code_raw, clues_raw, score_raw in block_pattern.findall(text):
            code = " ".join(code_raw.strip().split())
            clues = clues_raw.strip().replace("“", "").replace("”", "").replace('"', "")
            score = int(score_raw) if score_raw and score_raw.isdigit() else None
            
            if code and clues:
                all_extracted_items.append({"clues": clues, "initial code": code, "importance score": score})

    return all_extracted_items

# merge all agents results for one transcript (all chunks) one round
def merge_agents_one_transcript_round(base_path, transcript_id, round_id, agents, field="coding_output"):
    merged_items = []

    for agent in agents:
        fp = os.path.join(base_path, agent, f"{transcript_id}_round_{round_id}_final.json")
        if not os.path.exists(fp):
            print(f"[WARN] Missing file: {fp}")
            continue

        items = extract_res(fp, field=field)

        for it in items:
            it["agent"] = agent
            it["transcript_id"] = transcript_id
            it["round"] = round_id

        merged_items.extend(items)

    return merged_items

# run all transcripts
def run_merge_all(base_path, output_base, transcript_ids, rounds, agents, field="coding_output"):
    os.makedirs(output_base, exist_ok=True)
    output_paths = []

    for tid in transcript_ids:
        for r in rounds:
            merged_items = merge_agents_one_transcript_round(
                base_path=base_path,
                transcript_id=tid,
                round_id=r,
                agents=agents,
                field=field
            )

            out_payload = {
                "transcript_id": tid,
                "round": r,
                "agents": agents,
                "n_items": len(merged_items),
                "items": merged_items
            }

            out_fp = os.path.join(output_base, f"{tid}_round_{r}_merged_items.json")
            with open(out_fp, "w", encoding="utf-8") as f:
                json.dump(out_payload, f, indent=2, ensure_ascii=False)
            output_paths.append(out_fp)
    return output_paths

# hard dedup
def build_clue_pool_from_one_file(inf):
    # process merged files
    def norm_for_dedup(s: str) -> str:
        x = (s or "").lower()
        x = re.sub(r"\s+", " ", x).strip()
        x = re.sub(r"[^\w\s]", "", x)
        return x
    
    transcript_id = inf.get("transcript_id")
    round_num = inf.get("round")
    items = inf.get("items", [])

    clue_pool = []
    key_to_idx = {}

    for it in items:
        clue_text = (it.get("clues") or "").strip()
        code = (it.get("initial code") or "").strip()
        score_raw = it.get("importance score", None)
        score = int(score_raw) if score_raw is not None and str(score_raw).isdigit() else None

        if not clue_text or not code:
            continue

        k = norm_for_dedup(clue_text)
        if not k:
            continue

        if k in key_to_idx:
            idx = key_to_idx[k] # clue text is key
        else:
            idx = len(clue_pool)
            key_to_idx[k] = idx # position is value
            clue_pool.append({
                "clue_id": idx + 1,
                "text": clue_text,  
                "average_score": None,
                "sources": []
            })

        src = {
            "agent": it.get("agent"),
            "transcript_id": it.get("transcript_id"),
            "round": it.get("round"),
            "initial code": code,
            "importance score": score
        }
        if src not in clue_pool[idx]["sources"]:
            clue_pool[idx]["sources"].append(src)
    for cl in clue_pool:
        scores = [
            e.get("importance score")
            for e in cl["sources"]
            if isinstance(e.get("importance score"), int)
        ]
        cl["average_score"] = (sum(scores) / len(scores)) if scores else None

    return {
        "transcript_id": transcript_id,
        "round": round_num,
        "n_items_raw": len(items),
        "unique_clues": len(clue_pool),
        "clue_pool": clue_pool
    }

def run_build_clue_pools(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_paths = []

    merged_files = sorted(glob.glob(os.path.join(input_dir, "*_merged_items.json")))
    if not merged_files:
        print(f"[WARN] No *_merged_items.json found in: {input_dir}")
        return []

    for fp in merged_files:
        with open(fp, "r", encoding="utf-8") as f:
            merged = json.load(f)

        transcript_id = merged.get("transcript_id")
        round_num = merged.get("round")

        if not transcript_id or round_num is None:
            print(f"[WARN] Missing transcript_id/round in {fp}, skipping.")
            continue

        payload = build_clue_pool_from_one_file(
            inf=merged
        )

        out_fp = os.path.join(output_dir, f"clue_pool_{transcript_id}_round_{round_num}.json")
        with open(out_fp, "w", encoding="utf-8") as out:
            json.dump(payload, out, indent=2, ensure_ascii=False)
        output_paths.append(out_fp)

    return output_paths

# containment      
def norm_for_dedup(s: str) -> str:
    x = (s or "").lower()
    x = re.sub(r"\s+", " ", x).strip()
    x = re.sub(r"[^\w\s]", "", x)
    return x

def is_contained(long_text: str, short_text: str, min_chars=60) -> bool:
    L = norm_for_dedup(long_text)
    S = norm_for_dedup(short_text)
    if len(S) < min_chars:  
        return False
    return S in L

def merge_by_containment_keep_longest(clues):
    retained = []
    for c in sorted(clues, key=len, reverse=True):  # longest first
        if any(is_contained(r, c) for r in retained): # Do we already have a longer clue that fully includes this one?
            # c is contained in an already-retained longer clue -> drop it
            continue
        retained.append(c)
    return retained

def build_clue_pool_from_one_file_containment(inf):
    transcript_id = inf.get("transcript_id")
    round_num = inf.get("round")
    items = inf.get("items", [])

    candidates = []
    for it in items:
        clue_text = (it.get("clues") or "").strip()
        code = (it.get("initial code") or "").strip()
        score_raw = it.get("importance score", None)
        score = int(score_raw) if score_raw is not None and str(score_raw).isdigit() else None

        if not clue_text or not code:
            continue

        candidates.append({
            "text": clue_text,
            "evidence": [{
                "text": clue_text,
                "agent": it.get("agent"),
                "transcript_id": it.get("transcript_id"),
                "round": it.get("round"),
                "initial code": code,
                "importance score": score
            }]
        })

    n_items_raw = len(candidates)

    candidates = sorted(candidates, key=lambda x: len(x["text"]), reverse=True)

    retained = []
    for cand in candidates:
        absorbed = False # Did we merge this clue (cand) into an existing retained clue?
        for r in retained:
            if is_contained(r["text"], cand["text"], min_chars=60):
                existing_ids = {
                    (
                        e["agent"],
                        e["transcript_id"],
                        e["round"],
                        e["initial code"],
                        e["text"],
                        e["importance score"]
                    )
                    for e in r["evidence"]
                }
                for ev in cand["evidence"]:
                    ev_id = (
                        ev["agent"],
                        ev["transcript_id"],
                        ev["round"],
                        ev["initial code"],
                        ev["text"],
                        ev["importance score"]
                    )
                    
                    if ev_id not in existing_ids:
                        r["evidence"].append(ev)
                absorbed = True
                break # no need to check other retained clues

        if not absorbed: # this clue has new information
            retained.append(cand)

    for r in retained:
        r["text"] = max([e["text"] for e in r["evidence"]], key=len)

    clue_pool = []
    for i, r in enumerate(retained, start=1):
        scores = [
            e.get("importance score")
            for e in r["evidence"]
            if isinstance(e.get("importance score"), int)
        ]
        avg_score = (sum(scores) / len(scores)) if scores else None
        clue_pool.append({
            "clue_id": i,
            "text": r["text"],        # canonical (longest)
            "average_score": avg_score,
            "evidence": r["evidence"] # exact agent→text mapping
        })

    return {
        "transcript_id": transcript_id,
        "round": round_num,
        "n_items_raw": n_items_raw,
        "unique_clues": len(clue_pool),
        "clue_pool": clue_pool,
    }

def run_build_clue_pools_containment(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_paths = []

    merged_files = sorted(glob.glob(os.path.join(input_dir, "*_merged_items.json")))
    if not merged_files:
        print(f"[WARN] No *_merged_items.json found in: {input_dir}")
        return []

    for fp in merged_files:
        with open(fp, "r", encoding="utf-8") as f:
            merged = json.load(f)

        transcript_id = merged.get("transcript_id")
        round_num = merged.get("round")

        if not transcript_id or round_num is None:
            print(f"[WARN] Missing transcript_id/round in {fp}, skipping.")
            continue

        payload = build_clue_pool_from_one_file_containment(inf=merged)

        out_fp = os.path.join(output_dir, f"clue_pool_{transcript_id}_round_{round_num}.json")
        with open(out_fp, "w", encoding="utf-8") as out:
            json.dump(payload, out, indent=2, ensure_ascii=False)
        output_paths.append(out_fp)

    return output_paths
    
# overalp
def similar_enough(a, b, threshold=0.80, min_chars=200):
    A = norm_for_dedup(a)
    B = norm_for_dedup(b)

    if min(len(A), len(B)) < min_chars:
        return False

    r = SequenceMatcher(None, A, B).ratio()
    return r >= threshold

def merge_evidence_unique(dst_evidence, src_evidence):
    seen = set()
    for e in dst_evidence:
        seen.add((
            e.get("agent"),
            e.get("transcript_id"),
            e.get("round"),
            e.get("initial code"),
            e.get("text"),
            e.get("importance score")
        ))

    for e in src_evidence:
        k = (
            e.get("agent"),
            e.get("transcript_id"),
            e.get("round"),
            e.get("initial code"),
            e.get("text"),
            e.get("importance score")
        )
        if k not in seen:
            dst_evidence.append(e)
            seen.add(k)

def build_clue_pool_from_one_file_similar(inf, threshold=0.8, min_chars=200):
    transcript_id = inf.get("transcript_id")
    round_num = inf.get("round")
    clue_pool = inf.get("clue_pool", [])

    merged = []  # list of {clue_id, text, evidence}
    next_id = 1

    for cl in clue_pool:
        text = (cl.get("text") or "").strip()
        evidence = cl.get("evidence", [])
        if not text:
            continue

        placed = False
        for m in merged:
            if similar_enough(m["text"], text, threshold=threshold, min_chars=min_chars):
                # Keep the longer canonical text as representative 
                if len(text) > len(m["text"]):
                    m["text"] = text

                merge_evidence_unique(m["evidence"], evidence)
                placed = True
                break

        if not placed:
            merged.append({
                "clue_id": next_id,
                "text": text,
                "evidence": list(evidence) if isinstance(evidence, list) else []
            })
            next_id += 1
    for m in merged:
        scores = [
            e.get("importance score")
            for e in m["evidence"]
            if isinstance(e.get("importance score"), int)
        ]
        m["average_score"] = (sum(scores) / len(scores)) if scores else None

    out = {
        "transcript_id": transcript_id,
        "round": round_num,
        "n_items_raw": inf.get("n_items_raw"),
        "unique_clues_before": inf.get("unique_clues"),
        "unique_clues": len(merged),
        "clue_pool": merged,
    }
    return out

def run_build_clue_pools_similar(input_dir, output_dir, threshold=0.80, min_chars=200):
    os.makedirs(output_dir, exist_ok=True)
    output_paths = []

    files = sorted(glob.glob(os.path.join(input_dir, "clue_pool_*_round_*.json")))
    if not files:
        print(f"[WARN] No clue_pool_*_round_*.json found in: {input_dir}")
        return []

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            inf = json.load(f)

        transcript_id = inf.get("transcript_id")
        round_num = inf.get("round")

        if not transcript_id or round_num is None:
            print(f"[WARN] Missing transcript_id/round in {fp}, skipping.")
            continue

        payload = build_clue_pool_from_one_file_similar(
            inf=inf,
            threshold=threshold,
            min_chars=min_chars
        )

        out_fp = os.path.join(output_dir, f"clue_pool_{transcript_id}_round_{round_num}.json")
        with open(out_fp, "w", encoding="utf-8") as out:
            json.dump(payload, out, indent=2, ensure_ascii=False)
        output_paths.append(out_fp)

    return output_paths
