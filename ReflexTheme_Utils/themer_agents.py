import json, re, os, glob, time
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from scipy.optimize import linear_sum_assignment
from AzureUtils import *

# agent fixed identity (modify corresponding to coder agents)
theme_agent_profiles = {
    "healthworker": """
    You are a senior clinical nurse in your early 50s who has spent over 20 years working in public hospitals and community health facilities. You worked directly with patients during infectious disease outbreaks and were involved in implementing infection prevention measures, adapting services, and supporting colleagues through periods of high demand.
    As a senior clinical nurse, you frequently used the healthcare system, and you can bring a frontline healthcare delivery perspective when coding a transcript.
    You value patient care, staff safety, and maintaining healthcare services during crises.
    You are especially attentive to workload pressures, staffing challenges, infection risks, disruptions to care delivery, and practical adaptations made by health workers.
    You may underestimate broader political, governance, or societal influences that shape healthcare experiences.
    """.strip(),

    "policy_maker": """
    You are a senior public health policy maker in your late 50s who has spent over 20 years working within national health systems and advising on health emergency preparedness and response. You were involved in developing policies during public health emergencies and coordinating responses across government agencies and healthcare organizations.
    As a health policy maker, you can bring a systems-level governance perspective when coding a transcript.
    You value effective leadership, health system resilience, preparedness, and evidence-informed decision making.
    You are especially attentive to governance structures, policy implementation, resource allocation, institutional coordination, and lessons learned from previous outbreaks such as Ebola.
    You may underestimate the everyday realities and emotional experiences of patients and frontline health workers. 
    """.strip(),

    "patient": """
    You are a community member in your late 40s who relied on healthcare services before and during the COVID-19 pandemic. You experienced changes in healthcare access, public health restrictions, and communication from health authorities, while observing how your community responded to infectious disease outbreaks.
    As a patient, you frequently used the healthcare system, and you bring a lived-experience perspective when coding a transcript.
    You value accessible healthcare, trust, clear communication, and respectful treatment.
    You are especially attentive to barriers to care, experiences of healthcare services, public perceptions, trust in institutions, fear, uncertainty, and community responses to disease outbreaks.
    You may underestimate organizational, policy, or resource constraints faced by health systems 
    """.strip(),

    "qualitative_researcher": """
    You are a qualitative health researcher in your mid-50s who have spent over 15 years conducting qualitative studies involving patients, caregivers, and healthcare professionals. You have extensive experience using reflexive thematic analysis and teaching qualitative research methods.
    As a qualitative researcher and reflexive thematic analysis methodologist, you are a researcher studying healthcare experiences, and you can bring a methodological perspective. 
    You value reflexivity and interpretive transparency.
    You are especially attentive to interpretive transparency and reflexivity.
    You may underestimate operational healthcare challenges or policy constraints that are not explicitly discussed in the transcript.
    """.strip()
}

### Generate themes based on codebook  with clusters
# A. generate clusters for code + all clues under that codebook entry
def load_units_from_codebook(codebook_path):
    units = []

    with open(codebook_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data["entries"]

    for i, entry in enumerate(entries):
        code = entry["initial code"].strip()

        clues_list = []
        for clue in entry["clues"]:
            if clue:
                clues_list.append(clue)

        if not code or not clues_list:
            continue

        combined_clues = " ".join(clues_list)
        unit_text = f"Initial code: {code} | Supporting quotes: {combined_clues}"

        units.append({
            "entry_id": i,
            "initial code": code,
            "clues": clues_list,
            "transcript_ids": entry.get("transcript_ids", []),
            "code_clues": unit_text
        })

    return units

def _cluster_embeddings_agglomerative(E, distance_threshold, linkage="complete"):
    c = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage=linkage,
        metric="cosine"
    )
    labels = c.fit_predict(E)

    groups = defaultdict(list)
    for i, lab in enumerate(labels):
        groups[int(lab)].append(i)

    return labels, groups

def iterative_clusters(
    codebook_path,
    out_path,
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    distance_threshold=0.40, # similarity = 0.6
    linkage="complete",
    stop_ratio=0.90,
    max_rounds=5
):
    units = load_units_from_codebook(codebook_path)
    if not units:
        raise ValueError("No units loaded from codebook.json.")

    model = SentenceTransformer(model_name)
    X = model.encode(
        [u["code_clues"] for u in units],
        normalize_embeddings=True
    )

    membership = {i: [i] for i in range(len(units))}
    rounds = []
    current_embeddings = X
    current_ids = list(range(len(units)))
    prev_k = None

    for r in range(1, max_rounds + 1):
        labels, groups = _cluster_embeddings_agglomerative(
            current_embeddings,
            distance_threshold=distance_threshold,
            linkage=linkage
        )
        k = len(groups)

        if prev_k is not None and (k / prev_k) >= stop_ratio:
            break

        rounds.append({
            "round": r,
            "n_items_in": len(current_ids),
            "n_clusters_out": k
        })

        prev_k = k

        new_embeddings = []
        new_membership = {}
        new_ids = []

        for new_cid, idxs in enumerate(sorted(groups.values(), key=len, reverse=True)):
            rows = current_embeddings[idxs]
            centroid = rows.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

            new_embeddings.append(centroid)

            orig = []
            for j in idxs:
                orig.extend(membership[current_ids[j]])
            new_membership[new_cid] = orig
            new_ids.append(new_cid)

        current_embeddings = np.vstack(new_embeddings)
        membership = new_membership
        current_ids = new_ids

    cluster_list = []
    for cid, orig_idxs in sorted(membership.items(), key=lambda x: len(x[1]), reverse=True):
        items = [units[i] for i in orig_idxs]
        cluster_list.append({
            "cluster_id": int(cid),
            "n_items": len(items),
            "items": [
                {
                    "initial code": it["initial code"],
                    "clues": it["clues"],
                    "transcript_ids": it["transcript_ids"]
                }
                for it in items
            ]
        })

    out = {
        "n_units": len(units),
        "n_final_clusters": len(cluster_list),
        "rounds": rounds,
        "clusters": cluster_list
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    return out_path

def pack_small_clusters_by_similarity(
    in_path,
    out_path,
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    small_threshold=10,
    limit=10,
    min_cluster_size=5
):
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    clusters = data["clusters"]
    big = [c for c in clusters if c["n_items"] >= small_threshold]
    small = [c for c in clusters if c["n_items"] < small_threshold]

    if not small:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return out_path

    model = SentenceTransformer(model_name)

    def cluster_centroid(cluster):
        texts = []
        for it in cluster["items"]:
            combined_clues = " ".join(it["clues"])
            texts.append(f'Initial code: {it["initial code"]} | Supporting quotes: {combined_clues}')
        emb = model.encode(texts, normalize_embeddings=True)
        v = emb.mean(axis=0)
        return v / (np.linalg.norm(v) + 1e-12)

    # --- Greedy packing of small clusters ---
    centroids = [cluster_centroid(c) for c in small]
    sizes = [c["n_items"] for c in small]

    C = np.vstack(centroids)
    S = cosine_similarity(C)
    np.fill_diagonal(S, -1.0)

    remaining = set(range(len(small)))
    groups = []

    while remaining:
        seed = max(remaining, key=lambda i: sizes[i])
        group = [seed]
        total = sizes[seed]
        remaining.remove(seed)

        while True:
            best = None
            best_sim = -1

            for j in remaining:
                if total + sizes[j] <= limit:
                    sim = max(S[j, g] for g in group)
                    if sim > best_sim:
                        best_sim = sim
                        best = j

            if best is None:
                break

            group.append(best)
            total += sizes[best]
            remaining.remove(best)

        groups.append(group)

    packed = []
    for gid, g in enumerate(groups):
        merged_items = []
        for idx in g:
            merged_items.extend(small[idx]["items"])
        packed.append({
            "cluster_id": f"packed_small_{gid}",
            "n_items": len(merged_items),
            "items": merged_items,
        })

    new_clusters = big + packed

    # --- Post-merge: absorb any cluster still below min_cluster_size ---
    changed = True
    while changed:
        changed = False
        too_small = [c for c in new_clusters if c["n_items"] < min_cluster_size]
        if not too_small:
            break

        others = [c for c in new_clusters if c["n_items"] >= min_cluster_size]
        if not others:
            # everything is small — nothing to merge into, just keep as-is
            break

        other_centroids = np.vstack([cluster_centroid(c) for c in others])

        for ts in too_small:
            ts_centroid = cluster_centroid(ts).reshape(1, -1)
            sims = cosine_similarity(ts_centroid, other_centroids)[0]
            best_idx = int(np.argmax(sims))

            others[best_idx]["items"].extend(ts["items"])
            others[best_idx]["n_items"] = len(others[best_idx]["items"])
            new_clusters.remove(ts)
            changed = True

    # --- Finalize ---
    new_clusters.sort(key=lambda x: x["n_items"], reverse=True)
    for new_id, c in enumerate(new_clusters, start=1):
        c["cluster_id"] = new_id

    out = {
        "n_clusters_before_packing": len(clusters),
        "n_clusters_after_packing": len(new_clusters),
        "clusters": new_clusters,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    return out_path

def add_global_code_ids(in_path, out_path, start_id=1000):
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    clusters = data["clusters"]
    counter = start_id

    for cluster in clusters:
        for item in cluster["items"]:
            item["code_id"] = f"C{counter}"
            counter += 1

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return out_path

def prepare_theme_clusters(
    codebook_path,
    output_dir,
    verbose=True
):
    os.makedirs(output_dir, exist_ok=True)
    cluster_out_path = os.path.join(output_dir,"clusters.json")
    packed_out_path = os.path.join(output_dir,"packed_clustered.json")
    final_clusters_path = os.path.join(output_dir,"clustered_codes_with_ids.json")

    # 1. Initial clustering
    iterative_clusters(codebook_path=codebook_path,out_path=cluster_out_path)

    # 2. Pack small clusters
    pack_small_clusters_by_similarity(in_path=cluster_out_path,out_path=packed_out_path)

    # 3. Assign stable code IDs
    add_global_code_ids(in_path=packed_out_path,out_path=final_clusters_path)

    if verbose:
        with open(final_clusters_path,"r",encoding="utf-8") as f:
            clusters = json.load(f).get("clusters",[])
        print(
            f"Total clusters: "
            f"{len(clusters)}\n"
        )
        for c in clusters:
            print(
                f"  Cluster "
                f"{c['cluster_id']}: "
                f"{c['n_items']} codes"
            )

    return final_clusters_path

def clusters_to_text(in_path):
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = {}

    for c in data["clusters"]:
        cid = int(c.get("cluster_id"))
        items = c.get("items", [])

        lines = []

        for it in items:
            code_id = str(it.get("code_id", "")).strip()
            code = str(it.get("initial code", "")).strip()
            clues = it.get("clues", [])

            if not isinstance(clues, list):
                clues = [clues]

            lines.append(f"- Initial Code: {code} ({code_id})")
            lines.append("  - Supporting Quotes:")

            for clue in clues:
                clue = str(clue).strip()
                if clue:
                    lines.append(f"    - {clue}")

            lines.append("")

        out[cid] = "\n".join(lines).strip()

    return out

# B. theme agents
def themer(cluster_content, fixed_profile, research_question):
    system_instruction = f"""
    ### IDENTITY PROFILE ###
    {fixed_profile}
    
    ### YOUR TASK ###
    Your job is to examine the group of codes and quotes to find overarching "themes" — patterns of meaning that connect to the research question ({research_question}).
    
    ### KEY REQUIREMENTS for THEMES ###
    1. **Data Grounding**: Themes must be explicitly supported by, and anchored in, data (codes and quotes).
    2. **Analytical Depth**: Themes should go beyond surface-level description to provide nuanced, interpretive insights into the phenomenon.
    3. **Consistency and Distinctiveness**: Each theme should be internally coherent, meaning the codes within it fits together meaningfully, while being clearly distinguishable from other themes. Do not use 'and' to combine distinct concepts together into one theme.
    4. **Relevance to Research Questions**: Themes must directly answer or shed light on the core research questions and objectives.
    5. **Interpret, Don't Just Summarize**: Themes should interpret the data (codes and quotes) rather than simply listing the interview questions.
    6. **Number**: A manageable number of themes (generally up to 3) is preferred.

    ### EXAMPLES of CODING to THEME ###
    1. Codes: "Frustration with UI," "Confusing menu," "Long onboarding" --> Theme: "Poor User Experience".
    2. Codes: "Low salary," "No bonus," "Unpaid overtime" --> Theme: "Compensation Dissatisfaction." 

    ### OUTPUT FORMAT (STRICT)
    Return ONLY a JSON object (no Markdown, no ``` fences, no extra text).
    Schema:
    {{
    "candidate_themes": [
        {{
        "theme_name": "A clear, descriptive name that reflects its meaning and distinction from others",
        "definition": "Define the core concept of the theme",
        "related_initial_codes": ["C1000", "C1001"],
        "representative_quotes": ["verbatim supporting quote"]
        }}
    ]
    }}
    Rules:
    - Every input code_id MUST appear exactly once across all related_initial_codes lists.
    - A code_id cannot be assigned to more than one theme.
    - Do not omit any input code_id.
    - Do not invent any code_id that is not present in the input.
    - representative_quotes must be copied verbatim from the supplied Supporting Quotes.
    - Output must start with {{ and end with }}.
    - Do not include ```json or ``` anywhere.
    """
    
    user_content = f"""
    ### INPUT DATA 
    {cluster_content}
    """
    messages = [
    {"role": "system", "content": system_instruction.strip()},
    {"role": "user", "content": user_content.strip()}
    ]
    return messages

def healthworker_themer(clusters,research_question):
    return themer(
        cluster_content=clusters,
        fixed_profile=theme_agent_profiles["healthworker"],
        research_question=research_question
    )

def policy_maker_themer(clusters,research_question):
    return themer(
        cluster_content=clusters,
        fixed_profile=theme_agent_profiles["policy_maker"],
        research_question=research_question
    )

def patient_themer(clusters,research_question):
    return themer(
        cluster_content=clusters,
        fixed_profile=theme_agent_profiles["patient"],
        research_question=research_question
    )

def qualitative_researcher_themer(clusters,research_question):
    return themer(
        cluster_content=clusters,
        fixed_profile=theme_agent_profiles["qualitative_researcher"],
        research_question=research_question
    )

themer_functions = {
    "healthworker": healthworker_themer,
    "policy_maker": policy_maker_themer,
    "patient": patient_themer,
    "qualitative_researcher": qualitative_researcher_themer
}

# make sure code coverage complete
def build_expected_codes(final_clusters_path):
    with open(final_clusters_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    expected = {}
    for c in data["clusters"]:
        cid = int(c["cluster_id"])
        expected[cid] = set(it["code_id"] for it in c.get("items", []) if it.get("code_id"))
    return expected

def check_theme_coverage(codebook_results, expected_codes):
    reports = []

    for entry in codebook_results:
        cid = int(entry["cluster_id"])
        expected = set(str(x).strip() for x in expected_codes.get(cid, set()) if str(x).strip())

        cb = entry.get("codebook", {}) or {}
        themes = cb.get("candidate_themes", []) or []

        returned_list = []

        for t in themes:
            ids = t.get("related_initial_codes", []) or []
            if not isinstance(ids, list):
                ids = [ids]

            for code_id in ids:
                code_id = str(code_id).strip()
                if code_id:
                    returned_list.append(code_id)

        returned_set = set(returned_list)
        counts = Counter(returned_list)

        duplicates = sorted([code_id for code_id, n in counts.items() if n > 1])

        missing = sorted(expected - returned_set)
        extra = sorted(returned_set - expected)

        covered = len(expected) - len(missing)
        coverage = covered / max(len(expected), 1)
        missing_rate = len(missing) / max(len(expected), 1)

        reports.append({
            "cluster_id": cid,
            "n_expected": len(expected),
            "n_returned_total": len(returned_list),
            "n_returned_unique": len(returned_set),
            "n_missing": len(missing),
            "n_extra": len(extra),
            "n_duplicate": len(duplicates),
            "coverage": coverage,
            "missing_rate": missing_rate,
            "missing": missing,
            "extra": extra,
            "duplicates": duplicates
        })

    reports.sort(
        key=lambda x: (x["n_missing"], x["n_extra"], x["n_duplicate"], -x["n_expected"]),
        reverse=True
    )

    return reports

def extract_missing_cluster_content(missing_report, cluster_content):
    out = {}

    for cid, miss_ids in missing_report:
        cid = int(cid)
        cluster_content_full = cluster_content.get(cid, "")

        code_ids_miss = {str(c).strip() for c in miss_ids if str(c).strip()}

        parts = re.split(
            r'(?m)(?=^\s*-\s*Initial Code:\s*)',
            cluster_content_full
        )

        parts = [p for p in parts if p.lstrip().startswith("- Initial Code:")]

        kept_blocks = []

        for block in parts:
            m = re.search(r'\((C\d+)\)', block)
            if m and m.group(1) in code_ids_miss:
                kept_blocks.append(block.rstrip())

        out[cid] = "\n".join(kept_blocks).strip()

    return out

def repairer(cluster_miss_content, prior_json, missing_ids, fixed_profile, research_question):
    system_instruction = f"""
        ### IDENTITY PROFILE ###
        {fixed_profile}

        ### YOUR TASK ###
        You will receive:
        (1) The previous generated candidate themes in JSON output
        (2) A collection of missed "Initial Codes" and "Clues" (quotes) in that previous JSON output
        (3) A list of corresponding missed initial code ids that MUST be included.
        Your job is to place all these missed initial codes into candidate themes.

        ### REPAIR PROTOCOL ###
        1. Read missed initial codes and clues.
        2. Compare them against the 'PRIOR OUTPUT JSON'.
        3. If a missed code fits perfectly into an existing definition, append it.
        4. If it represents a new pattern, create a new candidate theme that is related to {research_question} and follows below theme requirements.
        5. Ensure a manageable total number of themes (generally up to 3).

        ### KEY REQUIREMENTS for THEMES ###
        1. **Data Grounding**: Themes must be explicitly supported by, and anchored in, data (codes and quotes).
        2. **Analytical Depth**: Themes should go beyond surface-level description to provide nuanced, interpretive insights into the phenomenon.
        3. **Consistency and Distinctiveness**: Each theme should be internally coherent, meaning the codes within it fits together meaningfully, while being clearly distinguishable from other themes. Do not use 'and' to combine distinct concepts together into one theme.
        4. **Relevance to Research Questions**: Themes must directly answer or shed light on the core research questions and objectives.
        5. **Interpret, Don't Just Summarize**: Themes should interpret the data (codes and quotes) rather than simply listing the interview questions.
        
        ### EXAMPLES of CODING to THEME ###
        1. Codes: "Frustration with UI," "Confusing menu," "Long onboarding" --> Theme: "Poor User Experience".
        2. Codes: "Low salary," "No bonus," "Unpaid overtime" --> Theme: "Compensation Dissatisfaction." 

        ### Rules ###
        - Return ONLY a JSON object with the SAME schema:
        {{"candidate_themes":[{{"theme_name":"...", "definition":"...", "related_initial_codes":["C1000"], "representative_quotes":["..."]}}]}}
        - You MUST include every missing initial code id in related_initial_codes.
        - You may (a) add missing code_ids to an existing theme, or (b) create new candidate theme(s). 
        - Do NOT invent new code_ids.
        - Output must start with {{ and end with }}.
        - Do not include ```json or ``` anywhere in output.
        """

    user_content = f"""
        ### PRIOR OUTPUT JSON
        {json.dumps(prior_json, ensure_ascii=False)}

        ### MISSED INITIAL CODES & CLUES 
        {cluster_miss_content}

        ### CORRESPONDING MISSED INITIAL CODE IDs (MUST BE INCLUDED)
        {missing_ids}

        Return corrected JSON ONLY.
        """

    return [
        {"role": "system", "content": system_instruction.strip()},
        {"role": "user", "content": user_content.strip()},
    ]

def healthworker_repairer(
    cluster_miss_content,
    prior_json,
    missing_ids,
    research_question
):
    return repairer(
        cluster_miss_content=cluster_miss_content,
        prior_json=prior_json,
        missing_ids=missing_ids,
        fixed_profile=theme_agent_profiles["healthworker"],
        research_question=research_question
    )

def policy_maker_repairer(
    cluster_miss_content,
    prior_json,
    missing_ids,
    research_question
):
    return repairer(
        cluster_miss_content=cluster_miss_content,
        prior_json=prior_json,
        missing_ids=missing_ids,
        fixed_profile=theme_agent_profiles["policy_maker"],
        research_question=research_question
    )

def patient_repairer(
    cluster_miss_content,
    prior_json,
    missing_ids,
    research_question
):
    return repairer(
        cluster_miss_content=cluster_miss_content,
        prior_json=prior_json,
        missing_ids=missing_ids,
        fixed_profile=theme_agent_profiles["patient"],
        research_question=research_question
    )

def qualitative_researcher_repairer(
    cluster_miss_content,
    prior_json,
    missing_ids,
    research_question
):
    return repairer(
        cluster_miss_content=cluster_miss_content,
        prior_json=prior_json,
        missing_ids=missing_ids,
        fixed_profile=theme_agent_profiles["qualitative_researcher"],
        research_question=research_question
    )

repairer_functions = {
    "healthworker": healthworker_repairer,
    "policy_maker": policy_maker_repairer,
    "patient": patient_repairer,
    "qualitative_researcher": qualitative_researcher_repairer
}

def run_theme_agents_single_round(
    final_clusters_path,
    output_dir,
    research_question,
    agents,
    no_progress_limit=10,
    verbose=True
):
    os.makedirs(output_dir, exist_ok=True)

    txt_dict = clusters_to_text(final_clusters_path)
    expected_map = build_expected_codes(final_clusters_path)

    all_agent_results = {}
    for agent in agents:
        if verbose:
            print("\n" + "=" * 80)
            print(f"Running theme agent: {agent}")
            print("=" * 80)

        if agent not in themer_functions:
            raise ValueError(f"Unknown theme agent: {agent}")

        theme_fn = themer_functions[agent]
        repair_fn = repairer_functions[agent]

        agent_output_dir = os.path.join(output_dir, agent)
        os.makedirs(agent_output_dir, exist_ok=True)

        def save(results, reports, prefix):
            path_r = os.path.join(agent_output_dir, f"{prefix}_theme_results.json")
            path_c = os.path.join(agent_output_dir,f"{prefix}_coverage_report.json")

            for path, obj in [(path_r, results),(path_c, reports)]:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(obj,f,indent=2,ensure_ascii=False)

        # Initial generation
        codebook_by_cid = {}
        failed_init = []

        for cid, cluster_text in txt_dict.items():
            messages = theme_fn(cluster_text, research_question)
            response = get_completion(messages)
            time.sleep(2)
            raw = response.choices[0].message.content.strip()
            try:
                parsed = json.loads(raw)
                codebook_by_cid[int(cid)] = parsed
            except json.JSONDecodeError:
                failed_init.append({
                    "cluster_id": cid,
                    "stage": "init_json_decode",
                    "raw": raw[:200]
                })
                codebook_by_cid[int(cid)] = {"candidate_themes": []}

        def snapshot():
            results = [{"cluster_id": cid,"codebook": cb} for cid, cb in codebook_by_cid.items()]
            reports = check_theme_coverage(results, expected_map)
            return results, reports

        initial_results, initial_reports = snapshot()

        total_expected = sum(r["n_expected"] for r in initial_reports)
        total_missing = sum(r["n_missing"] for r in initial_reports)
        overall_missing_rate = (total_missing / max(total_expected, 1))
        if verbose:
            print(f"[{agent}] Before repair, " 
                  f"missing rate: {overall_missing_rate}")

        save(initial_results,initial_reports,"before_repair")

        # Repair
        failed_repair = [] # record repair failures (error log)
        it = 0
        no_progress = 0
        prev_missing_total = None
        while True:
            it += 1
            current_results, reports = snapshot()
            bad_reports = [r for r in reports if r["n_missing"] > 0]
            if not bad_reports:
                if verbose:
                    print(
                        f"[{agent}] DONE. "
                        f"All clusters reached coverage=1."
                    )
                break

            missing_total = sum(r["n_missing"] for r in bad_reports)

            if verbose:
                print(
                    f"[{agent} Iter {it}] "
                    f"bad_clusters={len(bad_reports)} "
                    f"total_missing={missing_total}"
                )

            if prev_missing_total is not None and missing_total >= prev_missing_total:
                no_progress += 1 # get worse
            else:
                no_progress = 0
            prev_missing_total = missing_total

            if no_progress_limit is not None and no_progress >= no_progress_limit:
                if verbose:
                    print(
                        f"[{agent}] STOP: "
                        f"No progress for "
                        f"{no_progress_limit} iterations."
                    )
                break

            zipped = [(r["cluster_id"], r["missing"]) for r in bad_reports]
            miss_content_map = (extract_missing_cluster_content(zipped,txt_dict))

            for cid, miss_ids in zipped:
                prior_json = codebook_by_cid.get(int(cid))
                if prior_json is None:
                    failed_repair.append({"cluster_id": cid, "iter": it, "stage": "missing_prior_codebook"})
                    continue

                cluster_miss_content = (miss_content_map.get(int(cid), ""))
                messages = repair_fn(cluster_miss_content, prior_json, miss_ids, research_question)
                response = get_completion(messages)
                time.sleep(2)
                raw = response.choices[0].message.content.strip()
                try:
                    codebook_by_cid[int(cid)] = (json.loads(raw))
                except json.JSONDecodeError:
                    failed_repair.append({
                        "cluster_id": cid,
                        "iter": it,
                        "stage":
                            "repair_json_decode",
                        "raw": raw[:200]
                    })

        final_results, final_reports = snapshot()
        save(final_results, final_reports, "after_repair")

        all_agent_results[agent] = {
            "initial_results": initial_results,
            "initial_reports": initial_reports,
            "final_results": final_results,
            "final_reports": final_reports,
            "failed_init": failed_init,
            "failed_repair": failed_repair
        }

    return all_agent_results

def run_multiple_theme_agents_single_round_pipeline(
    codebook_path,
    output_dir,
    research_question,
    agents,
    no_progress_limit=10,
    verbose=True
):
    os.makedirs(output_dir, exist_ok=True)
    cluster_output_dir = os.path.join(output_dir, "clusters")
    agent_output_dir = os.path.join(output_dir, "theme_agents")

    final_clusters_path = prepare_theme_clusters(
        codebook_path=codebook_path,
        output_dir=cluster_output_dir,
        verbose=verbose,
    )

    final_agent_results = run_theme_agents_single_round(
        final_clusters_path=final_clusters_path,
        output_dir=agent_output_dir,
        research_question=research_question,
        agents=agents,
        no_progress_limit=no_progress_limit,
        verbose=verbose,
    )

    return {
        "final_clusters_path": final_clusters_path,
        "final_agent_results": final_agent_results,
    }
