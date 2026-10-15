import json, re, os

def _chunk_sort_key(filename):
    m = re.search(r"_chunk_(\d+)", filename)
    return int(m.group(1)) if m else 10**9

def format_coding_output(obj):
    if obj is None:
        return ""

    if isinstance(obj, str):
        return obj.strip()

    if isinstance(obj, list):
        parts = []
        for it in obj:
            if not isinstance(it, dict):
                continue
            code = (it.get("Initial Code") or "").strip()
            clues = (it.get("Clues") or "").strip()
            if code and clues:
                parts.append(f"Initial Code: {code}\nClues: {clues}")
        return "\n\n".join(parts).strip()

    return ""

def merge_agent_results(agent_name, transcript_id, round_num, base_dir, output_base_dir, suffix="_final"):
    """
    Merge all cleaned chunk-level coding JSON files for one agent, one transcript, and one round.
    """
    agent_dir = os.path.join(base_dir, agent_name)
    prefix = f"round_{round_num}_{transcript_id}_chunk_"

    json_files = [
        f for f in os.listdir(agent_dir)
        if f.startswith(prefix) and f.endswith("_clean.json")
    ]
    json_files.sort(key=_chunk_sort_key)

    if not json_files:
        raise FileNotFoundError(f"No cleanchunk JSONs found for {agent_name} coder agent, transcript {transcript_id}")

    merged_json_parts = []
    reflection_to_save = None

    for i, fn in enumerate(json_files):
        fp = os.path.join(agent_dir, fn)
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        if i == 0:
            reflection_to_save = data.get("reflection")
        coding_output = data.get("coding_output", "")
        coding_text = format_coding_output(coding_output)
        if coding_text:
            merged_json_parts.append(coding_text)

    merged_json_text = "\n\n".join(merged_json_parts)

    merged_json = {
        "agent": agent_name,
        "transcript_id": transcript_id,
        "round": round_num,
        "reflection": reflection_to_save,
        "coding_output": merged_json_text
    }
    
    output_agent_dir = os.path.join(output_base_dir, agent_name)

    os.makedirs(output_agent_dir, exist_ok=True)

    output_path = os.path.join(output_agent_dir, 
    f"{transcript_id}_round_"
    f"{round_num}{suffix}.json"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged_json, f, indent=4, ensure_ascii=False)

    return output_path

# clean json text
def clean_json_text(s):
    if not isinstance(s, str):
        return s

    s = (
        s.replace("**", "")
         .replace("### ", "")
         .replace("###", "")
    )
    s = re.sub(r"(?m)^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", "", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s

def clean_json(obj):
    if isinstance(obj, dict):
        return {
            clean_json_text(k) if isinstance(k, str) else k:
            clean_json(v)
            for k, v in obj.items()
        }

    if isinstance(obj, list):
        return [clean_json(x) for x in obj]

    if isinstance(obj, str):
        return clean_json_text(obj)

    return obj

def process_multiple_json_files(file_paths, suffix="_clean"):
    output_paths = []
    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            cleaned_data = clean_json(data)

            base, ext = os.path.splitext(file_path)
            new_path = f"{base}{suffix}{ext}"

            with open(new_path, "w", encoding="utf-8") as f:
                json.dump(cleaned_data, f, indent=4, ensure_ascii=False)
            output_paths.append(new_path)

        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON chunk output: {file_path}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to clean JSON output: {file_path}"
            ) from e
    return output_paths