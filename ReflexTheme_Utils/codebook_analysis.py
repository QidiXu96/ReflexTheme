import json
import os
import glob
import copy
import re
import matplotlib.pyplot as plt

BASE_DIR = "/ReflexTheme"

OPTION_DIRS = {
    3: os.path.join(BASE_DIR, "option_3_code_clue_threshold")
}

OPTION_LABELS = {
    3: "Code+Clue + Threshold"
}

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_entries(data):
    if isinstance(data, dict):
        return data.get("entries", [])
    elif isinstance(data, list):
        return data
    else:
        return []

def extract_transcript_id(path):
    filename = os.path.basename(path)
    match = re.search(r"codebook_after_(\d+)\.json", filename)
    return match.group(1) if match else filename

def count_unique_transcripts(entry):
    transcript_ids = entry.get("transcript_ids", [])

    if not isinstance(transcript_ids, list):
        transcript_ids = [transcript_ids]

    unique_transcripts = set(
        str(t).strip()
        for t in transcript_ids
        if str(t).strip()
    )

    return len(unique_transcripts)

def plot_codebook_size_growth(option_dirs, option_labels=None):
    all_results = {}

    plt.figure(figsize=(9, 6))

    for option_id, option_dir in option_dirs.items():

        files = sorted(glob.glob(os.path.join(option_dir, "codebook_after_*.json")))

        if not files:
            print(f"Option {option_id}: no codebook_after_XXXX.json files found.")
            continue

        transcript_ids = []
        codebook_sizes = []

        for path in files:
            data = load_json(path)
            entries = get_entries(data)

            transcript_ids.append(extract_transcript_id(path))
            codebook_sizes.append(len(entries))

        label = option_labels.get(option_id, f"Option {option_id}") if option_labels else f"Option {option_id}"

        plt.plot(
            range(1, len(codebook_sizes) + 1),
            codebook_sizes,
            marker="o",
            label=label
        )

        all_results[option_id] = {
            "transcript_ids": transcript_ids,
            "codebook_sizes": codebook_sizes
        }

    plt.xlabel("Transcript")
    plt.ylabel("Codebook Size")
    plt.title("Codebook Size Growth")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return all_results


def final_codebook_single_transcript_percentage(option_dirs, option_labels=None):
    """
    For each final codebook.json, calculate:
    - total initial codes
    - number of initial codes with only one unique transcript
    - percentage
    - number of codes after removing single-transcript codes
    """
    results = {}

    print("\nHF Final Codebook Single-Transcript Summary")
    print("=" * 105)
    print(
        f"{'Option':<35} "
        f"{'Total Codes':>12} "
        f"{'Single-TID Codes':>18} "
        f"{'Percentage':>12}"
        f"{'After Removing':>18} "
    )
    print("-" * 105)

    for option_id, option_dir in option_dirs.items():

        final_path = os.path.join(option_dir, "codebook.json")

        label = option_labels.get(option_id, f"Option {option_id}") if option_labels else f"Option {option_id}"

        if not os.path.exists(final_path):
            print(f"{label:<35} {'Missing codebook.json':>12}")
            continue

        data = load_json(final_path)
        entries = get_entries(data)

        total_codes = len(entries)
        n_single = 0

        for entry in entries:
            transcript_ids = entry.get("transcript_ids", [])

            if not isinstance(transcript_ids, list):
                transcript_ids = [transcript_ids]

            unique_transcripts = set(
                str(t).strip()
                for t in transcript_ids
                if str(t).strip()
            )

            if len(unique_transcripts) == 1:
                n_single += 1

        n_after_removing_single = total_codes - n_single
        pct_single = (n_single / total_codes * 100) if total_codes > 0 else 0

        print(
            f"{label:<35} "
            f"{total_codes:>12} "
            f"{n_single:>18} "
            f"{pct_single:>11.2f}%"
            f"{n_after_removing_single:>18} "
        )

        results[option_id] = {
            "option_label": label,
            "total_codes": total_codes,
            "n_single_transcript_codes": n_single,
            "pct_single_transcript_codes": pct_single,
            "n_codes_after_removing_single_transcript_codes": n_after_removing_single
        }

    return results

def remove_embeddings_from_all_final_codebooks(
    option_dirs,
    option_labels=None,
    input_filename="codebook.json",
    output_filename="codebook_no_embedding.json"
):
    results = {}

    for option_id, option_dir in option_dirs.items():

        label = (
            option_labels.get(option_id, f"Option {option_id}")
            if option_labels
            else f"Option {option_id}"
        )

        input_path = os.path.join(option_dir, input_filename)
        output_path = os.path.join(option_dir, output_filename)

        if not os.path.exists(input_path):
            print(f"{label:<35} {'Missing input file':>15}")
            results[option_id] = {
                "option_label": label,
                "status": "missing_input_file",
                "input_path": input_path,
                "output_path": None
            }
            continue

        data = load_json(input_path)
        clean_data = copy.deepcopy(data)

        entries = get_entries(clean_data)

        for entry in entries:
            if isinstance(entry, dict):
                entry.pop("embedding", None)

        save_json(clean_data, output_path)

        results[option_id] = {
            "option_label": label,
            "status": "saved",
            "input_path": input_path,
            "output_path": output_path,
            "n_codes": len(entries)
        }

    return results

def remove_embeddings_and_single_transcript_codes_from_all_final_codebooks(
    option_dirs,
    option_labels=None,
    input_filename="codebook.json",
    output_filename="codebook_no_embedding_no_single_transcript.json"
):
    results = {}

    for option_id, option_dir in option_dirs.items():

        label = (
            option_labels.get(option_id, f"Option {option_id}")
            if option_labels
            else f"Option {option_id}"
        )

        input_path = os.path.join(option_dir, input_filename)
        output_path = os.path.join(option_dir, output_filename)

        if not os.path.exists(input_path):
            print(f"{label:<35} {'Missing input file':>10}")
            results[option_id] = {
                "option_label": label,
                "status": "missing_input_file",
                "input_path": input_path,
                "output_path": None
            }
            continue

        data = load_json(input_path)
        clean_data = copy.deepcopy(data)

        entries = get_entries(clean_data)

        filtered_entries = []

        for entry in entries:
            if not isinstance(entry, dict):
                continue

            entry.pop("embedding", None)

            n_unique_transcripts = count_unique_transcripts(entry)

            # keep only codes appearing in more than one unique transcript
            if n_unique_transcripts > 1:
                filtered_entries.append(entry)

        original_n = len(entries)
        remaining_n = len(filtered_entries)
        removed_n = original_n - remaining_n
        pct_removed = (removed_n / original_n * 100) if original_n > 0 else 0

        if isinstance(clean_data, dict):
            clean_data["entries"] = filtered_entries
        else:
            clean_data = filtered_entries

        save_json(clean_data, output_path)

        results[option_id] = {
            "option_label": label,
            "status": "saved",
            "input_path": input_path,
            "output_path": output_path,
            "original_n_codes": original_n,
            "removed_single_transcript_codes": removed_n,
            "remaining_codes": remaining_n,
            "pct_removed_single_transcript_codes": pct_removed
        }

    return results