import os, json, re
from backend.AzureUtils import get_completion

"""
all chunks coded → one *_final.json per agent/transcript/round
→ compare four agents’ final coding outputs
→ update each agent’s reflexive thinking
→ save a versioned updated reflection
"""

def agent_self_reflection_update(fixed_profile, my_codes, peer_codes, current_reflection):
   if isinstance(current_reflection, dict):
        current_reflection = json.dumps(
            current_reflection,
            ensure_ascii=False,
            indent=2
        )

   ### system prompt ###
   system_prompt = f"""
   ### IDENTITY PROFILE ###
   {fixed_profile}. 
   You have completed coding for one transcript.
    
   ### REFLEXIVE THEMATIC ANALYSIS (RTA) GOAL ###
   Your goal is not to eliminate bias, determine which coder is correct, or achieve consensus. 
   Your goal is to make your own interpretive position more explicit and informed after reading peers' coding results.
   Treat peers as alternative interpretive lenses rather than correctness judges. Retain defensible differences in interpretation.

   ### COMPARATIVE REFLEXIVE TASK ###
   Compare your coding with peer coding and identify meaningful differences, including when:
   1. You and a peer coded the same phenomenon differently.
   2. A peer coded a phenomenon that you did not code.

   For each discrepancy:
   - Compare codes that concern the same underlying phenomenon.
   - Explain how your lens contributed to your coding or omission.
   - Use only code names and clue excerpts supplied in the inputs.
   - Do not invent quotations, transcript events, peer motivations, or consensus.
   - Do not force a discrepancy when the codes genuinely overlap.

   ### REQUIRED REFLEXIVE THINKING UPDATES ###
   1. Assumptions
      Revise or qualify your prior assumptions in light of the comparison. Preserve useful assumptions rather than replacing them merely because peers differed.

   2. Areas requiring further consideration
      State unresolved interpretive questions raised by the comparison. These must be genuine uncertainties, not generic statements about being more nuanced.

   3. Reflexive history
      Record this round's discrepancy reflexive analysis. Use numbered entries. For every discrepancy, include these labels exactly:
      [Phenomenon]
      [My code vs Peer code]
      [Evidence]
      [Interpretive difference]
      [Interpretive positioning]
      [Next-round adjustment]

   ### OUTPUT FORMAT (STRICT) ###
   Return ONLY a JSON object (no Markdown, no ``` fences, no extra text).
   {{
   "assumptions": "...",
   "further_consideration": "...",
   "reflexive_history": "..."
   }}
   Rules:
    - Output must start with {{ and end with }}.
    - Do not include ```json or ``` anywhere.
   """
   ### user prompt ###
   user_prompt = f"""
   ### YOUR CURRENT REFLECTION ###
   {current_reflection}

   ### YOUR CODING RESULTS ###
   {my_codes}

   ### PEER CODING RESULTS ###
   {peer_codes}

   ### TASK ###
   Reflect on the differences. Perform comparative critique using the rules above.
   And update your own reflexive thinking. Your next coding round should be guided by the updated reflexive thinking, while preserving your distinct identity profile based perspective.
   """

   messages = [
   {"role": "system", "content": system_prompt.strip()},
   {"role": "user", "content": user_prompt.strip()}
   ]

   return messages

def _parse_json_object(text):
    value = str(text or "").strip()

    # remove markdown fences if present
    value = value.replace("```json", "").replace("```", "").strip()

    # first try normal JSON parsing
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        # fallback: extract the first {...} block
        start = value.find("{")
        end = value.rfind("}")

        if start < 0 or end <= start:
            raise

        json_text = value[start:end + 1]

        # fix invalid backslash escapes such as \'
        json_text = re.sub(
            r'\\(?!["\\/bfnrtu])',
            r'\\\\',
            json_text
        )

        parsed = json.loads(json_text)

    if not isinstance(parsed, dict):
        raise ValueError("Reflection response is not a JSON object.")

    return parsed

def run_reflection_update(client, model, target_agent, fixed_profile, current_reflection, coding_results, agents):
   peer_agents = [a for a in agents if a != target_agent]

   if target_agent not in coding_results:
      raise ValueError(f"No coding results found for agent '{target_agent}'.")
   my_codes = coding_results[target_agent]

   peer_codes_list = []
   for peer in peer_agents:
      if peer not in coding_results:
            continue
      peer_codes_list.append(
            f"--- {peer.upper()} PERSPECTIVE ---\n"
            f"{coding_results[peer]}"
      )
   if not peer_codes_list:
      raise ValueError(f"No peer coding results available for '{target_agent}'.")
   peer_codes_content = "\n\n".join(peer_codes_list)

   messages = agent_self_reflection_update(
      fixed_profile=fixed_profile,
      my_codes=my_codes,
      peer_codes=peer_codes_content,
      current_reflection=current_reflection
   )
   response = get_completion(client, model, messages)
   updated_res = response.choices[0].message.content.strip()

   return _parse_json_object(updated_res)

def save_reflection(
   reflection,
   agent_name,
   output_base_dir
):
   agent_ref_dir = os.path.join(output_base_dir, agent_name)
   os.makedirs(agent_ref_dir, exist_ok=True)
   pattern = r"reflection_v(\d+)\.json$"
   files = [f for f in os.listdir(agent_ref_dir) if re.match(pattern, f)]

   if not files:
      next_version = 1
   else:
      versions = [int(re.search(pattern, f).group(1)) for f in files]
      next_version = max(versions) + 1

   output_path = os.path.join(agent_ref_dir, f"reflection_v{next_version}.json")

   with open(output_path, "w", encoding="utf-8") as f:
      json.dump(reflection, f, indent=4, ensure_ascii=False)
   return output_path

def get_latest_reflection(agent_name, output_base_dir):
   agent_ref_dir = os.path.join(output_base_dir, agent_name)
   if not os.path.exists(agent_ref_dir):
      return None

   pattern = r"reflection_v(\d+)\.json$"
   files = [f for f in os.listdir(agent_ref_dir) if re.match(pattern, f)]

   if not files:
      return None

   files.sort(key=lambda x: int(re.search(pattern, x).group(1)), reverse=True)
   latest_file_path = os.path.join(agent_ref_dir, files[0])
    
   with open(latest_file_path, 'r', encoding='utf-8') as f:
      return json.load(f)