import os, json, re, time
from AzureUtils import get_completion

"""
Purpose:
    Update each coder agent's reflexive thinking after comparing its coding results with peer coder outputs.

Use this file when:
    - Comparing coding results across coder agents
    - Generating updated assumptions and further considerations
    - Recording reflexive history between coding rounds
    - Loading the latest reflection for the next coding round

all chunks coded → one *_final.json per agent/transcript/round → compare four agents’ final coding outputs → update each agent’s reflexive thinking → save a versioned updated reflection
"""
REFLECTION_OUTPUT_BASE_DIR = ("/ReflexTheme/updated_reflections")

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

def run_reflection_update(target_agent, fixed_profile, transcript_id, round_num, base_path, agents):
   peer_agents = [a for a in agents if a != target_agent]

   target_file = os.path.join(base_path, target_agent, f"{transcript_id}_round_{round_num}_final.json") # all chunks of a transcript for a agent in a round
    
   if not os.path.exists(target_file):
      print(f"Error: Target file for {target_agent} not found at {target_file}")
      return None

   with open(target_file, 'r', encoding='utf-8') as f:
      target_data = json.load(f)

   my_codes = target_data.get("coding_output", "")
   my_reflection = target_data.get("reflection", {})

   peer_codes_list = []
   for peer in peer_agents:
      peer_file = os.path.join(base_path, peer, f"{transcript_id}_round_{round_num}_final.json")
      if os.path.exists(peer_file):
         with open(peer_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            peer_codes_list.append(f"--- {peer.upper()} PERSPECTIVE ---\n{data['coding_output']}")
      else:
         print("No peer results found!")
    
   peer_codes_content = "\n\n".join(peer_codes_list)

   messages = agent_self_reflection_update(
      fixed_profile=fixed_profile,
      my_codes=my_codes,
      peer_codes=peer_codes_content,
      current_reflection=my_reflection
   )
   response = get_completion(messages)
   time.sleep(2)
   updated_res = response.choices[0].message.content.strip()

   output_dir = os.path.join(REFLECTION_OUTPUT_BASE_DIR, target_agent)
   os.makedirs(output_dir, exist_ok=True)

   pattern = r"reflection_v(\d+)\.json"
   existing_files = [f for f in os.listdir(output_dir) if re.match(pattern, f)]
    
   if not existing_files:
      next_version = 1
   else:
      version_numbers = [int(re.search(pattern, f).group(1)) for f in existing_files]
      next_version = max(version_numbers) + 1

   output_path = os.path.join(output_dir, f"reflection_v{next_version}.json")

   json_str = updated_res.replace('```json', '').replace('```', '').strip()
   structured_update = json.loads(json_str)

   with open(output_path, "w", encoding='utf-8') as f:
      json.dump(structured_update, f, indent=4, ensure_ascii=False)

def get_latest_reflection(agent_name):
   agent_ref_dir = os.path.join(REFLECTION_OUTPUT_BASE_DIR, agent_name)
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
