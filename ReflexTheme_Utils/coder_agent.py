"""
Generic ReflexCoTI coder agent utilities.

Each coder is supplied as a structured profile.
The profile has three layers:

1. identity_core (fixed):
   - role
   - positional statement
   - synthetic biography
   - values
   - interpretive strengths
   - baseline blind spots

2. reflexive_layer (revisable):
   - working assumptions: claims or expectations (can be provided initially)
   - reflexive questions uncertainties: areas requiring further consideration (should be generated via collaboration)

3. analytic_record:
   - coding reflection thinking history (thought process)
"""

import json, os, re
from backend.AzureUtils import get_completion

def coder_agent(
    transcript,
    fixed_profile,
    reflection_content,
    coding_memory_content=None,
    research_question=None,
    interviewee=None
):
    if not research_question:
        raise ValueError("research question must be provided.")
    if not interviewee:
        raise ValueError("interviewee must be provided.")
    ### previous coding results (avaliable for 2+ round) ###
    coding_memory = ""

    if coding_memory_content:
        coding_memory = f"""
            ### YOUR PREVIOUS CODE RESULTS FOR THIS TRANSCRIPT CHUNK ###
            The following coding results were generated in an earlier round for this same transcript chunk. 
            Treat them as provisional interpretations, not labels that must be preserved.
            Return to the original transcript and re-evaluate both the clues and the Initial Code names using your current reflexive thinking.
            If a previous code/clue is too broad, abstract, outside the research question, or combines multiple phenomena, revise, rename, split, or remove it when warranted.
            Do not copy prior coding results automatically, and do not revise them merely to agree with other coders.

            {coding_memory_content}
            --- END PREVIOUS CODE RESULTS ---
            """

    ### reflexive content ###
    reflection = ""
    reflection_assumptions = ""
    reflection_consider = "N/A"
    reflection_history = "N/A"

    if isinstance(reflection_content, dict):
        reflection_assumptions = reflection_content.get("assumptions", "")
        reflection_consider = reflection_content.get("further_consideration", "N/A")
        reflection_history = reflection_content.get("reflexive_history", "N/A")

    if reflection_history != "N/A"  or reflection_consider != "N/A" or reflection_assumptions :
        reflection = f"""
        ### CURRENT REFLEXIVE THINKING ###
        In reflexive thematic analysis (Braun & Clarke, 2019), your subjectivity is a productive resource. RTA requires you to be 'knowing' and 'reflexive' about your influence on the data.
        Use your identity profile and the reflexive thinking to guide your interpretation.
        - Assumptions: {reflection_assumptions}
        - Areas requiring further consideration: {reflection_consider}
        - Reflection history: {reflection_history}
        Agreement with other coders is not the goal. Retain defensible differences in interpretation. Do not revise your coding merely to achieve consensus.
       """

    ### system prompt ###    
    system_prompt = f"""
        ### IDENTITY PROFILE ###
        {fixed_profile}

        ### TASK ###
        Your task is to code interviews which were conducted with {interviewee}. 
        Our end goal is to perform reflective thematic analysis (RTA) on the provided transcript.
        The steps that should be performed are as follows:
        1. Familiarization - Read and re-read the transcript to become deeply familiar with it.
        2. Coding - When identifying clues (quotes from the given transcript) and initial codes (smallest meaningful units) related to {research_question}, use your own identity profile and reflexive thinking.
        3. Reflection - Collaborate with multiple coders. Reflect on the similarities and differences. If there is a discrepancy, return to earlier steps of the process, conduct further reading and notation of the transcript, and iteratively develop insight more. 
    
        ### Requirements for CLUES ###
        - Be direct quotes from the transcript (no summarization or paraphrasing).
        - Include enough context for the coded phenomenon to be understood, but do not include surrounding sentences that introduce unrelated phenomena.
        - Each clue should center on ONE specific phenomenon relevant to the research question.
        - Each clue must be a continuous excerpt from a single location in the transcript.
        - If the same code is supported by separate passages, create separate code entries. Never combine non-contiguous text into one clue.

        ### Requirements for INITIAL CODES ###
        - The Initial Code should Only capture ONE specific action, experience, barrier, interaction, or consequence shown in the clue. And the Initial Code must be directly supported by its Clues. Do not introduce an action, experience, cause, consequence, or interpretation that is not reasonably supported by that excerpt. 
        - Generate the Initial Code from the specific phenomenon expressed in the Clue, not from the general topic of the transcript.
        - Avoid broad, abstract, or umbrella labels that could contain several different phenomena or only name a general topic.
        - Only identify codes that are directly relevant to the research question. Do not code information outside the research question simply because it is interesting or meaningful.

        Examples:
        Bad: "Caregiver involvement" — too broad, it names a general role but does not describe what the caregiver actually does.
        Good: "Caregiver supplements or explains the patient's answers"

        Bad: "Digital communication" - topic only, it identifies the communication channel but not the specific problem, action, or consequence.
        Good: "No response from digital portal delays prescribing"

        Bad: "Work stress" — too abstract, it labels an overall experience without showing the specific source or form of the stress.
        Good: "Employee feels overwhelmed by competing deadlines"

        Bad: "Social isolation" — too general; it names the condition but does not describe the participant's specific behavior or context.
        Good: "Participant withdraws from friends after moving to a new city"

        ### PRECISION CONTROL (EVIDENCE BUDGET — IMPORTANT) ###
        Your goal is HIGH RECALL *without* quote sprawl. Do this by selecting **few, high-signal** clues.
        **Per-code evidence limit:** Provide **no more than 3 complete sentences** as clues per Initial Code.

        Before producing the final output, check each Initial Code:
        1. Is it directly relevant to the research question?
        2. Does it represent only ONE specific phenomenon?
        3. Is it directly supported by the corresponding Clue?
        4. Does the name describe what is happening rather than only naming a topic?
        If any answer is no, revise the code before output.

        ### IMPORTANT (FORMAT IS STRICT) ###
        - The Clues field must contain **exactly ONE opening quote (") and ONE closing quote (")**.
        - Do NOT place quotation marks around individual sentences.
        - If you include multiple sentences, merge them into a single quoted paragraph.
        
        ### IMPORTANCE SCORE (REQUIRED) ###
        For each Initial Code, assign an Importance Score (1–5) based on the strength and centrality of its supporting clue:
        - 5 = Critical / central evidence (explicit, unambiguous, high-signal)
        - 4 = Strong evidence (clear link, minor ambiguity)
        - 3 = Moderate (relevant but indirect / needs context)
        - 2 = Weak (tangential / borderline)
        - 1 = Not evidence (should usually be omitted)
        """

    ### user prompt ###
    if reflection:
        user_prompt = f"""
            ### TASK ###
            Perform STEP 3: Return to the transcript chunk and reconsider your coding using your current reflexive thinking.
            Identify all initial codes related to {research_question}.

            ### OUTPUT FORMAT ###
            For EACH identified code, only provide:

            Initial Code: Specific, interpretive name
            Clues: "Sentence 1. Sentence 2. ..."
            Importance Score: 1-5 (write only one integer from 1 to 5)

            ### ORIGINAL TRANSCRIPT ###
            {transcript}
        """
    else:
        user_prompt = f"""
            ### TASK ###
            Perform STEP 2: CODING on the provided transcript chunk.
            Identify all initial codes related to {research_question}.

            ### OUTPUT FORMAT ###
            For EACH identified code, only provide:

            Initial Code: Specific, interpretive name
            Clues: "Sentence 1. Sentence 2. ..."
            Importance Score: 1-5 (write only one integer from 1 to 5)

            ### ORIGINAL TRANSCRIPT ###
            {transcript}
            """

    messages = [
    {"role": "system", "content": system_prompt.strip()},
    {"role": "user", "content": (coding_memory + reflection + user_prompt).strip()}
    ]
    return messages

def run_independent_coding(
    client,
    model,
    transcript,
    agent_name,
    reflection,
    transcript_id,
    chunk_id,
    round_number,
    fixed_profile,
    coding_memory_content=None,
    research_question=None,
    interviewee=None
):
    if not fixed_profile:
        raise ValueError(f"Prepared profile is required for agent '{agent_name}'.")

    if isinstance(reflection, dict) and agent_name in reflection:
        current_reflection = reflection[agent_name]
    else:
        current_reflection = reflection

    messages = coder_agent(
        transcript=transcript,
        fixed_profile=fixed_profile,
        reflection_content=current_reflection,
        coding_memory_content=coding_memory_content,
        research_question=research_question,
        interviewee=interviewee
    )
    response = get_completion(client, model, messages)
    res = response.choices[0].message.content.strip()

    return {
        "agent": agent_name,
        "round": round_number,
        "transcript": transcript_id,
        "chunk": chunk_id,
        "reflection": current_reflection,
        "coding_output": res
    }

def save_coding_result(result, output_base_dir):
    agent_name = result["agent"]
    round_number = result["round"]
    transcript_id = result["transcript"]
    chunk_id = result["chunk"]

    output_dir = os.path.join(output_base_dir, agent_name)
    os.makedirs(output_dir, exist_ok=True)

    file_name = (
        f"round_{round_number}_"
        f"{transcript_id}_"
        f"chunk_{chunk_id}")
    json_path = os.path.join(output_dir, f"{file_name}.json")
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    return json_path

def remove_importance_score(input_base_dir, output_base_dir):
    # input_base_dir: one transcript (all chunks) coding results under a coder agent
    pattern = r'(?im)^[ \t]*Importance\s+Score\s*:\s*\[?\s*[1-5]\s*\]?\s*$\n?'

    for agent in os.listdir(input_base_dir):
        agent_dir = os.path.join(input_base_dir, agent)

        if not os.path.isdir(agent_dir):
            continue

        new_agent_dir = os.path.join(output_base_dir, agent)
        os.makedirs(new_agent_dir, exist_ok=True)

        for fname in os.listdir(agent_dir):
            if not fname.endswith(".json"):
                continue

            file_path = os.path.join(agent_dir, fname)

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            coding_output = data.get("coding_output")

            if isinstance(coding_output, str):
                cleaned = re.sub(pattern, '', coding_output).strip()
                data["coding_output"] = cleaned

                new_path = os.path.join(new_agent_dir, fname)

                with open(new_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
