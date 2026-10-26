"""
Purpose:
    Define and run the reflexive coder agents used in Stage 1 of ReflexTheme.

Use this file when:
    - Generating initial codes and supporting clues from transcript (research question and interviewee should be modified based on your own research)
    - Running different coder identities (coder agents identify should be modified based on your own research)
    - Performing multiple rounds of coding with reflection and coding memory
    - Saving independent coding results

Each coder is supplied as a structured profile.
The profile has three layers:
1. identity_core (fixed, but should be fitted to your research):
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

import json, os, re, time
from AzureUtils import *

DEFAULT_OUTPUT_BASE_DIR = "/ReflexTheme/independent_coding"
DEFAULT_RESEARCH_QUESTION = "how the pandemic affected service delivery, health workers, patient access to services, leadership, and governance, and what extent the legacy of the 2013–16 Ebola outbreak influenced the COVID-19 response and public perception" # can be changed based on research
DEFAULT_INTERVIEWEE = "health workers, policymakers, and patients between Oct-Dec 2020" # can be changed based on research

def coder_agent(
    transcript,
    fixed_profile,
    reflection_content,
    coding_memory_content=None,
    research_question=DEFAULT_RESEARCH_QUESTION,
    interviewee=DEFAULT_INTERVIEWEE
):
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

def healthworker_coder(
    transcript,
    reflection_content,
    coding_memory_content=None
):
    fixed_profile = """
    You are a senior clinical nurse in your early 50s who has spent over 20 years working in public hospitals and community health facilities. You worked directly with patients during infectious disease outbreaks and were involved in implementing infection prevention measures, adapting services, and supporting colleagues through periods of high demand.
    As a senior clinical nurse, you frequently used the healthcare system, and you can bring a frontline healthcare delivery perspective when coding a transcript.
    You value patient care, staff safety, and maintaining healthcare services during crises.
    You are especially attentive to workload pressures, staffing challenges, infection risks, disruptions to care delivery, and practical adaptations made by health workers.
    You may underestimate broader political, governance, or societal influences that shape healthcare experiences.
    """.strip()  # can be changed based on research

    return coder_agent(
        transcript=transcript,
        fixed_profile=fixed_profile,
        reflection_content=reflection_content,
        coding_memory_content=coding_memory_content
    )

def policy_maker_coder(
    transcript,
    reflection_content,
    coding_memory_content=None
):
    fixed_profile = """
    You are a senior public health policy maker in your late 50s who has spent over 20 years working within national health systems and advising on health emergency preparedness and response. You were involved in developing policies during public health emergencies and coordinating responses across government agencies and healthcare organizations.
    As a health policy maker, you can bring a systems-level governance perspective when coding a transcript.
    You value effective leadership, health system resilience, preparedness, and evidence-informed decision making.
    You are especially attentive to governance structures, policy implementation, resource allocation, institutional coordination, and lessons learned from previous outbreaks such as Ebola.
    You may underestimate the everyday realities and emotional experiences of patients and frontline health workers. 
    """.strip()  # can be changed based on research

    return coder_agent(
        transcript=transcript,
        fixed_profile=fixed_profile,
        reflection_content=reflection_content,
        coding_memory_content=coding_memory_content
    )


def patient_coder(
    transcript,
    reflection_content,
    coding_memory_content=None
):
    fixed_profile = """
    You are a community member in your late 40s who relied on healthcare services before and during the COVID-19 pandemic. You experienced changes in healthcare access, public health restrictions, and communication from health authorities, while observing how your community responded to infectious disease outbreaks.
    As a patient, you frequently used the healthcare system, and you bring a lived-experience perspective when coding a transcript.
    You value accessible healthcare, trust, clear communication, and respectful treatment.
    You are especially attentive to barriers to care, experiences of healthcare services, public perceptions, trust in institutions, fear, uncertainty, and community responses to disease outbreaks.
    You may underestimate organizational, policy, or resource constraints faced by health systems.    
    """.strip()  # can be changed based on research

    return coder_agent(
        transcript=transcript,
        fixed_profile=fixed_profile,
        reflection_content=reflection_content,
        coding_memory_content=coding_memory_content
    )

def qualitative_researcher_coder(
    transcript,
    reflection_content,
    coding_memory_content=None
):
    fixed_profile = """
    You are a qualitative health researcher in your mid-50s who have spent over 15 years conducting qualitative studies involving patients, caregivers, and healthcare professionals. You have extensive experience using reflexive thematic analysis and teaching qualitative research methods.
    As a qualitative researcher and reflexive thematic analysis methodologist, you are a researcher studying healthcare experiences, and you can bring a methodological perspective. 
    You value reflexivity and interpretive transparency.
    You are especially attentive to interpretive transparency and reflexivity.
    You may underestimate operational healthcare challenges or policy constraints that are not explicitly discussed in the transcript.
    """.strip()  # can be changed based on research

    return coder_agent(
        transcript=transcript,
        fixed_profile=fixed_profile,
        reflection_content=reflection_content,
        coding_memory_content=coding_memory_content
    )

coder_functions = {
    "healthworker": healthworker_coder,
    "policy_maker": policy_maker_coder,
    "patient": patient_coder,
    "qualitative_researcher": qualitative_researcher_coder
}

def run_independent_coding(
    transcript_file,
    agent_name,
    reflection,
    transcript_id,
    chunk_id,
    round_number,
    coding_memory_content=None
):
    transcript_chunk = read_docx(transcript_file)

    coding_func = coder_functions.get(agent_name)

    if coding_func is None:
        raise ValueError(f"Agent '{agent_name}' not found in coder_functions.")

    if isinstance(reflection, dict) and agent_name in reflection:
        current_reflection = reflection[agent_name]
    else:
        current_reflection = reflection

    messages = coding_func(
        transcript=transcript_chunk,
        reflection_content=current_reflection,
        coding_memory_content=coding_memory_content
    )

    response = get_completion(messages)
    time.sleep(2)

    res = response.choices[0].message.content.strip()

    output_dir = os.path.join(DEFAULT_OUTPUT_BASE_DIR, agent_name)
    os.makedirs(output_dir, exist_ok=True)

    file_name = f"round_{round_number}_{transcript_id}_chunk_{chunk_id}"

    json_path = os.path.join(output_dir, f"{file_name}.json")

    md_path = os.path.join(output_dir, f"{file_name}.md")

    data_to_save = {
        "agent": agent_name,
        "round": round_number,
        "transcript": transcript_id,
        "chunk": chunk_id,
        "reflection": current_reflection,
        "coding_output": res
    }

    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(data_to_save, json_file, indent=4, ensure_ascii=False)

    with open(md_path, "w", encoding="utf-8") as md_file:
        md_file.write(res)

    return res, json_path, md_path

def remove_importance_score(final_coding_base_dir):
    # final_coding_base_dir: one transcript (all chunks) coding results under a coder agent
    pattern = r'(?im)^[ \t]*Importance\s+Score\s*:\s*\[?\s*[1-5]\s*\]?\s*$\n?'

    new_base_dir = final_coding_base_dir.replace(
        "final_coding",
        "final_coding_without_score"
    )

    for agent in os.listdir(final_coding_base_dir):
        agent_dir = os.path.join(final_coding_base_dir, agent)

        if not os.path.isdir(agent_dir):
            continue

        new_agent_dir = os.path.join(new_base_dir, agent)
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
