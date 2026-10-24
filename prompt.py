import os

from langchain_core.prompts import PromptTemplate

# ============================================================
# Chain 1: Sufficiency check (with time-query detection)
# ============================================================
sufficiency_check_prompt_str = """
You are a routing agent for a QA system.

User Query: {query}

Retrieved D-Level Summaries:
{d_balls_context}

Determine if the summaries contain ALL key facts needed to answer the query.

- If YES (answer is directly present or clearly deducible):
  Return: {{"is_sufficient": true, "target_d_idx": "D_x"}}

- If NO (specific facts, names, numbers, dates, relationships are missing):
  Also determine if the query asks about a date, time, year, or when something happened. Set "is_time_query" to true or false accordingly.
  Return: {{"is_sufficient": false, "is_time_query": true/false, "missing_entities": ["concrete entity name(s) the answer must contain"], "target_d_idx": "D_x"}}

CRITICAL: missing_entities must be concrete named entities (e.g., "August 2006", "emergency-essential program", "Fort Bliss"), NOT generic terms like "program", "date", "policy".

Return ONLY the JSON object.
"""

sufficiency_check_prompt = PromptTemplate(
    input_variables=["query", "d_balls_context"],
    template=sufficiency_check_prompt_str
)


# ============================================================
# Chain 2: Entity extraction (only called when is_sufficient=false)
# ============================================================
entity_extract_prompt_str = """
You are a search entity extractor.

User Query: {query}

Extract:
1. query_entities: Concrete named entities from the query that can match knowledge base entities.
   - Include: program/policy names, organization names, location names, operation names, person names, service branches
   - Also include specific qualified concepts explicitly written in the query when they can act as graph entity anchors (for example, "combat-essential systems" or "post-deployment health assessments").
   - If the question asks "Which X..." where X is a category (location, office, program, policy, date), X itself is the answer, NOT a search entity. Extract OTHER named entities from the question instead.
   - If no concrete named entities exist, return empty list.
   - Do NOT guess the unknown answer and do NOT add entities that are not explicitly supported by the query wording.

2. query_domain: 1 to 3 broad domain tags (e.g., "Defense Policy", "Health Surveillance", "Deployment Readiness", "Documentation Compliance", "Personnel Management").
   - CRITICAL: query_domain MUST NOT be empty. Always infer at least one domain tag from the query.
   - If unsure, choose the closest matching domain from what the question describes.

3. query_focus: A short description of the relation or attribute the question asks about.
   - Examples: "date when the revised policy was issued", "organization responsible for maintaining the system", "location with missing documentation".
   - This is NOT a search entity and must not be copied into query_entities unless the same phrase is explicitly a concrete entity in the query.

4. answer_type: The expected short-answer type, such as "date", "organization", "location", "program", "person", "number", or "short_text".

Return ONLY: {{"query_entities": [...], "query_domain": [...], "query_focus": "...", "answer_type": "..."}}
"""

entity_extract_prompt = PromptTemplate(
    input_variables=["query"],
    template=entity_extract_prompt_str
)

entity_retry_prompt = PromptTemplate(
    input_variables=["query"],
    template=(
        "The first entity extraction returned no usable graph anchors. Re-examine the query without guessing the answer.\n"
        "User Query: {query}\n\n"
        "Extract concrete named entities and specific qualified concepts that are explicitly present in the query and can match a knowledge-graph entity. "
        "Do not return generic answer placeholders such as 'specific date', 'specific location', or 'program name'.\n"
        "Also return query_domain, query_focus (the requested relation/attribute), and answer_type.\n"
        "Return JSON only: {{\"query_entities\": [...], \"query_domain\": [...], "
        "\"query_focus\": \"...\", \"answer_type\": \"...\"}}"
    )
)

def _build_answer_prompt():
    # Opt-in answer-format experiment. The default keeps the v5 prompt unchanged.
    if os.environ.get("GBRAG_ANSWER_PROMPT_V6", "0") == "1":
        return PromptTemplate(
            input_variables=["query", "context"],
            template=(
                "Answer the user query using only the context below.\n"
                "Return ONLY the final answer text, with no label, explanation, markdown, or citations.\n"
                "Use the shortest exact answer supported by the context. Preserve the source wording for names, agencies, dates, and numbers.\n"
                "The context has two factual layers: direct facts and [Multi-hop Bridge Clue] propositions.\n"
                "Use bridge propositions to connect relevant facts, but prefer a directly stated fact when one answers the question.\n"
                "Never combine unrelated candidates and never replace a specific entity with a broader organization.\n"
                "For a number/count question, return the requested number with its unit if needed.\n"
                "For a comparison question, identify both sides, compare the quantities or properties asked about, and return only the requested side/entity.\n"
                "For an agency/office/person question, return the exact named organization/person, not a surrounding list or parent department.\n"
                "For a date/year question, select the date tied to the exact event in the question.\n"
                "If no unambiguous answer is explicitly supported, return unknown.\n\n"
                "User Query: {query}\n\n"
                "Context:\n{context}\n\n"
                "Final Answer:"
            ),
        )
    return PromptTemplate(
        input_variables=["query", "context"],
        template=(
            "Answer the user query using only the context below.\n"
            "Return ONLY the final short answer text itself.\n"
            "The answer should be as short as possible, preferably a single entity name, date, number, or short noun phrase.\n"
            "Do NOT output labels such as 'Answer:', 'The answer is:', or 'Short Answer:'.\n"
            "Do NOT include explanations, citations, markdown, bullet points, or full sentences.\n"
            "If the context truly does not contain the needed fact, return unknown.\n\n"
            "=== CONTEXT FORMAT NOTE ===\n"
            "The context may contain two section types, both are valid factual information:\n"
            "1. [C_x extracted facts]: Facts extracted from a cluster.\n"
            "2. [Multi-hop Bridge Clue]: ALSO real propositions that bridge clusters. "
            "They contain valid factual statements and MUST be treated as equally important context, "
            "NOT as path descriptions or metadata.\n"
            "Pay special attention to [Multi-hop Bridge Clue] content — it is often the critical fact connecting different parts of the answer.\n"
            "IMPORTANT: The [Multi-hop Bridge Clue] sections contain relation propositions that directly "
            "answer the query's relationship. Pay special attention to dates, times, and entities mentioned "
            "in these bridge sections — they are more likely to directly address the question's core relationship. "
            "The [C_x extracted facts] sections provide supporting context, but the bridge edges carry the key relational information.\n\n"
            "=== TIME QUESTIONS (when/which date/which year) ===\n"
            "Multiple dates may appear in the context. Identify the specific action or event in the question, and match it to the corresponding date in the context. Do not confuse related but different dates (e.g., announcement vs. effective date).\n\n"
            "=== COMPARISON QUESTIONS (which had more/fewer, which is larger/smaller, compared with) ===\n"
            "You MUST think step by step:\n"
            "Step 1: Identify ALL numeric values in the context and which entity each belongs to.\n"
            "Step 2: Pay attention to the question's logic. For example:\n"
            "  - 'fewer X lacking documentation' = 'X has less deficiency' = 'X is better/more compliant'\n"
            "  - 'more records missing' = the entity with the higher number\n"
            "Step 3: Compare the numbers strictly and choose the correct entity.\n"
            "Step 4: Output ONLY the final answer entity name (no explanations).\n\n"
            "User Query: {query}\n\n"
            "Context:\n{context}\n\n"
            "Final Answer:"
        )
    )

def _build_d_summary_answer_prompt():
    return PromptTemplate(
        input_variables=["query", "context"],
        template=(
            "Answer the user query using ONLY the D-level summary context below.\n"
            "Return ONLY the final short answer text itself.\n"
            "The answer should be as short as possible, preferably a single entity name, date, number, or short noun phrase.\n"
            "Do NOT output labels such as 'Answer:', 'The answer is:', or 'Short Answer:'.\n"
            "Do NOT output JSON, markdown, explanations, bullet points, or full sentences.\n"
            "CRITICAL: Use the EXACT wording from the context whenever possible. "
            "If the context says 'Air Force locations', answer 'Air Force locations', NOT 'Air Force sites'.\n"
            "If the answer is not explicitly supported by the context, return unknown. "
            "Do NOT guess or infer based on partial information.\n\n"
            "=== COMPARISON QUESTIONS (which had more/fewer, which is larger/smaller, compared with) ===\n"
            "You MUST think step by step:\n"
            "Step 1: Identify ALL numeric values in the context and which entity each belongs to.\n"
            "Step 2: Pay attention to the question's logic.\n"
            "  - 'fewer X lacking documentation' = 'X has less deficiency' = 'X is better/more compliant'\n"
            "Step 3: Compare the numbers strictly and choose the correct entity.\n"
            "Step 4: Output ONLY the final answer entity name.\n\n"
            "User Query: {query}\n\n"
            "D-Level Context:\n{context}\n\n"
            "Final Answer:"
        )
    )

def _build_d_summary_verified_answer_prompt():
    """D-summary fast path: require an answer plus an exact supporting fact."""
    return PromptTemplate(
        input_variables=["query", "context"],
        template=(
            "Answer the user query using ONLY the target D-level context below.\n"
            "Return JSON only with exactly these fields:\n"
            "{{\"answer\": \"short answer or unknown\", "
            "\"supporting_fact\": \"one exact sentence copied from the context or empty\", "
            "\"confidence\": 0.0}}\n"
            "The supporting_fact must be copied verbatim from the context and must directly support the answer.\n"
            "If the context does not explicitly support one unambiguous answer, return answer=unknown, "
            "supporting_fact=\"\", and confidence below 0.65.\n"
            "Do not guess. Keep the answer as a short entity, date, number, or noun phrase.\n\n"
            "User Query: {query}\n\n"
            "Target D-Level Context:\n{context}\n"
        )
    )

def _build_nli_judge_prompt():
    return PromptTemplate(
        input_variables=["question", "gold_answer", "predicted_answer"],
        template=(
            "You are a semantics-aware QA evaluator.\n"
            "Question: {question}\n"
            "Reference answer: {gold_answer}\n"
            "Predicted answer: {predicted_answer}\n\n"
            "Judge whether the predicted answer correctly answers the question with the same fact as the reference answer.\n"
            "If the predicted answer is longer but still contains the correct answer and does not contradict it, label it as entailment.\n"
            "Return JSON only:\n"
            "{{\"label\": \"entailment|neutral|contradiction\"}}"
        )
    )


entity_prompt = PromptTemplate(
    input_variables=[
        "Proposition",
        "SupportContext",
    ],
    template="""
    Task:
    Extract proposition-level core entities from the current proposition. Use the support context only as a fallback for resolving pronouns, aliases, or ambiguous references. The main goal is core entity extraction from the current proposition, not context-based replacement.

    Input:
    - Current proposition: {Proposition}
    - Optional local support context: {SupportContext}

    Rules:
    1. Focus primarily on the current proposition.
    2. First identify all core entities explicitly expressed in the current proposition.
    3. If the current proposition contains no ambiguous reference, extract core entities only from the current proposition.
    4. If the current proposition contains both explicit core entities and ambiguous references, keep the explicit core entities and also resolve the ambiguous references using SupportContext when possible.
    5. If the current proposition contains no explicit core entity but only pronouns, aliases, or ambiguous references, you must use SupportContext to resolve them into core entity names when reliable.
    6. SupportContext is only for pronoun, alias, or ambiguous-reference resolution. Do not use it to introduce unrelated entities or to replace a more specific entity already expressed in the proposition with a broader context entity.
    7. If the support context is insufficient to uniquely resolve an ambiguous reference, do not guess and do not force a completion.
    8. Keep only proposition-level core entities. Valid entity types include:
       - Person
       - Organization / department / agency
       - Named operation / campaign
       - Named program
       - Named policy / regulation / directive
       - Named system / database
       - Geopolitical location only if explicitly stated and central
       - Personnel group only if explicitly stated and central
    9. Strictly exclude:
       - Time expressions (years, dates, periods, relative time, time ranges)
       - Location expressions unless the location name itself is a central named entity
       - Quantities, statistics, ordinals, durations, measurements
       - Abstract concepts or generic descriptors
    10. Do not output pronouns, generic phrases, or ambiguous references themselves. If they can be resolved, output the resolved formal entity names instead.
    11. Do not replace a specific proposition-level entity with a higher-level background entity from context.
        Example: if the proposition explicitly centers on "federal civilian employees" or "force health protection and surveillance policies", do not output only "DOD".
    12. Deduplicate the results. entity_list must contain 0 to 3 items only.
    13. Do not treat generic common nouns such as "components", "documentation", "records", "procedures", "requirements", or "policies" as valid entities unless they are explicitly named or can be uniquely resolved to a formal entity.

    Output format:
    Output ONLY one line:
    entity_list=["Entity 1", "Entity 2"]
    If there is no valid entity, output:
    entity_list=[]
"""
    )

time_prompt = PromptTemplate(
        input_variables=[
            "Proposition",
            "SupportContext",
        ],
        template="""
    Task:
    Extract absolute and precise time expressions from the current proposition. If the proposition contains vague or relative time expressions, use the support context only to resolve them into absolute time expressions.

    Input:
    - Current proposition: {Proposition}
    - Optional local support context: {SupportContext}

    Rules:
    1. Focus only on the current proposition.
    2. Extract explicit absolute time expressions when they are clearly stated, including:
       - Year
       - Month + year
       - Full date
       - Explicit time span or date range
    3. If the current proposition contains a vague or relative time expression such as "recently", "in recent years", "last year", or "at that time", you must use SupportContext to resolve it into an absolute time expression when possible.
    4. If a vague or relative time expression cannot be resolved reliably into an absolute time expression, do not output it.
    5. Do not guess. Do not use world knowledge beyond the current proposition and SupportContext.
    6. Do not copy time expressions from SupportContext unless they clearly resolve a time expression already present in the current proposition.
    7. If both a coarse-grained and a more precise time expression refer to the same time, keep only the more precise one.
       Example: if "August 2006" is present, do not also output "2006".
    8. If the proposition does not express any time information, output an empty list.
    9. Deduplicate the results.
    10. time_entity_list must contain 0 to 3 items only.

    Examples:
    Input: "In August 2006, DOD issued a revised policy."
    Output:
    time_entity_list=["August 2006"]

    Input: "DOD established the emergency-essential program in 1985."
    Output:
    time_entity_list=["1985"]

    Input: "In recent years, the department expanded the program."
    If SupportContext clearly resolves it to 2018-2020, output:
    time_entity_list=["2018-2020"]
    Otherwise, output:
    time_entity_list=[]

    Output format:
    Output ONLY one line:
    time_entity_list=["Time 1", "Time 2"]
    If there is no valid time expression, output:
    time_entity_list=[]
    """
    )

tag_prompt = PromptTemplate(
        input_variables=[
            "Proposition",
        ],
        template="""
    Task:
    Assign 1 to 2 reusable high-level domain labels to the current proposition.

    Input:
    - Current proposition: {Proposition}

    Core requirements:
    - extend_key_list must not be empty.
    - You must return 1 or 2 labels.
    - Returning [] is forbidden.

    Label design rules:
    1. Labels must describe the main domain or subdomain of the current proposition.
    2. Labels must be reusable across many similar propositions in the same dataset.
    3. Labels must not be entity names, person names, organization names, operation names, program names, location names, or direct copies of the proposition.
    4. Labels must not be overly broad, empty, vague, or event-specific.

    Avoid labels like:
    - "Policy"
    - "Health"
    - "Military"
    - "Medicine"
    - "System"
    - "Operations"

    Prefer labels like:
    - "Defense Policy"
    - "Military Health"
    - "Health Surveillance"
    - "Deployment Readiness"
    - "Clinical Trials"
    - "Cancer Treatment"
    - "Diagnostic Testing"
    - "Infectious Disease"

    Specificity rules:
    1. Prefer the nearest meaningful reusable domain label, not the broadest parent label.
    2. If the proposition supports both a broad domain and a more specific subdomain, return both only when both are clearly useful and non-redundant.
    3. If only one strong label is supported, return only one label.
    4. If uncertain, choose the closest non-generic reusable domain phrase rather than a broad or vague label.
    5. Do not output two labels that are near-duplicates or differ only slightly in wording.

    Formatting rules:
    - Use short noun phrases.
    - Prefer 2 to 4 words.
    - Labels must be in English Title Case.
    - Do not use full sentences.
    - Do not include punctuation unless necessary.

    Example:
    Proposition: "DOD issued a revised force health protection policy."
    Output:
    extend_key_list=["Defense Policy", "Military Health"]

    Output format:
    Output ONLY one line:
    extend_key_list=["Label 1", "Label 2"]
"""
    )


def _build_forceful_answer_prompt():
    """Stronger prompt used when the standard prompt returns unknown."""
    return PromptTemplate(
        input_variables=["query", "context"],
        template=(
            "You MUST answer the user query using ONLY the provided context.\n"
            "Even if the answer is not perfectly explicit, extract the MOST LIKELY correct answer from the context.\n"
            "Do NOT output 'unknown' unless the context contains absolutely no relevant information.\n"
            "Return ONLY the final short answer text itself.\n"
            "The answer should be as short as possible: a single entity name, date, number, or short noun phrase.\n"
            "Do NOT output labels, explanations, citations, markdown, or bullet points.\n\n"
            "User Query: {query}\n\n"
            "Context:\n{context}\n\n"
            "Final Answer:"
        )
    )
