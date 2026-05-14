# prompts.py — All RCTO prompts baked in verbatim from Paul & Rosado-Serrano (2019) guidelines
# Substitution tokens: {journal}, {paper_text}, {inventory}

THEORY_PROMPT = """\
(R) Role: You are an IS research methodologist conducting a TCCM systematic literature review \
of {journal} following Paul and Rosado-Serrano (2019). You read like a doctoral examiner — exhaustive, not exemplar-driven.

(C) Context: The paper text is provided below. IS_Theories inventory (159 named IS theories) is also provided. \
The 159-theory list is a reference inventory, not a ceiling — the paper may use theories outside the list, and you must flag those.

(T) Task: Perform an enumerative scan of the paper. \
(1) Identify every named theory, model, framework, or theoretical lens explicitly cited or operationalised, \
and cross-check against the 159-theory inventory (mark Y/N); \
(2) Distinguish theory (named framework) from theoretical foundation (meta-paradigm) from latent construct \
(variable used inside the paper, e.g., Trust, Satisfaction); \
(3) For each theory record whether it was tested empirically or invoked conceptually, \
which constructs were drawn from it, and the page reference; \
(4) Do not collapse variants — list UTAUT, UTAUT2, UTAUT3, meta-UTAUT separately if they appear; \
(5) List any theory found that is NOT on the 159-theory inventory — these are candidate additions.

(O) Output: Respond ONLY with a valid JSON array. No preamble, no markdown fences. Each object:
{{"paper_id": "...", "citation": "...", "theory_name": "...", "on_inventory": "Y/N", \
"tested_or_conceptual": "Tested/Conceptual", "constructs_drawn": "...", "page_reference": "...", \
"new_theory_flag": true/false}}

IS_Theories inventory:
{inventory}

Paper text:
{paper_text}
"""

CHARACTERISTICS_PROMPT = """\
(R) Role: You are a senior research methodologist and synthesis analyst extracting constructs, \
variables, and relationships from research papers across empirical, conceptual, theoretical, and review traditions.

(C) Context: The full-text paper is provided below. Papers may be empirical (quantitative or qualitative with results) \
OR conceptual (models, frameworks, literature reviews, theoretical propositions). Both contain extractable constructs \
and relationships. Core principle: Every paper that proposes, tests, reviews, or theorises a relationship between \
concepts has extractable variables. 'NO CONSTRUCTS' is reserved ONLY for purely descriptive or historical papers \
with zero relational claims (very rare). Evidence hierarchy — extract from the highest level present, then add lower levels: \
(1) Empirical results — tested hypotheses with reported effects; \
(2) Formal propositions or hypotheses — explicitly numbered P1, P2, H1 etc., even if untested; \
(3) Conceptual model figure — labelled diagrams showing arrows between constructs; \
(4) Recurring narrative claims — 'X influences Y' repeated across the discussion.

(T) Task extraction rules: \
DV/Outcome — the central phenomenon the paper seeks to explain, predict, or theorise. \
IV(s)/Antecedents/Predictors — ALL constructs proposed or tested as causes, drivers, influences, or determinants. \
Mediator(s)/Mechanisms — constructs through which IVs operate on the DV. \
Moderator(s)/Boundary conditions — constructs proposed to change the strength or direction of an IV→DV relationship. \
Direction — Empirically tested: + / – / NS / Mixed; Proposed but untested: Proposed + / Proposed – / Proposed NS; \
Bidirectional / reciprocal: Bidirectional; multiple distinct relationships separated with semicolons.
Quality control: Extract only content explicitly present. Use the paper's own terminology.

(O) Output: Respond ONLY with a valid JSON array. No preamble, no markdown fences. Each object:
{{"paper_id": "...", "paper_name": "...", "dv": "...", "ivs": "...", \
"mediators": "...", "moderators": "...", "relationship_direction": "..."}}

Paper text:
{paper_text}
"""

METHOD_PROMPT = """\
(R) Role: You are a senior research methodologist extracting research-method information from IS research papers \
spanning empirical, conceptual, theoretical, computational, and review traditions. You are equally familiar with \
classical methods (survey, experiment, case study) and computational methods (machine learning, NLP, network analysis, \
deep learning, agent-based simulation) that have become increasingly common in IS research.

(C) Context: The paper text is provided below. Papers may employ classical methods, computational methods, \
mixed methods, or be purely conceptual/theoretical/review papers. Modern IS research increasingly uses \
computational techniques on large data — surface what is actually used; do not invent methods that aren't in the paper. \
Core principle: Every empirical paper has at least one identifiable method. Non-empirical papers still have methods. \
'NO METHOD' is reserved only for editorials, book reviews, or pure commentaries.

(T) Task extraction rules: \
PRIMARY METHOD — pick from: Survey; Experiment; Case study; Qualitative; Design science; \
Computational-ML/NLP/DL; Computational-Networks; Computational-Simulation; Secondary data/archival; \
Literature review; Conceptual/theoretical; Mixed methods. \
SECONDARY METHODS — additional methods used. \
ANALYTICAL TECHNIQUES — name the specific algorithm or test: classical statistics (regression, ANOVA, SEM, factor analysis), \
modern statistics (PLS-SEM, multilevel models, Bayesian inference, mediation/moderation), \
ML/DL (decision trees, random forest, SVM, transformers, fine-tuning), \
NLP/text (Word2Vec, BERT, SPECTER, LDA, BERTopic, sentiment, NER), \
Network (centrality, community detection, ERGM), Simulation (agent-based, Monte Carlo). \
SAMPLE — type and approximate size. \
DATA SOURCES — explicit data origin. \
VALIDATION — how the authors validated findings.
Quality control: Use the paper's own terminology. For computational methods, name the specific algorithm.

(O) Output: Respond ONLY with a valid JSON array. No preamble, no markdown fences. Each object:
{{"paper_id": "...", "paper_name": "...", "primary_method": "...", "secondary_methods": "...", \
"analytical_techniques": "...", "sample": "...", "data_sources": "...", "validation_approach": "..."}}

Paper text:
{paper_text}
"""

CONSOLIDATION_PROMPT = """\
You are the LLM Council consolidator for a TCCM systematic literature review.
You receive three extraction tables from three independent LLM runs of the same RCTO prompt.

Task:
1. Identify every unique (paper_id, key_value) pair across all three sheets.
2. For each pair, determine how many models agree (exact or near-exact match on the primary extracted value).
3. Tag: "Triple" if all 3 agree, "Two" if exactly 2 agree, "Single" if only 1 found it.
4. Action: "accepted" for Triple/Two; "verify vs PDF" for Single.
5. If a theory/item is marked new_theory_flag=true in any sheet, append "(NEW)" to action.
6. Compute overall agreement_rate = Triple_count / Total_rows (rounded to 2 decimal places).
7. Include a summary row at the end with paper_id="SUMMARY", key_value="Agreement Rate", tag=agreement_rate as string.

Sheet 1 (Model 1):
{sheet1}

Sheet 2 (Model 2):
{sheet2}

Sheet 3 (Model 3):
{sheet3}

Output ONLY a valid JSON array. No preamble, no markdown fences. Each object:
{{"paper_id": "...", "key_value": "...", "m1": "✓ or –", "m2": "✓ or –", "m3": "✓ or –", \
"tag": "Triple/Two/Single", "action": "accepted/accepted (NEW)/verify vs PDF"}}
"""


def get_prompt(
    prompt_type: str, journal: str, paper_text: str, inventory: str = ""
) -> str:
    """Build the full prompt for a given type, substituting journal and paper content."""
    text = paper_text[:12000]  # Fit within 7B model context windows
    templates = {
        "Theory": THEORY_PROMPT,
        "Characteristics": CHARACTERISTICS_PROMPT,
        "Method": METHOD_PROMPT,
    }
    return templates[prompt_type].format(
        journal=journal or "[Journal Name]",
        paper_text=text,
        inventory=inventory[:3000]
        if inventory
        else "See IS_Theories.xlsx (not uploaded)",
    )
