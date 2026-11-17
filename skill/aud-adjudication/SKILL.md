---
name: aud-adjudication
description: >
  Classify Alcohol Use Disorder (AUD) status from clinical chart notes
  using a six-point ordinal scale. Designed for LLM-based chart review
  that replicates expert hepatology fellow adjudication. Applicable to
  any EHR note corpus where AUD status needs structured extraction.
version: "1.0"
license: CC-BY-4.0
---

# AUD Chart Adjudication — Guidance for the LLM Reviewer

This is a self-contained instruction set for an LLM (or a new human reviewer) classifying whether a patient has Alcohol Use Disorder (AUD) from their clinical chart. It is written to be followed literally and in order.

# 1. What You Are Being Asked to Do

You will be given the clinical chart documentation for one patient at a time. Your single job is to read the chart and assign exactly one label from a six-point scale describing how confident the evidence is that this patient has Alcohol Use Disorder.

You are not diagnosing the patient. You are reporting what the documentation supports. Think of yourself as a careful reader summarizing what the chart says about alcohol — not a doctor deciding what is true.

Three things must accompany every classification: (a) which evidence categories supported your decision, (b) any screening-tool scores you found (AUDIT/CAGE), and (c) a short supporting quote with where it came from (HPI, problem list, social history, etc.). Never give a label without the quote that justifies it.

# 2. Read This Before You Start

- Your task is to replicate what a careful chart reviewer would conclude. Follow the rules precisely, not your own clinical intuition.
- The review window matters. Focus on documentation most relevant to the clinical question. Evidence far outside the period of interest is context, not primary basis.
- One patient may have many documents. Read all of them for that patient before deciding. Evidence in any one note counts for the whole patient.

# 3. The Six-Point Scale

Assign exactly one:

- Definitely Yes: An explicit AUD diagnosis, documented alcohol dependence or abuse, alcoholic cirrhosis (the only alcohol-attributed disease that qualifies on its own), a positive AUDIT (>=8) or CAGE (>=2), OR documented alcohol withdrawal and its management (including CIWA monitoring). Any ONE of these is enough.
- Probably Yes: No standalone trigger — BUT the chart documents heavy/extensive prior drinking, problematic patterns, OR alcohol-attributed sequelae OTHER than cirrhosis (alcoholic hepatitis, alcoholic pancreatitis, ALD, Wernicke/Korsakoff). These rise to Definitely Yes only when combined with a corroborating trigger.
- Uncertain: Alcohol use is mentioned, but quantity, pattern, or severity cannot be determined (e.g., "Alcohol: Yes" or "drinks daily" with no further detail).
- Probably No: Alcohol is mentioned only in the context of clearly social or occasional consumption.
- Definitely No: Documentation explicitly indicates the patient is a non-drinker, OR alcohol is explicitly ruled out as a contributor (e.g., "no history of ETOH abuse," "cirrhosis not due to alcohol").
- Not Enough Data: No mention of alcohol use anywhere in the reviewed chart.

How these collapse for analysis: Definitely Yes + Probably Yes = "AUD Present." Definitely No + Probably No = "AUD Absent." Uncertain and Not Enough Data are handled separately.

## 3.1 The Definitely Yes Triggers — Memorize These

Any ONE of the following locks the classification at Definitely Yes. Nothing later in the chart can downgrade it.

1. Alcoholic cirrhosis named as such — Cirrhosis is the ONLY alcohol-attributed disease that is a standalone trigger.
2. Explicit AUD diagnosis or "alcohol dependence / alcohol abuse / alcoholism" in the narrative — Direct evidence.
3. Positive screening tool: AUDIT >=8 or CAGE >=2 — Validated instruments crossing their clinical threshold.
4. Documented alcohol withdrawal OR its management — including CIWA protocol, benzodiazepine taper for withdrawal, DTs, detox. CIWA placement counts EVEN IF the patient never manifests withdrawal signs.

# 4. The Principles You Must Follow

## 4.1 Narrative beats structured fields (most important rule)

When structured fields (e.g., Social History box "Alcohol: No") and provider narrative (HPI, Assessment/Plan, progress notes) disagree, the narrative wins. Structured fields are often auto-populated or carried forward unchanged. A structured "Alcohol: No" next to narrative "history of alcoholic cirrhosis" = Definitely Yes.

Exception: if a structured field is the ONLY evidence and there is no narrative about alcohol at all, accept the structured field at face value.

## 4.2 Current sobriety never lowers the classification

AUD is about whether the disorder exists or existed. "Sober since 2015," "in remission," "attends AA," and "quit drinking" are evidence that there was a disorder to recover from. If Definitely Yes criteria are met historically, the answer stays Definitely Yes.

## 4.3 One trigger is enough

Do not wait for corroboration. A single Definitely Yes trigger settles the classification.

## 4.4 Definitely No includes "ruled out as a contributor"

Use the broader definition: not only "patient is a non-drinker," but also any explicit statement ruling alcohol out — e.g., "no history of ETOH abuse," "cirrhosis not due to alcohol," "non-alcoholic steatohepatitis" when stated to exclude alcohol.

## 4.5 Classify only what is documented

Never infer a diagnosis the chart does not state. "Cirrhosis" with no stated cause is NOT automatically alcoholic.

# 5. How to Handle Marginal Cases

- Structured field conflicts with narrative → follow the narrative (Section 4.1).
- Single acute intoxication or "etoh fall" visit, no diagnosis/pattern/screen/withdrawal/sequelae → Probably Yes or Uncertain, NOT Definitely Yes. One event is not a disorder.
- Alcohol-attributed sequelae OTHER than cirrhosis (alcoholic hepatitis, pancreatitis, ALD, Wernicke/Korsakoff) WITHOUT a corroborating trigger → Probably Yes. They rise to Definitely Yes only when paired with another trigger.
- "Alcohol: Yes" or "drinks daily" with no quantity/pattern/severity → Uncertain.
- Clearly social/occasional use → Probably No.
- Conflicting frequency entries in structured fields → let the clinical picture in the narrative decide.
- Patient sober/in remission but historical triggers present → Definitely Yes (Section 4.2).
- No mention of alcohol anywhere → Not Enough Data.

# 6. Common Mistakes and Corrections

- Trusting structured "Alcohol: No" over narrative → Narrative wins. Alcoholic cirrhosis in HPI = Definitely Yes regardless of structured field.
- Downgrading because patient is sober → Current abstinence is irrelevant. Sober 10 years + alcoholic cirrhosis = still Definitely Yes.
- Requiring several triggers → Any ONE trigger is sufficient.
- CIWA only counting if withdrawal occurred → CIWA initiation alone is the trigger.
- "Cirrhosis not due to alcohol" classified as Not Enough Data → This is Definitely No.
- Single intoxication = Definitely Yes → One acute event without diagnosis/pattern/withdrawal is Probably Yes or Uncertain.
- Inferring unstated diagnoses → Classify only what is documented. No alcohol mention = Not Enough Data.

# 7. Worked Examples

Example 1 — Structured-field trap: Social History "Alcohol: No" but HPI says "hx of alcohol related cirrhosis with portal HTN" → Definitely Yes. Narrative wins.

Example 2 — Clean denial: "Alcohol use: Denies." No alcohol-related anything → Definitely No.

Example 3 — CIWA without withdrawal: Patient on CIWA protocol, no formal AUD diagnosis → Definitely Yes. CIWA initiation is a standalone trigger.

Example 4 — Single intoxication: ER note for etoh fall, one event, no diagnosis/pattern → Probably Yes or Uncertain, NOT Definitely Yes.

Example 5 — Sober for years: "Alcoholic cirrhosis" + "sober since 2014, attends AA" → Definitely Yes. Sobriety never lowers.

Example 6 — Alcohol ruled out: "NASH cirrhosis. No history of ETOH abuse." → Definitely No.

Example 7 — Non-cirrhosis sequela alone: "Alcoholic pancreatitis" with no other triggers → Probably Yes (only cirrhosis is a standalone trigger).

Example 8 — Sequela + corroboration: "Alcoholic hepatitis" + "CIWA initiated" → Definitely Yes. CIWA is a standalone trigger.

# 8. Final Checklist

Before submitting each classification:
- Did I read every document for this patient?
- Did I check whether any single Definitely Yes trigger is present?
- If structured and narrative disagree, did I follow the narrative?
- Did I avoid downgrading because the patient is currently sober?
- Is my label backed by an actual quote from the chart, with its source location?
- Did I record any AUDIT/CAGE scores and the evidence categories I used?
- Am I classifying only what is documented — not what I assume?
