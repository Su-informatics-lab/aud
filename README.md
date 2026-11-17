# MedDx: alcohol use disorder phenotyping from clinical notes

Analysis code and the phenotyping skill for the manuscript *A language model identifies
alcohol use disorder that diagnosis codes miss in cirrhosis*.

## The skill

MedDx classifies a patient's clinical notes for alcohol use disorder (AUD) with a
set of instructions, the skill, run on an open-weight language model at temperature 0.

The skill is [`skill/aud-adjudication/SKILL.md`](skill/aud-adjudication/SKILL.md). It is
the text printed in Supplementary Note 3 of the paper, in Agent Skill format.

- Four rules, each sufficient for AUD.
- A six-grade answer, which the analysis collapses to AUD+, AUD− and indeterminate.
- Every classification names its rule findings, the note type or section it drew on,
  and a verbatim quotation.

[`skill/variations/`](skill/variations/) holds the instruction that accompanies the
notes and the two paraphrases of it used as skill variations (Extended Data Fig. 5).

The skill was fixed before any model output was compared with the hepatologists'
ground truth. It was validated as a phenotype for research and quality measurement
across a registry, not as a diagnosis for an individual patient.

## The code

| path | |
|---|---|
| `rebuild.sh` | the whole analysis in order, in three steps |
| `downstream/` | ground truth, agreement, accuracy, patient groups, comparators |
| `registry/` | the registry estimate, treatment and the sensitivity analyses |
| `figures/` | the figure panels |
| `provenance.md` | which script produces which figure, table and note |

The analysis starts from the hepatologists' classifications and the per-patient model
classifications. Python 3.12 with pandas, numpy, scipy, pyarrow, openpyxl, matplotlib and seaborn; the
comparators also need scikit-learn, xgboost, torch and transformers. Without the data,
one check runs:

```sh
cd downstream && python3 -c "import stats_core; stats_core._selftest()"
```

## Not here

The study data are protected health information. The language-model serving, the note
extraction and the chart-review application are not released.

## Licence

Code: MIT ([`LICENSE`](LICENSE)). The skill and the documentation: CC BY 4.0
([`LICENSE-docs`](LICENSE-docs)). `downstream/npj/nature_style.py` is a third-party
helper, kept unmodified.
