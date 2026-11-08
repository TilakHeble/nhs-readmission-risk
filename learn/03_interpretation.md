# 03 · Interpreting coefficients: odds ratios, risks and marginal effects

🟢 Beginner · [Course index](README.md) · Previous: [02](02_maximum_likelihood.md) · Next: [04 · Inference](04_inference.md)

## The question

A model says the emergency-admission coefficient is 0.6. What should you tell a ward manager?

## Setup

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from nhsread.ingest.reader import deprivation_cells, read_compendium
from nhsread.synth.simulate import simulate_cohort

cohort = simulate_cohort(deprivation_cells(read_compendium()))
m = smf.logit("readmit_30d ~ emergency + charlson + np.log1p(los_days) + C(age_band, Treatment('16-74')) + sex",
              data=cohort).fit(disp=0)
print(m.params.round(3))
```

## Three scales, three stories

| Scale | What a coefficient means | Emergency admission example |
|---|---|---|
| log-odds | additive change in log-odds | +β |
| **odds ratio** | multiplies the odds by $e^\beta$ | the odds are multiplied by $e^\beta$ |
| probability | depends on the patient's starting risk | see below |

```python
or_table = pd.DataFrame({"coef": m.params, "odds ratio": np.exp(m.params),
                         "OR 95% low": np.exp(m.conf_int()[0]), "OR 95% high": np.exp(m.conf_int()[1])})
print(or_table.round(3))
```

**Odds ratios are not risk ratios.** With an outcome around 15%, an odds ratio of 1.8 corresponds to a smaller risk ratio. The rarer the outcome, the closer OR and RR become (the "rare disease assumption").

```python
def or_to_rr(or_, p0):
    """Risk ratio implied by an odds ratio at baseline risk p0."""
    return or_ / (1 - p0 + p0 * or_)


for p0 in [0.01, 0.10, 0.15, 0.40]:
    print(f"baseline risk {p0:.0%}: OR 1.8 means RR {or_to_rr(1.8, p0):.2f}")
```

Saying "emergency admissions have 80% higher risk" when the OR is 1.8 overstates the effect. Saying "80% higher **odds**" is correct but means little to most people.

## Average marginal effects: percentage points

The clearest communication is usually in percentage points. The **average marginal effect** (AME) of a binary variable is the average, over real patients, of the change in predicted probability when you switch it from 0 to 1, holding everything else as it is.

```python
d0, d1 = cohort.copy(), cohort.copy()
d0["emergency"], d1["emergency"] = 0, 1
ame = (m.predict(d1) - m.predict(d0)).mean()
print(f"AME of an emergency admission: {100 * ame:+.1f} percentage points")
print(m.get_margeff(at="overall").summary_frame().loc[["emergency", "charlson"]].round(4))
```

The statsmodels `get_margeff` treats every variable as continuous (a derivative). The manual 0→1 version is the right one for binary variables, and the two are close here.

## Dummy variables and reference categories

`C(age_band, Treatment('16-74'))` creates indicators for `<16` and `75+`, each compared with **16-74**. The coefficient of `75+` is the log odds ratio *for 75+ versus 16-74*, holding the other variables fixed. Change the reference and the coefficients change, but the predictions do not.

## Interactions

An interaction lets one variable's effect depend on another. In the real NHS data the deprivation gradient is steep for adults and almost flat for children:

```python
from nhsread.models import grouped as G

res = G.fit(cells := deprivation_cells(read_compendium()), "C(imd_c) * age + sex + C(year)")
print(np.exp(res.params.filter(like="imd")).round(3))
```

Read `C(imd_c)[T.5]` as the odds ratio for least vs most deprived **in the reference age group (16-74)**. The `C(imd_c)[T.5]:age[T.<16]` term says how much that odds ratio is multiplied for children. Multiplying the two gives the children's gradient.

## Confounding: a coefficient depends on what else is in the model

```python
demo = smf.logit("readmit_30d ~ C(imd)", data=cohort).fit(disp=0)
full = smf.logit("readmit_30d ~ C(imd) + charlson + np.log1p(los_days) + prior_ed_6m + np.log1p(prior_adm_12m)",
                 data=cohort).fit(disp=0)
print(pd.DataFrame({"deprivation only": np.exp(demo.params), "plus clinical": np.exp(full.params)}).filter(
    like="imd", axis=0).round(3))
```

The deprivation odds ratios change because deprivation acts partly *through* comorbidity and prior hospital use. Neither number is "wrong": they answer different questions (total vs direct association). Always say what a coefficient is adjusted for.

## Practice

**Q1.** The odds ratio for 75+ vs 16-74 is 1.5 and the 16-74 readmission rate is 14%. Roughly what is the 75+ rate?

<details><summary>Answer</summary>

Odds for 16-74 = 0.14 / 0.86 ≈ 0.163. Multiply by 1.5: 0.244. Back to probability: 0.244 / 1.244 ≈ 19.6%. (The risk ratio is 1.4, not 1.5.)
</details>

**Q2.** Why can't you read a logistic coefficient as a percentage-point effect?

<details><summary>Answer</summary>

Because the effect on probability depends on the starting probability: the same log-odds change moves p more near 0.5 than near 0 or 1. You need to average the change over the population (AME) or report it at chosen patient profiles.
</details>

**Q3.** You change the reference category for deprivation from "most deprived" to "least deprived". What happens to the coefficients, standard errors and predictions?

<details><summary>Answer</summary>

Coefficients change (they now compare with a different group) and so do their standard errors. Fitted probabilities, log-likelihood and AIC are identical: it is the same model parameterised differently.
</details>

**Q4.** In a model with an age x deprivation interaction, a colleague quotes the main-effect odds ratio for deprivation as "the effect of deprivation". What's wrong?

<details><summary>Answer</summary>

With an interaction, the main effect is the effect only for the reference age group. For other ages you must combine it with the interaction term. Quote effects by age group, or an average marginal effect.
</details>
