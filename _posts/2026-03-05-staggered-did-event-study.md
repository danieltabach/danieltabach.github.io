---
layout: single
title: "When Treatment Timing Is Messy: Event-Study Design for Staggered Rollouts"
date: 2026-03-05
categories: [applied]
tags: [python, causal-inference, difference-in-differences, event-study, econometrics]
author_profile: true
header:
  teaser: /assets/images/posts/staggered-did/event-study-plot.png
toc: true
toc_label: "Contents"
toc_sticky: true
---

*You designed an experiment. Leadership signed off. The rollout took on a life of its own. Now what?*

---

## Introduction

You work at a company with hundreds of locations. Your data science team designs an experiment: roll out a new product to a subset of locations, keep the rest as a control group, and measure the impact on efficiency metrics. Clean design. Leadership approves it. Everyone is aligned.

In practice though, large-scale rollouts almost always end up staggered.

The rollout was supposed to start in March across all treatment locations simultaneously. Instead, some locations start in March, some in June, and maybe a few don't start until September. Each location has multiple trainees who need to get ramped on the new system, and the training takes about six months per person. Some managers set up all their staff through training at the same time. Others stagger it one person at a time. So, by the end of the year, you're staring at a dataset where every treatment location adopted on its own timeline, and there is no single "before" and "after" to compare. So how do we adequately capture treatment effects? Is there a way to make conclusions when perfect compliance to a test isnt feasible?

This is the staggered adoption problem — one of the most common compliance challenges in field experiments. The good news: there's a well-established framework for getting a clean read even when adoption timing is messy.

This post walks through the problem, why simpler approaches fall short, and a technique called "event-study design" that handles it. We'll focus on the *framework* itself: how to think about it, how to apply it, and how to read the results. Full code is available in the [Appendix](#appendix-full-code) for a reproducible walkthrough!

---

## What Does Staggered Adoption Actually Look Like?

 Imagine a location with 3 trainees who each need 6 months of training on the new software. Each orange bar below is one person's training timeline:

There is no single month where treatment switches on. Every location defines its own adoption timeline.
![Staggered Adoption Skeleton](/assets/images/posts/staggered-did/skeleton_staggered.png)
*Three locations, each with trainees on different timelines. Some managers start everyone together, others spread it out.*
---

## Motivation behind the framework

One could make an argument "why not pick a single cutoff date?". Maybe you use the median adoption month, or the month when "most" locations were done. Then you compare everything before that date to everything after.

The problem is subtle, at any given calendar month, your treatment group is a *mix* of locations at completely different stages. Some have been fully operational on the new system for months. Some just finished training. Some haven't started. When you average across all of them, you're averaging together strong effects, weak effects, and zero effects. The result is a diluted number that underestimates the real impact.

This is **composition bias**. The mix of locations changes at every time point. At Month 8, your average includes some fully-adopted locations, some mid-training, and some that haven't started. At Month 14, it's a completely different mix. You're not comparing the same thing to itself across time. The composition of your treatment group is shifting underneath you, and that shifting can even flip the directionality of your estimate.

![Naive DiD Problem](/assets/images/posts/staggered-did/naive-did-problem.png)
*Left: A naive single-cutoff DiD looks clean but averages across locations at different adoption stages. Right: The actual location-level data is messy. A single cutoff blurs all of this together.*

This type of approach in the econometrics literature is called "two-way fixed effects" (TWFE): essentially, a single regression that controls for each location and each time period, then estimates one overall treatment effect. It's the default DiD model, and it works well when everyone gets treated at the same time. But when adoption is staggered (like in our case), the model quietly starts using already-treated locations as comparisons for non-treated ones, which can pull the estimate in the wrong direction. We looked at this concept briefly when we picked an average midpoint for the dates. [Callaway and Sant'Anna (2021)](https://doi.org/10.1016/j.jeconom.2020.12.001) and [Sun and Abraham (2021)](https://doi.org/10.1016/j.jeconom.2020.09.006) formalized this problem and showed just how misleading those estimates can be when treatment effects evolve over time.


---

## The Framework: Normalizing to Event Time

In practice, it's not always easy to land on one calendar date, so we are comfortable with each location defining its own timeline.

### Defining the Adoption Window

For each treatment location, we identify the **adoption window**: the period from when the *first* trainee completes training to when the *last* trainee completes training.

![Pre-Period, Period 0, Post-Period Zones](/assets/images/posts/staggered-did/pre-zero-post-examples.png)
*The same locations from above, now with the framework overlaid. Red = Pre-Period (baseline), gray = Period 0 (adoption window, blacked out), blue = Post-Period (where we measure the effect).*

This window is the messy middle. Some trainees have the new skills, others don't. The location is in flux. Think of it like renovating a kitchen. You wouldn't judge the renovation by looking at your cooking output during the week the contractors are ripping out cabinets. You'd compare your cooking *before* the renovation to your cooking *after* it's done.

### The Three Zones

We split each location's timeline into three zones:

**Pre-Period (before the window):** The location before any trainee completed training. This is the baseline. We measure metrics here and call these periods -1, -2, -3, and so on, counting backwards from the adoption window.

**Period 0 (the adoption window itself):** We black this out. Don't measure it. Don't interpret it. It's the kitchen-under-construction phase. Metrics here are a blend of old and new with no clean interpretation.

**Post-Period (after the window):** The location after *all* trainees finished training. This is where we look for the treatment effect. We call these periods +1, +2, +3, counting forward.

### The Shift

Here's where it comes together. Every treatment location, regardless of when it actually adopted, is now on the same relative timeline. Period -3 means "three months before the adoption window started" for every location. Period +2 means "two months after the last trainee finished" for every location.

![Event Time Normalization](/assets/images/posts/staggered-did/event-time-normalization.png)
*Left: The same locations in calendar time — their adoption points are scattered across different months. Right: After normalizing to event time, they all align at Period 0. Different calendar dates, same relative timeline.*
---

## Getting a Clean Sample

One of the tradeoffs for when we want to get a better read on real treatment effects is reducing our sample. For this example - we apply three filters, and each one has the same underlying logic: without it, the read is less trustworthy and more likely to be scrutinized for noise.

### Filter 1: Bounded Adoption Window

If a location's adoption window stretches beyond 6 months, we set it aside for this analysis. The reasoning is straightforward: if it took 14 months from first completion to last, the blackout window would eat almost the entire timeline. There wouldn't be enough clean pre or post data left to measure anything meaningful.

In practice, most locations naturally fell within this window. Training itself takes 6 months, and most managers either trained everyone together (adoption window = 0 months) or staggered within a few months. The locations with extremely long adoption windows were the exception.

### Filter 2: Minimum Pre-Period Data

We require at least 6 months of pre-period data for every location. Why? Because we need to validate that treatment and control locations were trending similarly *before* adoption. If a location only has 2 months of pre-data, we can't tell whether its pre-trend was flat, rising, or falling. That makes the parallel trends test unreliable for that location.

There's a more technical reason too. If some locations contribute data at Period -6 and others don't, the *set of locations* in your average changes across time periods. That's composition bias again, the same problem that broke the single-cutoff approach. By requiring every location to have data across the full analysis window, we keep the composition constant. Period -6 and Period +6 contain the exact same set of locations. The comparison is apples to apples.

### Filter 3: Minimum Post-Period Data

Same logic in the other direction. We need enough post-adoption months to actually measure whether the treatment effect materializes, ramps up, or fades. A location with only 1 month of post-data contributes almost nothing to the analysis and introduces noise.

### The Tradeoff

Every filter costs sample size — we're trading breadth for a cleaner read. It's worth being upfront about this: the treatment effect we estimate applies to "locations that adopted within a reasonable timeframe and had enough data on both sides." That's a valid and useful population, but it's not the same as "all locations." In the causal inference literature, this is similar to a **Local Average Treatment Effect (LATE)**: the effect for the compliers, not the full population.

---

## The Comparison: Treatment vs. Control

We've focused on treatment locations, but we haven't talked about the control group yet.

Control locations never adopted the software. They don't have a natural "event date." So we assign them a **pseudo-event** based on the median adoption timing of the treatment group. This sounds arbitrary, and in a sense it is. But it works because the control group's job is to capture the background time trend. What would have happened to the metric if no one adopted anything? The control group answers that.

The **parallel trends assumption** is the foundation. Before any treatment happened, were treatment and control locations moving in the same direction at the same rate? If yes, we can attribute the post-adoption divergence to the treatment. If not, the results become much harder to interpret.

![Parallel Trends](/assets/images/posts/staggered-did/parallel-trends.png)
*Average metric for treatment and control groups over calendar time. The two groups track closely in the pre-treatment period, then diverge after adoption begins.*

We validated this before running the analysis. The two groups moved together before treatment started. That's not *proof* that the assumption holds in the post-period (it's fundamentally untestable there), but it's the best evidence we can get.

*A note on this chart: the Y-axis is zoomed in to make the parallel pre-trends visible. The actual metric values are close together — the divergence looks more dramatic than it is because of the scale. The treatment effect is meaningful but modest in absolute terms, which is actually typical for real-world interventions.*

---

## The Event-Study Regression

At this point, we have all the ingredients. Treatment locations normalized to event time, control locations with pseudo-event times, and a metric to compare. Now we formalize the estimation.

### The Intuition: What DiD Is Actually Computing

At its core, Difference-in-Differences is just this:

| Group | Pre-Period Avg | Post-Period Avg | Change |
|-------|---------------|-----------------|--------|
| Treatment | 53.7 | 59.8 | +6.1 |
| Control | 53.4 | 55.5 | +2.1 |
| **DiD Estimate** | | | **+4.0** |

The treatment group's metric went up by 6.1, but the control group also went up by 2.1 (background trend — maybe seasonality, maybe the whole market improving). The difference-in-differences strips out that background trend: 6.1 - 2.1 = **4.0 units attributable to the treatment**.

The event-study regression is doing exactly this, but **period by period** instead of collapsing everything into one number. That way you can see whether the effect builds over time, appears immediately, or fades — which is much more informative than a single average.

### The Formal Model

The regression model:

$$Y_{it} = \alpha_i + \gamma_t + \sum_{k \neq -1} \beta_k \cdot D_{it}^{k} + \epsilon_{it}$$

Here is what each piece means in plain language.

**$Y_{it}$** is the efficiency metric for location $i$ in month $t$.

**$\alpha_i$** is a **location fixed effect**. Each location is compared to itself. Location 101 might naturally run at a higher efficiency than Location 205. The fixed effect absorbs that difference. We're not comparing Location 101 to Location 205. We're comparing Location 101 *before adoption* to Location 101 *after adoption*.

**$\gamma_t$** is a **time fixed effect**. This absorbs anything that affected *all* locations in a given month. Maybe December is always slower because of holidays. Maybe there was a system outage in Month 14 that hit everyone. Time fixed effects soak that up.

**$D_{it}^{k}$** is a dummy variable equal to 1 if location $i$ is a treatment location and is in relative period $k$. This is how we isolate the treatment effect at each specific distance from adoption.

**$\beta_k$** is what we care about. These are the **treatment effect coefficients**. $\beta_{+3}$ tells us: "three months after the adoption window closed, how much higher (or lower) was the metric for treatment locations compared to what we'd expect?"

We omit Period -1 as the reference category. All other $\beta_k$ values are measured relative to the period right before adoption started completing. This gives us a built-in sanity check: the pre-period betas should be near zero.

---

## Reading the Event-Study Plot

This is the payoff. Everything we've built leads to this chart:

![Event Study Plot](/assets/images/posts/staggered-did/event-study-plot.png)
*Each point is the estimated treatment effect at that relative period. The shaded band is the 95% confidence interval. Pre-period estimates hover near zero. Post-period estimates show a clear positive lift.*

Here is how to read it, piece by piece.

**The left side (Periods -6 to -1):** These are the pre-treatment coefficients. They hover near zero and their confidence intervals cross zero. This means treatment and control locations were trending the same way *before* adoption happened. The parallel trends assumption holds. If any of these were significantly different from zero, it would mean the two groups were already diverging before the software rollout, and the entire analysis would be suspect.

**The green band (Period 0):** The adoption window. We estimate a coefficient here, but don't read too much into it. Locations are mid-transition. It's the kitchen-under-construction period.

**The right side (Periods +1 to +6):** The treatment effects. The effect starts at about +2.8 in Period +1 and ramps up to about +4.5 by Period +6. This ramp-up pattern makes sense. It takes time for people to get comfortable with a new system. You wouldn't expect efficiency to instantly jump the day after the last person finishes training.

**The confidence intervals** are tighter in the middle periods and wider at the extremes. Fewer locations have data at Periods -6 and +6 (despite our balanced panel requirement, some locations are right at the edge). More data = tighter intervals.

---

## Results: Measuring the Lift

The event-study plot validates the framework, but what's the actual answer? How much did the treatment improve the metric?

### Treatment vs. Control

Here's the most intuitive view — the average metric for treatment and control groups, shown both in calendar time (left) and event time (right):

![Treatment vs Control Lift](/assets/images/posts/staggered-did/treatment-vs-control-lift.png)
*Left: In calendar time, the treatment group gradually pulls ahead as more locations complete adoption. Right: In event time, the gap becomes crisp — you can see the treatment effect building cleanly after Period 0.*

In calendar time, the divergence is gradual because locations adopt at different times (the staggering blurs the signal). In event time, the gap snaps into focus because every location is aligned to the same reference point.

### The Numbers

From our simulation, the DiD summary looks like this:

| Group | Pre-Period Avg | Post-Period Avg | Change |
|-------|---------------|-----------------|--------|
| Treatment | 53.67 | 59.75 | +6.08 |
| Control | 53.44 | 55.54 | +2.10 |
| **DiD Estimate** | | | **+3.98** |

The event-study regression gives us something richer: the effect at each relative period. Averaging the post-period coefficients (Periods +1 through +6) gives us the overall treatment effect estimate, which we can compare against simpler approaches in the next section.

---

## What Happens When You Get It Wrong

This is why the framework matters. We ran the same simulated data through three different estimation approaches to see how much the choice of method affects the result. Since this is synthetic data, we know the true treatment effect built into the simulation (4.5 units at full ramp-up), so we can measure how far off each approach lands.

![Method Comparison](/assets/images/posts/staggered-did/method-comparison.png)
*Three approaches to the same data. The naive calendar cutoff and standard TWFE both underestimate by about 27% because composition bias dilutes the signal. The event-study design gets closest to the truth, landing within 12%.*

### Approach A: Naive Calendar Cutoff

Pick the median adoption month, split everything into "before" and "after," compare treatment to control. This is the simplest possible approach and what most people try first. It systematically underestimates because at any given calendar month, your treatment group includes locations at completely different stages — some fully adopted, some mid-training, some not started. You're averaging together real effects with zeros.

### Approach B: Standard Two-Way Fixed Effects (TWFE)

Run a single regression with location fixed effects, time fixed effects, and one treatment indicator. This is the textbook DiD model, and it works perfectly when all locations adopt at the same time. Under staggered adoption, it uses the same single calendar cutoff for "post" — the fixed effects absorb level differences between locations and time periods, but the treatment estimate still suffers from the same composition bias. In our simulation, it lands at the same estimate as the naive approach.

### Approach C: Event-Study Design

The approach we've been building throughout this post. By normalizing each location to its own event time and estimating period-by-period effects, we avoid the composition bias that plagues the other two methods. The estimate lands closest to the true effect.

The takeaway: the choice of analytical framework isn't just an academic exercise. On this data, the naive approach underestimated the true effect by about 27%. If you presented that to leadership, you'd be telling them the intervention was significantly less effective than it actually was. The event-study design recovers most of the signal, landing within about 12% of the truth.

---

## What Could Go Wrong

No approach covers everything. Here are the main things to keep in mind.

### Pre-Period Contamination

Some trainees are already *in* training during the pre-period months. If training itself disrupts operations (trainee pulled away for classes, other staff covering, workload shifting), then the pre-period metrics might be slightly depressed compared to a truly untouched baseline. This biases your estimate *downward*: the pre-period looks worse than it would without any training disruption, so the measured lift is smaller than the real lift. If you find a positive effect despite this, the true effect is probably larger. That makes your estimates conservative.

### Selection Bias from Filtering

By filtering to locations with clean adoption windows and sufficient data, we're studying a specific subset. Locations that adopted within the expected timeframe might have had smoother operations or stronger management support. The treatment effect for those locations could be higher than what you'd see for locations that had a harder time with adoption. It's worth being upfront about this scope — the result speaks to locations that implemented properly, which is still a very useful answer, just not a universal one.

### Parallel Trends

This is the core assumption behind the whole framework. Treatment and control locations need to have been following the same trajectory before adoption — if they weren't, any post-adoption divergence could be driven by something other than the treatment. We can check this in the pre-period by looking at whether the two groups were trending together. We can't prove it holds in the post-period, but if the pre-trends align closely, that's strong supporting evidence.

### When to Use Something Else

This approach works well when you have a clear treatment and control group, treatment timing varies but you can define a bounded adoption window, you have enough pre and post periods, and the parallel trends assumption is plausible.

Consider alternatives when you don't have a control group (synthetic control methods), when treatment is continuous rather than binary (dose-response models), when adoption timing is endogenous (instrumental variables), or when you have very few treated units.

---

## Summary

Staggered adoption is easy to overlook, but it can quietly bias your estimates in ways that are hard to detect. Standard DiD assumes a clean before/after boundary. When treatment timing varies, that boundary doesn't exist, and forcing one creates composition bias that can dilute or reverse your results.

The fix is to stop looking for a universal boundary and let each unit define its own. Event-time normalization aligns all units to a common reference point, blacks out the messy transition period, and measures outcomes in relative time. The event-study regression estimates treatment effects at each distance from adoption, with built-in validation through the flat pre-period test.

A few things to remember. The adoption window threshold is a judgment call. Pick something defensible, then run sensitivity checks. Filtering to clean adopters trades sample size for a trustworthy read. Require balanced data on both sides of the window to avoid composition bias sneaking back in. And the parallel trends assumption matters more than anything else — always test it, always plot it.

This technique goes by several names: event-study design, staggered DiD, group-time treatment effects. Callaway and Sant'Anna (2021), Sun and Abraham (2021), and Goodman-Bacon (2021) are the formal references. But the intuition is simple: when treatment timing is messy, normalize it.

---

## References

- Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-Differences with multiple time periods. *Journal of Econometrics*, 225(2), 200-230. [Link](https://doi.org/10.1016/j.jeconom.2020.12.001)
- Sun, L., & Abraham, S. (2021). Estimating dynamic treatment effects in event studies with heterogeneous treatment effects. *Journal of Econometrics*, 225(2), 175-199. [Link](https://doi.org/10.1016/j.jeconom.2020.09.006)
- Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment timing. *Journal of Econometrics*, 225(2), 254-277. [Link](https://doi.org/10.1016/j.jeconom.2021.03.014)

---

## Appendix: Full Code

<details>
<summary><strong>Event-Study Regression in Python</strong></summary>

The regression takes event-time normalized data and estimates treatment effects at each relative period.

```python
import statsmodels.api as sm
import pandas as pd
import numpy as np

# Assumes a DataFrame 'event_df' with columns:
#   location_id, month, relative_period, metric, treatment_group
#
# relative_period: negative = pre-treatment, 0 = adoption window,
#                  positive = post-treatment
# treatment_group: 1 = treatment, 0 = control

# Filter to the analysis window (-6 through +6)
analysis_df = event_df[
    (event_df['relative_period'] >= -6) &
    (event_df['relative_period'] <= 6)
].copy()

# Create treatment × period interaction dummies
# Period -1 is the reference (omitted) category
for k in range(-6, 7):
    if k == -1:
        continue  # Reference period — all others measured relative to this
    col_name = f'treat_period_{k}'
    # Equals 1 only if this row is a treatment location AND in period k
    analysis_df[col_name] = (
        (analysis_df['relative_period'] == k) &
        (analysis_df['treatment_group'] == 1)
    ).astype(int)

# Location fixed effects: each location compared to itself
location_dummies = pd.get_dummies(
    analysis_df['location_id'], prefix='loc', drop_first=True
)

# Time fixed effects: absorb month-specific shocks hitting all locations
time_dummies = pd.get_dummies(
    analysis_df['month'], prefix='month', drop_first=True
)

# Assemble the regression matrix
period_cols = [f'treat_period_{k}' for k in range(-6, 7) if k != -1]
X = pd.concat([analysis_df[period_cols], location_dummies, time_dummies],
              axis=1)
X = sm.add_constant(X)
y = analysis_df['metric']

# OLS with clustered standard errors at the location level
# Clustering accounts for within-location correlation over time
# Without it, confidence intervals would be too narrow
model = sm.OLS(y, X).fit(
    cov_type='cluster',
    cov_kwds={'groups': analysis_df['location_id']}
)

# Extract the coefficients for the event-study plot
results = []
for k in range(-6, 7):
    if k == -1:
        # Reference period: coefficient is 0 by definition
        results.append({'period': k, 'coef': 0, 'ci_low': 0, 'ci_high': 0})
        continue
    col = f'treat_period_{k}'
    results.append({
        'period': k,
        'coef': model.params[col],
        'ci_low': model.conf_int().loc[col, 0],
        'ci_high': model.conf_int().loc[col, 1]
    })

results_df = pd.DataFrame(results)
print(results_df[['period', 'coef', 'ci_low', 'ci_high']].to_string(index=False))
```

</details>

<details>
<summary><strong>Full Simulation Script (Generates All Data and Plots)</strong></summary>

You can [download the complete simulation script](/assets/data/staggered_did_simulation.py) and run it yourself. It generates synthetic data mimicking staggered adoption, performs event-time normalization, runs the event-study regression, and produces all the data-driven plots in this post.

```bash
pip install numpy pandas matplotlib statsmodels seaborn
python staggered_did_simulation.py
```

Uses `np.random.seed(42)` for reproducibility.

</details>
