---
layout: single
title: "When Treatment Timing Is Messy: Event-Study Design for Staggered Rollouts"
date: 2026-03-05
categories: [applied]
tags: [python, causal-inference, difference-in-differences, event-study, econometrics]
author_profile: true
classes: wide
header:
  teaser: /assets/images/posts/staggered-did/event-study-plot.png
toc: false
---

*You designed an experiment. Leadership signed off. The rollout took on a life of its own. Now what?*

---

## Introduction

You work at a company with hundreds of locations. The data science team designs an experiment: roll out new software to a subset of locations, keep the rest as a control group, and measure the impact on some efficiency metric. Everything is ready to go and the experiment looks sound.

In practice though, large-scale rollouts almost always end up staggered. The rollout was supposed to start in March across all treatment locations simultaneously. Instead, some locations start in March, some in June, and a few don't start until September. By the end of the year, every treatment location adopted on its own timeline, and there's no single "before" and "after" to compare.

This is the staggered adoption problem, one of the most common compliance challenges in field experiments. The good news: there's a well-established framework for handling it.

This post walks through how standard Difference-in-Differences works, where the assumptions break under staggered adoption, and how an approach called "event-study design" fixes it. Full code is in the [Appendix](#appendix-the-formal-model) if you want to skip the wall of text.

---

## What Does Staggered Adoption Actually Look Like?

Lets look at single-location examples with 3 trainees who each need 6 months of training on the new software. Each orange arrow below is one person's training timeline (For this example lets assume in the data we assume they either start/end on the first/last day of the month):

There is no single month where treatment switches on. Every location defines its own adoption timeline.
![Staggered Adoption Skeleton](/assets/images/posts/staggered-did/skeleton_staggered.png)
*For these example, each trainee is on different timelines. Some managers start everyone together, others spread it out.*

---

## Why Simpler Approaches Fall Short

The natural first instinct: just pick a single cutoff date. Maybe you use the median adoption month, or the month when "most" locations were done. Then you compare everything before that date to everything after.

The problem is; at any given calendar month, your treatment group becomes a *mix* of locations at completely different stages. Some have been fully operational on the new software for months. Some just finished training. Some haven't started. When you average across all of them, you're averaging together strong effects, weak effects, and zero effects. The result is a diluted number that underestimates the real impact.

This is **composition bias**. The mix of locations changes at every time point. At Month 8, your average might include some fully-adopted locations, some mid-training, and some that haven't started. At Month 14, it's a completely different mix. You're not comparing the same thing to itself across time. The composition of your treatment group is shifting underneath you, and that shifting can even flip the directionality of your estimate.

![Naive DiD Problem](/assets/images/posts/staggered-did/naive-did-problem.png)
*Left: A naive single-cutoff DiD looks clean but averages across locations at different adoption stages. Right: The actual location-level data is messy. A single cutoff blurs all of this together.*

There *is* a standard approach to this kind of problem called "two-way fixed effects" (TWFE): a single regression that controls for each location and each time period, then estimates one overall treatment effect. It works well when everyone gets treated at the same time. However, when adoption is staggered to a larger extent (like in this example), it quietly starts using already-treated locations as comparisons for late adopters, which can pull the estimate in the wrong direction. [Callaway and Sant'Anna (2021)](https://doi.org/10.1016/j.jeconom.2020.12.001) and [Sun and Abraham (2021)](https://doi.org/10.1016/j.jeconom.2020.09.006) formalized this problem and showed just how misleading those estimates can be when treatment effects evolve over time.

---

## The Fix: Normalize to Event Time

Instead of forcing a single cutoff, we let each location define its own timeline (for this example we filter this downstream).

### Defining the Adoption Window

For each treatment location, we identify the **adoption window**: the period from when the *first* trainee completes training to when the *last* trainee completes training.

![Pre-Period, Period 0, Post-Period Zones](/assets/images/posts/staggered-did/pre-zero-post-examples.png)
*The same locations from above, now with the framework overlaid. Red = Pre-Period (baseline), gray = Period 0 (adoption window, blacked out), blue = Post-Period (where we measure the effect).*

This window is the messy middle. Some trainees have the new skills, others don't. The location is in flux. Think of it like a mid-season roster overhaul. You wouldn't judge the new lineup during the first few games when players are still learning each other's tendencies. You'd compare the team's record before the trades to their record after the new players have settled in.

### The Three Zones

We split each location's timeline into three zones:

**Pre-Period (before the window):** The location before any trainee completed training. This is the baseline. We measure metrics here and call these periods -1, -2, -3, and so on, counting backwards from the adoption window.

**Period 0 (the adoption window itself):** We black this out. Don't measure it. Don't interpret it. It's the mid-roster-change stretch. Metrics here are a blend of old and new with no clean read.

**Post-Period (after the window):** The location after *all* trainees finished training. This is where we look for the treatment effect. We call these periods +1, +2, +3, counting forward.

### The Shift

Here's where it comes together. Every treatment location, regardless of when it actually adopted, is now on the same relative timeline. Period -3 means "three months before the adoption window started" for every location. Period +2 means "two months after the last trainee finished" for every location.

![Event Time Normalization](/assets/images/posts/staggered-did/event-time-normalization.png)
*Left: The same locations in calendar time, their adoption points are scattered across different months. Right: After normalizing to event time, they all align at Period 0. Different calendar dates, same relative timeline.*

In the chart, the star markers indicate the moment when each location's first trainee completed training (the point where treatment starts taking effect).

---

## Getting a Clean Sample

Trimming the sample feels counterintuitive, but keeping every location regardless of data quality introduces more noise than it removes. We apply three filters, and each one has the same underlying logic: without it, the read is less trustworthy.

### Filter 1: Bounded Adoption Window

If a location's adoption window stretches beyond 6 months, we set it aside for this analysis. If it took 14 months from first completion to last, the blackout window would eat almost the entire timeline. There wouldn't be enough clean pre or post data left to measure anything meaningful.

Most locations in our dataset fell within this window. Training itself takes 6 months, and most managers either trained everyone together (adoption window = 0 months) or staggered within a few months. The locations with extremely long adoption windows were the exception.

This filter has a downside worth noting. By blacking out the entire adoption window, we lose information about what happens during the transition itself. If some trainees are already creating impact while others are still in training, that partial effect gets thrown away. In cases where the transition period IS the thing you want to study (for example, measuring how quickly a team ramps up), a different framework would be more appropriate.

### Filter 2: Minimum Pre-Period Data

In this example, we use at least 6 months of pre-period data for every location. We need to validate that treatment and control locations were trending similarly *before* adoption. If a location only has 2 months of pre-data, we can't tell whether its pre-trend was flat, rising, or falling. That makes the parallel trends test unreliable for that location.

There's a more technical reason too. If some locations contribute data at Period -6 and others don't, the *set of locations* in your average changes across time periods. That's composition bias again, the same problem that broke the single-cutoff approach. By requiring every location to have data across the full analysis window, we keep the composition constant. Period -6 and Period +6 contain the exact same set of locations. The comparison is apples to apples.

### Filter 3: Minimum Post-Period Data

Same logic in the other direction. We need enough post-adoption months to actually measure whether the treatment effect materializes, ramps up, or fades. If we only see one month of post-data, we can't tell whether the lift is real, still building, or just noise from a good month. A location with that little data contributes almost nothing to the analysis.

### The Tradeoff

Every filter costs sample size. We're trading breadth for a cleaner read. It's worth being upfront about this: the treatment effect we estimate applies to "locations that adopted within a reasonable timeframe and had enough data on both sides." That's a valid and useful population, but it's not the same as "all locations." In the causal inference literature, this is similar to a **Local Average Treatment Effect (LATE)**: the effect for the compliers, not the full population.

---

## Treatment vs. Control

We've focused on treatment locations, but we haven't talked about the control group yet.

Control locations never adopted the software during the observation period. Their role is to capture what would have happened to the metric if no one adopted anything. For each treatment location at, say, Period +2 (which might land in May for one location and August for another), we compare that metric to the control group's data at the same calendar month. The control group shows us the background trend at every point in time.

Here's what that looks like concretely. Three treatment locations adopted at different times. For each one, the arrows show the comparison: treatment metric vs. control metric at the same calendar month.

![Control Group Comparison](/assets/images/posts/staggered-did/control-comparison.png)
*Three treatment locations with different adoption timings. The double arrows show where each location's post-adoption metric is compared to the control group average at the same calendar month. Different event times, same calendar-month comparison.*

A reasonable question here is whether choosing concurrent comparisons introduces bias. It doesn't. This is the standard Difference-in-Differences setup: comparing outcomes at the same point in time between a group that received the intervention and a group that didn't. The control group serves as the counterfactual. If the metric would have gone up anyway (due to seasonality, company-wide trends, or market conditions), the control group captures that, and the difference strips it out. The control group IS the holdout.

The **parallel trends assumption** is the foundation. Before any treatment happened, were treatment and control locations moving in the same direction at the same rate? If yes, we can attribute the post-adoption divergence to the treatment. If not, the results become much harder to interpret.

For visualization purposes (plotting treatment vs control on the same event-time axis), we align the control group to a reference point based on the median adoption timing. This is just a presentation choice so we can put both lines on the same chart. The actual analysis compares treatment and control at the same calendar time through the regression's time fixed effects.

---

## Validating the Setup

Before looking at the results, we validate that the framework is working.

### Parallel Trends

![Parallel Trends](/assets/images/posts/staggered-did/parallel-trends.png)
*Average metric for treatment and control groups over calendar time. The two groups track closely in the pre-treatment period, then diverge after adoption begins.*

The two groups moved together before treatment started. That's not *proof* that the assumption holds in the post-period (it's fundamentally untestable there), but it's the best evidence we can get.

### Event-Study Coefficients

The event-study regression (details in the [Appendix](#appendix-the-formal-model)) estimates a separate treatment effect at each relative time period. This gives us two things: a confirmation that the setup is sound, and a look at how the effect builds over time.

![Event Study Plot](/assets/images/posts/staggered-did/event-study-plot.png)
*Each point is the estimated treatment effect at that relative period. The shaded band is the 95% confidence interval. Pre-period estimates hover near zero. Post-period estimates show a clear positive lift.*

**Pre-period (Periods -6 to -1):** The coefficients hover near zero and their confidence intervals cross zero. This confirms treatment and control locations were trending the same way *before* adoption happened. If any of these were significantly different from zero, it would mean the two groups were already diverging before the software rollout, and the entire analysis would be suspect.

**Post-period (Periods +1 to +6):** The effect starts at about +2.8 in Period +1 and ramps up to about +4.5 by Period +6. This ramp-up pattern makes sense. It takes time for people to get comfortable with a new system. You wouldn't expect efficiency to instantly jump the day after the last person finishes training.

---

## Results: What Did We Find?

The idea behind DiD is simple: look at how much the treatment group's metric changed from pre to post, subtract how much the control group changed over the same window (to strip out background trends), and whatever is left is the treatment effect.

### The Numbers

| Group | Pre-Period Avg | Post-Period Avg | Change |
|-------|---------------|-----------------|--------|
| Treatment | 53.67 | 59.75 | +6.08 |
| Control | 53.44 | 55.54 | +2.10 |
| **DiD Estimate** | | | **+3.98** |

The treatment group's metric went up by 6.08. The control group also went up by 2.10 (background trend, maybe seasonality, maybe the whole market improving). The difference-in-differences strips out that background trend: 6.08 - 2.10 = **3.98 units attributable to the treatment**.

That's the kind of number that changes a recommendation. If the lift is real and meaningful, it justifies expanding the rollout. If the naive approach had been the only estimate on the table, you'd be underselling the program by nearly 30%.

### Treatment vs. Control Over Time

Here's the most intuitive view. The average metric for treatment and control groups, shown both in calendar time (left) and event time (right):

![Treatment vs Control Lift](/assets/images/posts/staggered-did/treatment-vs-control-lift.png)
*Left: In calendar time, the treatment group gradually pulls ahead as more locations complete adoption. Right: In event time, the gap becomes crisp. You can see the treatment effect building cleanly after Period 0.*

In calendar time, the divergence is gradual because locations adopt at different times (the staggering blurs the signal). In event time, the gap snaps into focus because every location is aligned to the same reference point.

The event-study regression gives us something richer than a single number: the effect at each relative period. Averaging the post-period coefficients (Periods +1 through +6) gives us an overall treatment effect estimate of about 3.94, consistent with the simple DiD calculation above.

---

## What Happens When You Get It Wrong

This is why the framework matters. We ran the same simulated data through three different estimation approaches to see how much the choice of method affects the result. Since this is synthetic data, we know the true treatment effect built into the simulation (4.5 units at full ramp-up), so we can measure how far off each approach lands.

### Approach A: Naive Calendar Cutoff

Pick the median adoption month, split everything into "before" and "after," compare treatment to control. This is the simplest possible approach and what most people try first. It systematically underestimates because at any given calendar month, your treatment group includes locations at completely different stages (some fully adopted, some mid-training, some not started). You're averaging together real effects with zeros.

### Approach B: Standard Two-Way Fixed Effects (TWFE)

Run a single regression with location fixed effects, time fixed effects, and one treatment indicator. Unlike the naive approach, this uses each location's actual adoption date to define "post-treatment." So a location that adopted in March counts as "treated" starting in March, while a location that adopted in August starts in August. That's better than forcing a single cutoff on everyone.

The problem is subtler. TWFE estimates a single overall treatment coefficient. To get that one number, it compares treated locations to untreated ones. But when adoption is staggered, some locations adopted early and some adopted late. By the time the late adopters start their post-period, the early adopters have already been treated for months. The regression quietly uses those already-treated early adopters as part of the comparison group for the late adopters. Panel B in the chart below shows this: early and late adopters are on different trajectories, but TWFE collapses them into one coefficient. The estimate improves over naive, but it still carries bias.

### Approach C: Event-Study Design

The approach we've been building throughout this post. By normalizing each location to its own event time and estimating period-by-period effects, we avoid the composition bias that plagues the other two methods. The estimate lands closest to the true effect.

### The Comparison

![Method Comparison](/assets/images/posts/staggered-did/method-comparison.png)
*The naive calendar cutoff underestimates by about 27%. Standard TWFE improves to about 16% bias. The event-study design gets closest to the truth, landing within 12%.*

Here's how the three approaches look visually on the same data:

![Method Lift Comparison](/assets/images/posts/staggered-did/method-lift-comparison.png)
*Same data, three approaches. Panel A uses a single calendar cutoff. Panel B splits treatment into early and late adopters, showing the heterogeneity that TWFE pools into one number. Panel C normalizes to event time, where the treatment effect snaps into focus.*

The takeaway: the choice of analytical framework isn't just an academic exercise. On this data, the naive approach underestimated the true effect by about 27%. If you presented that to leadership, you'd be telling them the intervention was significantly less effective than it actually was. The event-study design recovers most of the signal, landing within about 12% of the truth.

---

## Limitations

No approach covers everything. Here are the main things to keep in mind.

### Pre-Period Contamination

Some trainees are already *in* training during the pre-period months. If training itself disrupts operations (trainee pulled away for classes, other staff covering, workload shifting), then the pre-period metrics might be slightly depressed compared to a truly untouched baseline. This biases your estimate *downward*: the pre-period looks worse than it would without any training disruption, so the measured lift is smaller than the real lift. If you find a positive effect despite this, the true effect is probably larger. That makes your estimates conservative.

### Selection Bias from Filtering

By filtering to locations with clean adoption windows and sufficient data, we're studying a specific subset. Locations that adopted within the expected timeframe might have had smoother operations or stronger management support. The treatment effect for those locations could be higher than what you'd see for locations that had a harder time with adoption. It's worth being upfront about this scope. The result speaks to locations that implemented properly, which is still a very useful answer, just not a universal one.

### Parallel Trends

This is the core assumption behind the whole framework. Treatment and control locations need to have been following the same trajectory before adoption. If they weren't, any post-adoption divergence could be driven by something other than the treatment. We can check this in the pre-period by looking at whether the two groups were trending together. We can't prove it holds in the post-period, but if the pre-trends align closely, that's strong supporting evidence.

### When to Use Something Else

This approach works well when you have a clear treatment and control group, treatment timing varies but you can define a bounded adoption window, you have enough pre and post periods, and the parallel trends assumption is plausible.

Consider alternatives when you don't have a control group (synthetic control methods), when treatment is continuous rather than binary (dose-response models), when adoption timing is endogenous (instrumental variables), or when you have very few treated units.

---

## Summary

Staggered adoption is easy to overlook, but it can quietly bias your estimates in ways that are hard to detect. Standard DiD assumes a clean before/after boundary. When treatment timing varies, that boundary doesn't exist, and forcing one creates composition bias that can dilute or reverse your results.

The fix: stop looking for a universal boundary and let each unit define its own. Normalize to event time, black out the messy transition, and measure outcomes relative to each unit's adoption. The event-study regression gives you treatment effects at each distance from adoption, with built-in validation through the flat pre-period test.

A few things to keep in mind. The adoption window threshold is a judgment call, so pick something defensible and run sensitivity checks. Filtering to clean adopters trades sample size for a trustworthy read. Require balanced data on both sides of the window to avoid composition bias sneaking back in. And the parallel trends assumption matters more than anything else. Always test it, always plot it.

If you're running experiments at scale and adoption timing isn't perfectly controlled (it almost never is), this is the framework. It goes by several names: event-study design, staggered DiD, group-time treatment effects. Callaway and Sant'Anna (2021), Sun and Abraham (2021), and Goodman-Bacon (2021) are the formal references. But the intuition is simple: when treatment timing is messy, normalize it.

---

## References

- Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-Differences with multiple time periods. *Journal of Econometrics*, 225(2), 200-230. [Link](https://doi.org/10.1016/j.jeconom.2020.12.001)
- Sun, L., & Abraham, S. (2021). Estimating dynamic treatment effects in event studies with heterogeneous treatment effects. *Journal of Econometrics*, 225(2), 175-199. [Link](https://doi.org/10.1016/j.jeconom.2020.09.006)
- Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment timing. *Journal of Econometrics*, 225(2), 254-277. [Link](https://doi.org/10.1016/j.jeconom.2021.03.014)

---

## Appendix: The Formal Model

<details markdown="1">
<summary><strong>Event-Study Regression Specification</strong></summary>

The event-study regression formalizes the period-by-period DiD comparison. The model:

$$Y_{it} = \alpha_i + \gamma_t + \sum_{k \neq -1} \beta_k \cdot D_{it}^{k} + \epsilon_{it}$$

Here is what each piece means in plain language.

**$Y_{it}$** is the efficiency metric for location $i$ in month $t$.

**$\alpha_i$** is a **location fixed effect**. Each location is compared to itself. Location 101 might naturally run at a higher efficiency than Location 205. The fixed effect absorbs that difference. We're not comparing Location 101 to Location 205. We're comparing Location 101 *before adoption* to Location 101 *after adoption*.

**$\gamma_t$** is a **time fixed effect**. This absorbs anything that affected *all* locations in a given month. Maybe December is always slower because of holidays. Maybe there was a system outage in Month 14 that hit everyone. Time fixed effects soak that up.

**$D_{it}^{k}$** is a dummy variable equal to 1 if location $i$ is a treatment location and is in relative period $k$. This is how we isolate the treatment effect at each specific distance from adoption.

**$\beta_k$** is what we care about. These are the **treatment effect coefficients**. $\beta_{+3}$ tells us: "three months after the adoption window closed, how much higher (or lower) was the metric for treatment locations compared to what we'd expect?"

We omit Period -1 as the reference category. All other $\beta_k$ values are measured relative to the period right before adoption started completing. This gives us a built-in sanity check: the pre-period betas should be near zero.

</details>

---

## Appendix: Full Code

The complete simulation is below. It generates synthetic panel data, normalizes to event time, runs the event-study regression, and benchmarks against naive and TWFE approaches. You can also [download the script](/assets/data/staggered_did_simulation.py) which includes all plotting functions.

```bash
pip install numpy pandas matplotlib statsmodels seaborn
python staggered_did_simulation.py
```

<details markdown="1">
<summary><strong>1. Generate Synthetic Panel Data</strong></summary>

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

np.random.seed(42)

def generate_synthetic_data():
    """
    Create a panel dataset of 200 locations over 24 months.
    120 treatment locations adopt new software on staggered
    timelines. 80 control locations never adopt.

    Treatment effect: ramps up over 3 months post-adoption,
    peaking at 4.5 units (3 months * 1.5 per month).
    """
    records = []

    # --- Treatment locations (IDs 1 through 120) ---
    for loc_id in range(1, 121):
        # Each location starts adoption in a random month
        adoption_start = np.random.randint(3, 16)

        # 1-5 employees per location
        n_employees = np.random.randint(1, 6)

        # 40% train everyone together, 60% stagger
        if np.random.random() < 0.4:
            starts = [adoption_start] * n_employees
        else:
            spread = np.random.randint(1, 4)
            starts = [
                adoption_start + np.random.randint(0, spread)
                for _ in range(n_employees)
            ]

        # Training takes 6 months per person
        completions = [s + 6 for s in starts]
        first_completion = min(completions)
        last_completion = max(completions)
        window = last_completion - first_completion

        # Skip locations with adoption windows > 6 months
        if window > 6:
            continue

        for month in range(1, 25):
            # Baseline metric with trend and noise
            baseline = 50 + 0.3 * month
            noise = np.random.normal(0, 2)

            # Treatment effect ramps up after first completion
            effect = 0
            if month >= first_completion:
                months_post = min(
                    month - first_completion + 1, 3
                )
                effect = months_post * 1.5

            records.append({
                'location_id': loc_id,
                'month': month,
                'metric': baseline + effect + noise,
                'treatment': 1,
                'first_completion': first_completion,
                'last_completion': last_completion,
                'adoption_window': window,
            })

    # --- Control locations (IDs 121 through 200) ---
    for loc_id in range(121, 201):
        for month in range(1, 25):
            baseline = 50 + 0.3 * month
            noise = np.random.normal(0, 2)

            records.append({
                'location_id': loc_id,
                'month': month,
                'metric': baseline + noise,
                'treatment': 0,
                'first_completion': np.nan,
                'last_completion': np.nan,
                'adoption_window': np.nan,
            })

    return pd.DataFrame(records)
```

</details>

<details markdown="1">
<summary><strong>2. Normalize to Event Time</strong></summary>

```python
def normalize_to_event_time(data):
    """
    Convert calendar time to relative event time.

    Treatment locations: event_time = month - first_completion
      Period -3 = three months before first trainee finished
      Period  0 = month of first completion (adoption window)
      Period +2 = two months after last trainee finished

    Control locations: aligned to median treatment timing
    for visualization purposes. The regression uses time
    fixed effects for the actual comparison.
    """
    median_adoption = data.loc[
        data['treatment'] == 1, 'first_completion'
    ].median()

    data['event_time'] = np.nan
    data['period'] = np.nan

    # Treatment: relative to their own first_completion
    treat = data['treatment'] == 1
    data.loc[treat, 'event_time'] = (
        data.loc[treat, 'month']
        - data.loc[treat, 'first_completion']
    )

    # Control: relative to median adoption time
    ctrl = data['treatment'] == 0
    data.loc[ctrl, 'event_time'] = (
        data.loc[ctrl, 'month'] - median_adoption
    )

    # Bin into periods -6 through +6
    in_window = data['event_time'].between(-6, 6)
    data.loc[in_window, 'period'] = (
        data.loc[in_window, 'event_time'].astype(int)
    )

    return data
```

</details>

<details markdown="1">
<summary><strong>3. Event-Study Regression</strong></summary>

```python
def run_event_study_regression(data):
    """
    Estimate treatment effects at each relative period.

    Model: Y_it = alpha_i + gamma_t + sum(beta_k * D_it^k)

    D_it^k = 1 if location i is treated AND in period k.
    Period -1 is the omitted reference category.
    Pre-period betas near zero = parallel trends confirmed.
    Post-period betas = treatment effect at each distance.
    """
    reg_data = data[
        data['period'].notna()
        & (data['period'] >= -6)
        & (data['period'] <= 6)
    ].copy()

    # Treatment x period interaction dummies
    # Each one isolates the treatment effect at period k
    for k in range(-6, 7):
        reg_data[f'period_{k}'] = (
            (reg_data['period'] == k)
            & (reg_data['treatment'] == 1)
        ).astype(int)

    # Drop period -1 (reference category)
    reg_data = reg_data.drop(columns=['period_-1'])

    # Fixed effects
    loc_fe = pd.get_dummies(
        reg_data['location_id'],
        prefix='loc', drop_first=True
    )
    time_fe = pd.get_dummies(
        reg_data['month'],
        prefix='month', drop_first=True
    )

    # Build regression matrix
    period_cols = [
        f'period_{k}' for k in range(-6, 7) if k != -1
    ]
    X = pd.concat(
        [reg_data[period_cols], loc_fe, time_fe], axis=1
    ).astype(float)
    X = sm.add_constant(X, has_constant='add')
    Y = reg_data['metric'].astype(float)

    results = sm.OLS(Y, X).fit()

    # Extract coefficients for each period
    coefs, ci_lo, ci_hi = {}, {}, {}
    for k in range(-6, 7):
        if k == -1:
            coefs[k], ci_lo[k], ci_hi[k] = 0, 0, 0
            continue
        col = f'period_{k}'
        coefs[k] = results.params[col]
        ci = results.conf_int().loc[col]
        ci_lo[k], ci_hi[k] = ci[0], ci[1]

    return results, coefs, ci_lo, ci_hi
```

</details>

<details markdown="1">
<summary><strong>4. DiD Summary and Benchmarking</strong></summary>

```python
def compute_did_summary(data):
    """
    Simple DiD table: pre/post averages for treatment
    and control. Pre = periods -6 to -1, post = +1 to +6.
    """
    analysis = data[
        data['period'].notna() & (data['period'] != 0)
    ].copy()
    analysis['post'] = (analysis['period'] > 0).astype(int)

    summary = analysis.groupby(
        ['treatment', 'post']
    )['metric'].mean().unstack()
    summary.columns = ['Pre Avg', 'Post Avg']
    summary['Change'] = summary['Post Avg'] - summary['Pre Avg']
    summary.index = ['Control', 'Treatment']

    did = (
        summary.loc['Treatment', 'Change']
        - summary.loc['Control', 'Change']
    )
    print(f"DiD Estimate: {did:.2f}")
    return summary, did


def run_naive_cutoff(data):
    """
    Naive approach: single calendar cutoff at the median
    adoption month. Splits all data into before/after.
    Underestimates because of composition bias.
    """
    cutoff = int(
        data.loc[data['treatment'] == 1, 'first_completion']
        .median()
    )
    pre = data[data['month'] < cutoff]
    post = data[data['month'] >= cutoff]

    treat_change = (
        post[post['treatment'] == 1]['metric'].mean()
        - pre[pre['treatment'] == 1]['metric'].mean()
    )
    ctrl_change = (
        post[post['treatment'] == 0]['metric'].mean()
        - pre[pre['treatment'] == 0]['metric'].mean()
    )
    return treat_change - ctrl_change


def run_twfe(data):
    """
    Two-way fixed effects: single treatment coefficient.
    Uses each location's actual adoption date (better than
    naive), but still pools into one average effect.
    Under staggered timing, already-treated locations
    implicitly serve as comparisons for late adopters.
    """
    cutoff = int(
        data.loc[data['treatment'] == 1, 'first_completion']
        .median()
    )
    reg = data.copy()

    # Treatment locations: post = after THEIR adoption
    treat = reg['treatment'] == 1
    reg.loc[treat, 'post'] = (
        reg.loc[treat, 'month']
        >= reg.loc[treat, 'first_completion']
    ).astype(int)

    # Control locations: post = after median cutoff
    ctrl = reg['treatment'] == 0
    reg.loc[ctrl, 'post'] = (
        reg.loc[ctrl, 'month'] >= cutoff
    ).astype(int)

    reg['treat_x_post'] = reg['treatment'] * reg['post']

    # Fixed effects
    loc_fe = pd.get_dummies(
        reg['location_id'], prefix='loc', drop_first=True
    )
    time_fe = pd.get_dummies(
        reg['month'], prefix='month', drop_first=True
    )

    X = pd.concat(
        [reg[['treat_x_post']], loc_fe, time_fe], axis=1
    ).astype(float)
    X = sm.add_constant(X, has_constant='add')
    Y = reg['metric'].astype(float)

    model = sm.OLS(Y, X).fit()
    return model.params['treat_x_post']


# --- Run everything ---
data = generate_synthetic_data()
data = normalize_to_event_time(data)

results, coefs, ci_lo, ci_hi = run_event_study_regression(data)
summary, did = compute_did_summary(data)

naive = run_naive_cutoff(data)
twfe = run_twfe(data)
event_study_avg = np.mean([coefs[k] for k in coefs if k > 0])
true_effect = 4.5  # Built into the simulation

print(f"\nNaive:       {naive:.2f} ({((naive-true_effect)/true_effect)*100:+.0f}% bias)")
print(f"TWFE:        {twfe:.2f} ({((twfe-true_effect)/true_effect)*100:+.0f}% bias)")
print(f"Event-Study: {event_study_avg:.2f} ({((event_study_avg-true_effect)/true_effect)*100:+.0f}% bias)")
print(f"True Effect: {true_effect}")
```

Uses `np.random.seed(42)` for reproducibility. The [full script](/assets/data/staggered_did_simulation.py) includes all plotting functions that generate the charts in this post.

</details>
