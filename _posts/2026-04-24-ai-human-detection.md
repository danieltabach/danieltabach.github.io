---
layout: single
title: "Can Humans Detect AI? Mining Textual Signals of AI-Assisted Writing Under Varying Scrutiny Conditions"
date: 2026-04-24
categories: [research]
tags: [python, experimentation, nlp, ai-detection, controlled-experiment, streamlit, supabase]
author_profile: true
classes: wide
header:
  teaser: /assets/images/posts/ai-human-detection/teaser.jpg
  og_image: /assets/images/posts/ai-human-detection/teaser.jpg
toc: true
toc_sticky: true
---

*Does the threat of being caught change the way people use AI to write? And if it does, can other people tell the difference?*

---

## Paper

**[Read the full paper on ArXiv](https://arxiv.org/abs/2604.23471)**

*Code and data available upon request.*

---

## Summary

This study asks whether the threat of AI detection changes how people write with AI, and whether other people can tell the difference. I built a two-phase controlled experiment where 21 participants wrote opinion pieces on remote work using an AI chatbot (Claude). Half were randomly warned that their submission would be scanned by an AI detection tool. The other half received no warning. Both groups had access to the same chatbot.

In Phase 2, 251 independent judges evaluated 1,999 paired comparisons, each time choosing which document in the pair was "written by a human." Judges were not told that both writers had access to AI.

### Key Findings

Across all evaluations, judges selected the warned writer's document as human **54.13%** of the time versus **45.87%** for the unwarned writer. A two-sided binomial test rejects chance guessing at **p = 0.000243**, and the result holds across both writing stances.

Yet on every measurable text feature I extracted, including AI overlap scores, lexical diversity, sentence structure, and pronoun usage, the two groups were indistinguishable. Four classifiers trained on 10 features could not beat chance. **The judges are picking up on something that feature-based methods do not capture.**

### Counter-Intuitive Mechanism

The warning did not reduce AI use. It **polarized** behavior. Treatment participants took more turns, spent more time, and used more tokens. Several treatment participants abandoned the chatbot entirely and wrote everything themselves. No control participant did this. Yet the submitted text was nearly identical on every measurable feature.

### What I Built

- **Writer platform:** Streamlit app with 9-stage session flow, Anthropic API (Claude Sonnet) chatbot, 3 column layout (chat, notepad, submission), counterbalanced assignment, soft timer
- **Judge platform:** Streamlit app with paired comparison, forced 10 second reading delay, optional document expansion, position randomization
- **Backend:** Supabase PostgreSQL with row level security, real time data writes
- **Testing:** 29 automated tests (AppTest and pytest), bot dry runs for abuse scenarios (prompt injection, mass token usage)
- **Recruitment:** Self funded. \\$25 raffle for writers, \\$50 raffle for judges. \\$100 Reddit ad campaign (32K impressions). Multi platform outreach across Reddit, LinkedIn, WhatsApp, and Instagram.
- **Analysis:** Custom AI overlap scoring (trigram overlap, longest common substring, sequence matching), 6 stylometric features, 4 classifiers with SMOTE and stratified CV

### Robustness

The effect strengthens under stricter filtering. Higher confidence judges lean further toward selecting the treatment document. Slower readers detect the signal more reliably. The result is distributed across the document pool, not driven by outliers. Removing the strongest outlier weakens but does not eliminate the effect.

---

## References

1. Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-Differences with multiple time periods. *Journal of Econometrics*, 225(2), 200-230.
2. Sun, L., & Abraham, S. (2021). Estimating dynamic treatment effects in event studies with heterogeneous treatment effects. *Journal of Econometrics*, 225(2), 175-199.
3. Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment timing. *Journal of Econometrics*, 225(2), 254-277.
4. Bargh, J. A., Chen, M., & Burrows, L. (1996). Automaticity of social behavior. *Journal of Personality and Social Psychology*, 71(2), 230-244.
5. Noy, S., & Zhang, W. (2023). Experimental evidence on the productivity effects of generative AI. *Science*, 381(6654), 187-192.
