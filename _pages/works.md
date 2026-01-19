---
layout: single
title: "Writing & Projects"
permalink: /works/
author_profile: true
---

<style>
.work-section {
  margin-bottom: 3rem;
}
.work-section h2 {
  border-bottom: 1px solid #444;
  padding-bottom: 0.5rem;
  margin-bottom: 0.5rem;
}
.work-section > p {
  color: #999;
  font-size: 0.95em;
  margin-bottom: 1.5rem;
}
.work-list {
  list-style: none;
  padding: 0;
  margin: 0;
}
.work-list li {
  margin-bottom: 1.25rem;
  padding-bottom: 1.25rem;
  border-bottom: 1px solid #333;
}
.work-list li:last-child {
  border-bottom: none;
}
.work-title {
  font-size: 1.1em;
  font-weight: 600;
}
.work-title a {
  text-decoration: none;
}
.work-meta {
  font-size: 0.85em;
  color: #888;
  margin-top: 0.25rem;
}
.work-tags {
  margin-top: 0.4rem;
}
.work-tags span {
  display: inline-block;
  background: #333;
  color: #ccc;
  padding: 0.15rem 0.5rem;
  border-radius: 3px;
  font-size: 0.75em;
  margin-right: 0.35rem;
  margin-bottom: 0.25rem;
}
.viz-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1.5rem;
}
.viz-card {
  border: 1px solid #444;
  border-radius: 6px;
  padding: 1.25rem;
  transition: border-color 0.2s;
}
.viz-card:hover {
  border-color: #666;
}
.viz-card a {
  text-decoration: none;
}
.viz-card .work-title {
  margin-bottom: 0.5rem;
}
.standalone-project {
  background: #1a1a1a;
  border: 1px solid #444;
  border-radius: 6px;
  padding: 1.25rem;
  margin-top: 1rem;
}
.standalone-project h4 {
  margin: 0 0 0.25rem 0;
}
.standalone-project .project-subtitle {
  color: #888;
  font-size: 0.9em;
  margin-bottom: 0.75rem;
}
.standalone-project p {
  margin-bottom: 1rem;
  font-size: 0.95em;
}
</style>

<div class="work-section">
<h2>Models From Scratch</h2>
<p>Implementing machine learning algorithms from scratch in Python and NumPy.</p>

<ul class="work-list">
{% assign from_scratch = site.posts | where_exp: "post", "post.categories contains 'from-scratch'" | sort: 'title' %}
{% for post in from_scratch %}
<li>
  <div class="work-title"><a href="{{ post.url }}">{{ post.title }}</a></div>
  <div class="work-meta">{{ post.date | date: "%B %Y" }}</div>
  <div class="work-tags">
    {% for tag in post.tags limit:4 %}
    <span>{{ tag }}</span>
    {% endfor %}
  </div>
</li>
{% endfor %}
</ul>
</div>

---

<div class="work-section">
<h2>Applied Projects</h2>
<p>Real-world applications of ML and data analysis.</p>

<ul class="work-list">
{% assign applied = site.posts | where_exp: "post", "post.categories contains 'applied'" %}
{% for post in applied %}
<li>
  <div class="work-title"><a href="{{ post.url }}">{{ post.title }}</a></div>
  <div class="work-meta">{{ post.date | date: "%B %Y" }}</div>
  <div class="work-tags">
    {% for tag in post.tags limit:4 %}
    <span>{{ tag }}</span>
    {% endfor %}
  </div>
</li>
{% endfor %}
</ul>

<div class="standalone-project">
  <h4>Drift Detection Algorithm Benchmarking</h4>
  <div class="project-subtitle">Georgia Tech | Research Project</div>
  <p>Comparative analysis of time-series drift detection algorithms (PELT, ADWIN, CUSUM, KS, Bayesian Changepoint) for production ML monitoring. Evaluated detection latency, false positive rates, and implementation complexity across noisy, evolving data environments.</p>
  <a href="https://github.com/danieltabach" class="btn btn--primary">View on GitHub</a>
</div>
</div>

---

<div class="work-section">
<h2>Data Visualization</h2>
<p>Interactive dashboards and visual explorations.</p>

<div class="viz-grid">
{% assign dataviz = site.posts | where_exp: "post", "post.categories contains 'data-viz'" %}
{% for post in dataviz %}
<div class="viz-card">
  <a href="{{ post.url }}">
    <div class="work-title">{{ post.title }}</div>
    <div class="work-meta">{{ post.date | date: "%B %Y" }}</div>
    <div class="work-tags">
      {% for tag in post.tags limit:3 %}
      <span>{{ tag }}</span>
      {% endfor %}
    </div>
  </a>
</div>
{% endfor %}
</div>
</div>
