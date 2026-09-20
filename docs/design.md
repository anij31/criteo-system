# Production-Grade ML System

## Project 1: Criteo Click-Through Rate Prediction

**Status:** Draft
**Version:** 0.1
**Date:** 2026-09-20
**Authors:** Project Team
**Review Type:** Pre-implementation design review

---

## 1. Context and Scope

We are building the first project in a multi-project machine-learning engineering program whose primary objective is to develop **production ML engineering intuition and muscle memory**, not to showcase a sophisticated model.

The first system will solve a tabular binary-classification problem using the Criteo click-through-rate (CTR) dataset. Criteo describes the dataset as advertising traffic collected over 24 days, with each row representing a display ad served and the first field representing whether it was clicked. It contains 13 integer features and 26 anonymized categorical features; values may be missing, and rows are chronologically ordered.

The first model will intentionally be **logistic regression**. The purpose is to establish a trustworthy end-to-end ML system before introducing model complexity. This follows Google's guidance to get the infrastructure right around a simple first model.

The system will run primarily on a Fedora Linux machine with 16 GB RAM and 1 TB SSD. Google Colab will be used as ephemeral additional compute when local resources are insufficient. No paid cloud runtime will be required.

The architecture will nevertheless be designed with **cloud portability**, particularly GCP portability, in mind.

This project is intentionally treated as a production system even though it is an educational/hackathon environment.

---

## 2. Problem Statement

Build a reliable ML system that:

1. ingests a realistic, messy tabular dataset;
2. creates a reproducible training dataset;
3. validates data before training;
4. performs explicit feature engineering;
5. trains and evaluates a simple model;
6. tracks experiments and model lineage;
7. registers a deployable model;
8. exposes predictions through an inference service;
9. measures application/system/model behavior;
10. detects data and model problems;
11. supports automated testing;
12. can be deliberately broken and subsequently debugged;
13. can eventually expose selected operational metrics through a public project website;
14. can be mapped conceptually onto a GCP deployment architecture.

Google's Rules of ML emphasize solid end-to-end infrastructure, simple initial models, explicit metrics, testable infrastructure, careful feature engineering, temporal evaluation, and monitoring for training-serving skew. These principles form the foundation of this design.

---

## 3. Goals

### Primary Goals

* Build a complete ML lifecycle rather than an isolated notebook.
* Make the entire lifecycle reproducible.
* Establish testing as part of the ML system rather than as an afterthought.
* Use a simple model so that engineering failures remain visible.
* Maintain a single coherent path for feature processing in training and inference.
* Implement data, model, service, and infrastructure observability.
* Practice realistic failure detection and debugging.
* Make the system modular enough to reuse for future ML projects.
* Make the local architecture conceptually transferable to GCP, AWS, or Azure.
* Produce a demonstrable system suitable for technical interviews and portfolio review.

### Success Criteria

The project is considered successful when:

`versioned data → validated data → reproducible features → trained model → evaluated model → registered model → served model → observable inference`

can be executed repeatedly without manually modifying hidden state.

A second success criterion is that intentionally introduced failures produce **observable evidence and a diagnosable failure path**.

---

## 4. Non-Goals

The first project will **not** attempt to:

* deploy actual production workloads to paid GCP infrastructure;
* process the entire Criteo corpus locally;
* maximize predictive performance;
* build a state-of-the-art CTR model;
* operate a Kubernetes cluster;
* implement every possible MLOps technology;
* build a custom data warehouse;
* implement real-time streaming infrastructure;
* reproduce Criteo's proprietary feature semantics;
* create a massive distributed training cluster.

These are deliberate constraints, not omissions.

---

# 5. Design

## 5.1 Design Principles

The project follows these principles:

**Infrastructure before sophistication.**
The first model is intentionally simple.

**Metrics before optimization.**
We define what the system measures before trying to improve the model. Google explicitly recommends designing and implementing metrics early.

**One path where possible.**
Training and serving should share feature-transformation code to reduce training-serving skew. Google explicitly recommends reusing code between training and serving.

**Fail early.**
Invalid data should fail validation before expensive training.

**Everything important is versioned.**
Code, data definition, sampled dataset, features, experiments, and model artifacts need traceability.

**Observe before debugging.**
A system that cannot show evidence of failure is difficult to debug.

**Prefer established technology.**
We use mature tools rather than building infrastructure ourselves.

**Minimize unnecessary infrastructure.**
A component must justify its operational and learning value.

**Design for replacement.**
A future recommendation model, clustering system, neural network, or LLM should be able to reuse the same broad lifecycle.

---

## 5.2 High-Level Architecture

```text
                         ┌──────────────────────┐
                         │      GitHub          │
                         │ Code + CI/CD + Docs  │
                         └──────────┬───────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────┐
│                         FEDORA HOST                         │
│                                                             │
│  ┌─────────────┐     ┌──────────────┐     ┌─────────────┐  │
│  │ Data Source │ --> │ Data Ingest  │ --> │ DVC/MinIO   │  │
│  └─────────────┘     └──────────────┘     └──────┬──────┘  │
│                                                   │         │
│                                                   ▼         │
│                                           ┌─────────────┐   │
│                                           │ Validation  │   │
│                                           └──────┬──────┘   │
│                                                  │          │
│                                                  ▼          │
│                                           ┌─────────────┐   │
│                                           │ Features    │   │
│                                           └──────┬──────┘   │
│                                                  │          │
│                                                  ▼          │
│                    ┌────────────────────────────────────┐   │
│                    │ Training / Evaluation               │   │
│                    │ scikit-learn + Ray where useful     │   │
│                    └────────────────┬───────────────────┘   │
│                                     │                       │
│                                     ▼                       │
│                              ┌─────────────┐                │
│                              │   MLflow    │                │
│                              │ Tracking +  │                │
│                              │  Registry   │                │
│                              └──────┬──────┘                │
│                                     │                       │
│                                     ▼                       │
│                              ┌─────────────┐                │
│                              │ FastAPI /   │                │
│                              │ Ray Serve   │                │
│                              └──────┬──────┘                │
│                                     │                       │
│                    ┌────────────────┼─────────────────┐     │
│                    ▼                ▼                 ▼     │
│               Predictions       Prometheus        Evidently │
│                                   │                 │       │
│                                   ▼                 ▼       │
│                                Grafana          ML reports  │
└─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                             Public Website
                            (later phase)
```

Airflow will orchestrate the lifecycle. MLflow will provide experiment tracking and model lifecycle management. The overall pipeline follows Google's description of ML pipelines as automated workflows for developing, testing and deploying models over time.

---

# 6. Data Design

## 6.1 Source Dataset

We will use the **Criteo Click Logs** dataset.

The source contains:

* 24 chronological data files;
* 1 target/label;
* 13 numerical/integer features;
* 26 categorical features;
* missing values;
* anonymized categorical values;
* chronologically ordered examples.

This makes the dataset appropriate for demonstrating realistic data engineering without requiring a complex ML model.

## 6.2 Working Dataset

We will **not download or process the entire source dataset**.

Instead, the ingestion layer will create a small, reproducible working dataset.

Initial target:

**~500,000 examples**

The sampler will record:

* source version/location;
* source files/days used;
* sampling algorithm;
* random seed;
* requested sample size;
* actual sample size;
* sampling metadata;
* checksums.

The resulting dataset will be stored in Parquet and versioned.

The sampling procedure itself is part of the data pipeline and therefore must be reproducible.

## 6.3 Temporal Integrity

Although the working dataset is small, it must retain temporal ordering.

We will not simply shuffle 500k examples and call a random train/test split representative of deployment.

The intended structure is:

```text
Earlier period
      │
      ├── Training
      │
      ├── Validation
      │
      └── Future holdout / test
```

This directly supports Google's guidance to evaluate models on data later in time than the training data.

---

# 7. Feature Engineering

The first feature pipeline will intentionally remain modest.

It will demonstrate:

* missing-value handling;
* numerical transformations where justified;
* categorical encoding;
* train-only fitting of transformations;
* deterministic transformation;
* feature schema generation;
* feature metadata.

The feature transformation must be usable in both:

```text
training → feature transformation
serving  → same feature transformation
```

rather than maintaining two independent implementations.

This is specifically intended to reduce training-serving skew.

---

# 8. Machine Learning

## 8.1 First Model

**Logistic regression**

Why:

* mathematically interpretable;
* computationally inexpensive;
* strong baseline for binary classification;
* exposes data/feature problems clearly;
* straightforward to serialize and serve;
* appropriate for learning classification thoroughly;
* consistent with Google's recommendation to keep the first model simple and focus on infrastructure.

## 8.2 Evaluation

At minimum we will measure:

* ROC-AUC;
* PR-AUC;
* log loss;
* accuracy only as a secondary descriptive metric;
* precision/recall at selected thresholds;
* calibration;
* performance across relevant data slices;
* train vs validation vs future-holdout behavior.

The primary model objective will remain deliberately simple.

---

# 9. Experiment Management

Every meaningful training run should produce:

```text
run_id
dataset_version
feature_version
code_commit
hyperparameters
training configuration
evaluation metrics
model artifact
environment information
```

MLflow will be responsible for experiment tracking and model registration.

The model registry becomes the boundary between:

```text
candidate model
      ↓
validated model
      ↓
deployable model
```

The goal is to make the phrase **“Which model is currently being served, and how was it produced?”** answerable from metadata.

---

# 10. Pipeline Orchestration

Airflow will manage the logical lifecycle:

```text
ingest
  ↓
validate
  ↓
prepare
  ↓
feature_engineer
  ↓
train
  ↓
evaluate
  ↓
quality_gate
  ↓
register
  ↓
deploy
```

A production-like pipeline must be capable of stopping itself when an upstream condition is invalid.

Examples:

```text
bad schema       → stop
invalid labels   → stop
missing features → stop
model quality ↓  → don't promote
deployment fail  → alert
```

Google explicitly recommends preventing poor models from being released and monitoring deployment/inference systems.

---

# 11. Testing Strategy

Testing is **not a phase**. It spans the entire architecture.

## 11.1 Unit Tests

Test:

* parsing;
* transformations;
* feature calculations;
* validation rules;
* metric calculations;
* utility functions;
* configuration handling.

## 11.2 Data Tests

Test:

* schema;
* types;
* missingness;
* valid label values;
* expected cardinality;
* impossible values;
* duplicate behavior;
* data freshness;
* leakage-oriented conditions.

## 11.3 ML Tests

Test:

* deterministic training where expected;
* reproducibility;
* model artifact creation;
* minimum quality thresholds;
* metric calculations;
* regression against baseline;
* calibration behavior.

## 11.4 Integration Tests

Test:

```text
data → validation → features → model
```

and:

```text
model artifact → inference service
```

## 11.5 Training/Serving Consistency Tests

The same example should pass through the training and inference feature paths and produce equivalent representations/predictions within defined tolerances.

This directly addresses Google's Rule #37 on measuring training-serving skew.

## 11.6 System Tests

Test:

* API contract;
* malformed requests;
* service failure;
* model loading failure;
* monitoring instrumentation;
* metrics emission;
* pipeline retry behavior.

---

# 12. Observability

Observability will be treated as a first-class system capability.

## System Metrics

Prometheus will capture metrics such as:

* request rate;
* request latency;
* error rate;
* model loading failures;
* pipeline execution outcomes;
* task duration.

## ML/Data Metrics

We will expose:

* prediction distribution;
* feature distributions;
* missingness;
* data drift;
* model evaluation metrics;
* training/holdout performance;
* future-holdout performance;
* model version.

Evidently will be used for ML/data monitoring, while Grafana will provide operational dashboards.

Google emphasizes monitoring both inference infrastructure and training-serving skew.

---

# 13. Failure Injection and Debugging

A defining feature of this project is deliberate failure.

We will intentionally introduce failures such as:

```text
schema change
missing feature
incorrect categorical handling
training-serving transformation mismatch
data leakage
threshold bug
bad model promotion
feature distribution shift
broken inference dependency
monitoring failure
```

The objective is not merely to fix the bug.

For each failure we want to establish:

```text
symptom
→ observed metric/log
→ hypothesis
→ investigation
→ root cause
→ fix
→ regression test
```

This converts the project into training for **ML debugging**, not just ML deployment.

---

# 14. Runtime and Resource Strategy

## Fedora

Fedora is the persistent environment.

It owns:

* repository checkout;
* Docker infrastructure;
* datasets;
* MinIO;
* MLflow;
* PostgreSQL;
* Airflow;
* monitoring;
* inference;
* logs;
* persistent model artifacts.

## Colab

Colab is an ephemeral compute worker.

It is used when:

* training becomes expensive;
* GPU is useful;
* experimentation exceeds local resources.

Colab must not become the system of record.

Persistent state remains on Fedora.

## Resource Policy

We will not run every subsystem continuously.

Docker Compose profiles will allow:

```text
development profile
training profile
serving profile
observability profile
```

to be started selectively.

This is necessary because the machine has 16 GB RAM.

---

# 15. Technology Stack

| Area                | Technology              |
| ------------------- | ----------------------- |
| Language            | Python                  |
| Environment         | uv + `pyproject.toml`   |
| Version control     | Git + GitHub            |
| Data versioning     | DVC                     |
| Object storage      | MinIO                   |
| Data format         | Parquet                 |
| Data processing     | PyArrow + DuckDB        |
| Feature/ML pipeline | scikit-learn            |
| Distributed compute | Ray                     |
| Orchestration       | Apache Airflow          |
| Experiment tracking | MLflow                  |
| Model registry      | MLflow                  |
| API                 | FastAPI                 |
| Serving             | Ray Serve               |
| Testing             | pytest                  |
| Lint/format         | Ruff                    |
| Type checking       | mypy                    |
| Hooks               | pre-commit              |
| Containers          | Docker + Docker Compose |
| Metrics             | Prometheus              |
| Dashboard           | Grafana                 |
| ML monitoring       | Evidently               |
| CI/CD               | GitHub Actions          |
| Public project site | GitHub Pages            |
| Cloud mapping       | GCP + Terraform         |

The stack is deliberately opinionated. Components are included because they teach or demonstrate a useful production concern; we are not attempting to maximize the number of technologies.

Made With ML provides a useful implementation foundation around MLOps concerns including testing, experiment tracking, serving, orchestration, CI/CD and monitoring; this project extends that baseline where our reliability/debugging goals require it.

---

# 16. GCP Mapping

GCP will initially be a **deployment target model**, not a runtime dependency.

| Local              | GCP conceptual counterpart                           |
| ------------------ | ---------------------------------------------------- |
| MinIO              | Google Cloud Storage                                 |
| DuckDB/local data  | BigQuery / data processing services                  |
| Airflow            | Managed Airflow / Cloud Composer                     |
| Ray                | managed/external compute or GKE                      |
| MLflow             | Vertex AI / MLflow-compatible architecture           |
| FastAPI/Ray Serve  | Cloud Run / GKE                                      |
| Prometheus/Grafana | Cloud Monitoring / managed observability equivalents |
| GitHub Actions     | Cloud Build / GitHub Actions                         |
| Docker             | Artifact Registry + container runtime                |
| Terraform          | Terraform-managed GCP infrastructure                 |

We will document these mappings without claiming actual GCP deployment.

---

# 17. Security and Privacy

This project does not initially involve sensitive user data beyond the public benchmark dataset.

Nevertheless, the system will practice production habits:

* secrets never committed to Git;
* configuration separated from code;
* least-privilege mindset;
* no public write access;
* inference endpoint treated as untrusted input;
* input validation on APIs;
* public website contains only deliberately exposed metrics;
* local infrastructure remains private by default.

The eventual public metrics mechanism must never expose:

* credentials;
* internal service configuration;
* raw training data;
* private logs;
* machine filesystem information;
* arbitrary internal API access.

---

# 18. Public Demonstration

The public website is a **late-stage presentation layer**, not part of the core ML pipeline.

It will eventually show selected information such as:

```text
Current model version
Latest evaluation
Pipeline status
Inference latency
Request count
Error rate
Data-quality status
Drift status
Recent experiment results
Architecture
Testing coverage
Failure/debugging demonstrations
```

The website should consume **published, deliberately selected metrics**, rather than exposing the entire Fedora machine.

The first objective is simply to prove that metrics generated by the Fedora-hosted system can reach a public presentation layer.

---

# 19. Alternatives Considered

## Criteo vs another tabular dataset

**Chosen: Criteo.**

The goal is not domain expertise. The dataset is valuable because it is genuinely industrial in origin, chronological, categorical-heavy, incomplete, and large enough to force realistic data engineering.

A simpler benchmark would reduce the engineering pressure we specifically want.

## Pandas-only vs PyArrow/DuckDB

**Chosen: PyArrow + DuckDB.**

Pandas remains available for convenient analysis, but the canonical data path should avoid assuming that the entire dataset fits comfortably in memory.

## Kubernetes vs Docker Compose

**Chosen: Docker Compose.**

Kubernetes would add substantial operational complexity relative to the learning value on a 16 GB machine.

The architecture will remain container-oriented so that Kubernetes/GKE can become a later deployment target.

## Managed cloud vs local

**Chosen: local Fedora + optional Colab.**

This preserves persistence, eliminates cloud cost, gives us direct control over the environment, and still permits cloud-oriented architectural reasoning.

## Complex first model vs logistic regression

**Chosen: logistic regression.**

Model complexity is intentionally deferred until the infrastructure has earned our trust.

This directly follows Google's recommendation to keep the first model simple and get infrastructure right.

---

# 20. Rollout Strategy

We will not build the entire system sequentially and discover at the end that the components do not connect.

Development will happen through **vertical slices**.

### Slice 1 — Minimum ML System

```text
data
→ features
→ logistic regression
→ evaluation
```

### Slice 2 — Reproducible ML

```text
versioned data
→ validated data
→ features
→ training
→ MLflow
```

### Slice 3 — Deployable ML

```text
registry
→ inference service
→ API tests
```

### Slice 4 — Observable ML

```text
inference
→ Prometheus
→ Grafana
→ Evidently
```

### Slice 5 — Reliable ML

```text
failure
→ detection
→ diagnosis
→ fix
→ regression test
```

### Slice 6 — Automated ML

```text
GitHub
→ CI
→ Airflow
→ training/evaluation
→ quality gate
→ registry
```

### Slice 7 — Public Demonstration

```text
Fedora metrics
→ safe publication layer
→ website
```

Each slice must work before the following slice is considered complete.

---

# 21. Hackathon Execution Model

Project #1 is intentionally executed as a time-boxed hackathon.

Each milestone has:

```text
Objective
Deliverables
Definition of Done
Tests
Timebox
Evidence
```

The governing principle is:

> **Perfection is not the goal; a complete, defensible, working system is.**

When a decision is sufficiently good, we choose it and move.

The architecture is allowed to evolve later through explicit design decisions rather than endless up-front optimization.

---

# 22. Future Reuse

This project is the **reference implementation** for the broader ML engineering program.

Future projects should reuse the same platform and lifecycle:

```text
Project 1: Criteo + Logistic Regression
Project 2: Another classical ML problem
Project 3: Recommendation
Project 4: Clustering
Project 5: Deep Learning
Project 6+: LLM / small-model fine-tuning
```

The purpose is to distinguish:

**stable ML engineering structure**

from

**model/problem-specific variation**.

For later systems we expect the following to change:

* data representation;
* feature/embedding strategy;
* training algorithm;
* evaluation methodology;
* serving requirements;
* monitoring requirements.

The following should remain conceptually stable:

* versioning;
* reproducibility;
* testing;
* data validation;
* experiment tracking;
* model lifecycle;
* deployment;
* observability;
* failure handling;
* CI/CD;
* documentation;
* operational feedback loops.

---

# 23. Open Questions

These remain intentionally unresolved until implementation forces a decision:

1. Exact Criteo source-access mechanism and sampling implementation.
2. Exact categorical representation for logistic regression.
3. Whether Ray is useful for Project #1 training or should initially remain a platform capability.
4. Exact Airflow deployment footprint on 16 GB RAM.
5. Safe mechanism for publishing selected Fedora metrics to the public website.
6. Exact model-promotion policy and quality thresholds.
7. Which metrics should be considered system SLOs versus diagnostic metrics.
8. Which components should be permanently running versus profile-based.

These are implementation decisions, not reasons to delay starting the project.

---

# 24. Design Review Checklist

Before implementation begins, the team should be able to answer “yes” to:

* Do we know what the system is supposed to do?
* Do we have a simple measurable ML objective?
* Is the dataset acquisition reproducible?
* Is the train/validation/test strategy defined?
* Can training and serving share feature logic?
* Can bad data stop the pipeline?
* Can bad models stop promotion?
* Can we identify exactly which model is serving?
* Can we observe inference?
* Can we observe data/model behavior?
* Can we test infrastructure independently of the model?
* Can we deliberately break the system and diagnose the failure?
* Can the architecture be mapped onto GCP?
* Can the entire project run without paid cloud infrastructure?
* Can the core platform be reused for the next ML problem?

---

# 25. Decision

**Proceed with implementation using this design as v0.1.**

The first implementation target is **not the full stack**.

The first engineering milestone is to turn this design into the **repository/project skeleton and implementation backlog**, then build the first vertical slice:

> **Criteo sample → validated dataset → feature pipeline → logistic regression → evaluation**

Everything else will be layered onto a working system rather than built in the abstract.
