# Contributing Guidelines

Thank you for your interest in contributing to the **6G Smart Factory Network & Manufacturing Intelligence Platform**.

## Code of Conduct & Principles
- **Scientific Integrity**: All empirical metrics, statistical thresholds, and machine learning evaluations must be derived strictly from reproducible code execution.
- **Safety & Leakage Prevention**: Never introduce outcome-side target variables (such as contemporaneous speed or error rate) into diagnostic models where labels are deterministically derived from them.

---

## Branching Strategy & Workflow
We follow a simplified GitHub Flow model:
1. `main`: Production-ready branch. All commits must pass verification and maintain working cloud dashboard compatibility.
2. Feature Branches: Create dedicated descriptive branches for all modifications:
   - `feat/feature-name` for new modeling, analytics, or UI modules.
   - `fix/bug-description` for bug repairs or numerical corrections.
   - `docs/topic-name` for manuscript, README, and documentation enhancements.

---

## Commit Message Conventions
Commits strictly follow the [Conventional Commits specification](https://www.conventionalcommits.org/):

| Type | Description |
| :--- | :--- |
| `feat:` | A new analytical module, dashboard page, or model architecture. |
| `fix:` | A bug fix or patch in pipeline execution or dashboard rendering. |
| `docs:` | Documentation changes only (e.g., README, research paper, docstrings). |
| `chore:` | Routine maintenance, dependencies, repository configurations, or `.gitignore`. |
| `refactor:` | Code restructuring without altering external functionality or metrics. |
| `perf:` | Performance optimizations reducing memory overhead or execution latency. |

### Example Commit Messages:
```bash
feat: add segmented breakpoint regression with block bootstrap CI
fix: correct fallback path resolution for cloud container execution
docs: add LICENSE and fix README clone URL
chore: pin exact package versions in requirements.txt
```

---

## Python Code Standards
- **Style**: Follow PEP 8 guidelines.
- **Docstrings**: Every module must begin with a descriptive module docstring. All public functions must include full docstrings detailing `Parameters`/`Args` and `Returns` with explicit type annotations.
- **Dependencies**: Any new dependency must be added to `requirements.txt` with exact version pinning (`package==x.y.z`).
- **Path Resolution**: Never hardcode local system paths (`C:\Users\...`). Always resolve file paths relative to `os.path.dirname(__file__)` or repository root.

---

## Submitting Pull Requests
1. Fork or branch from `main`.
2. Ensure `requirements.txt` installs cleanly in a fresh Python 3.10+ virtual environment.
3. Verify that the Streamlit dashboard launches without errors:
   ```bash
   streamlit run app/dashboard.py
   ```
4. Submit a Pull Request detailing the changes, empirical motivation, and validation results.
