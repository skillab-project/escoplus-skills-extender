# Contributing to ESCOPlus Skills Extender

Thank you for your interest in contributing! This document outlines the process for reporting issues, proposing changes, and submitting code.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Pull Request Process](#pull-request-process)

---

## Code of Conduct

This project is part of the [SkillLab](https://github.com/skillab-project) EU Horizon research initiative. All contributors are expected to engage respectfully and constructively. Harassment or discrimination of any kind will not be tolerated.

---

## Getting Started

Before contributing, please:

1. Read the [README](README.md) to understand the project's purpose and architecture.
2. Check the [open issues](https://github.com/skillab-project/escoplus-skills-extender/issues) to see if your bug or feature request has already been reported.
3. For significant changes, open an issue first to discuss your proposed approach before writing code.

---

## How to Contribute

### Reporting Bugs

Open an issue and include: a clear title, steps to reproduce, expected vs. actual behaviour, environment details (OS, Python version), and any error messages or stack traces.

### Suggesting Enhancements

Open a feature request issue describing: the problem you are solving, your proposed approach, and any alternatives you considered. Contributions to the non-ESCO skill pool lists (AI or Green) are especially welcome — see the `AI_SKILLS_EXTENDED` and `GREEN_SKILLS_EXTENDED` lists in `escoplus_skills_extender.py`.

### Submitting Code Changes

All code contributions are made via Pull Requests. The general workflow is: fork → branch → implement → test → PR.

---

## Development Setup

1. **Fork and clone your fork:**

   ```bash
   git clone https://github.com/<your-username>/escoplus-skills-extender.git
   cd escoplus-skills-extender
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your `.env` file:**

   ```env
   TRACKER_API=https://skillab-tracker.csd.auth.gr/api
   TRACKER_USERNAME=your_username
   TRACKER_PASSWORD=your_password
   ```

5. **Start the development server:**

   ```bash
   uvicorn escoplus_skills_extender:app --host 0.0.0.0 --port 8000 --reload
   ```

---

## Coding Standards

- Follow [PEP 8](https://peps.python.org/pep-0008/) for all Python code.
- Use descriptive names for variables and functions.
- Add docstrings to any new functions or classes.
- Do not commit credentials or secrets — use `.env` exclusively.
- When extending the AI or Green skill lists, add entries in lowercase and group them logically with a comment block.
- Do not modify the `technology_skills.csv` format — the parser expects an `Example` column with comma/semicolon-separated values.

---

## Testing

Run the test suite with:

```bash
pytest tests/
```

When contributing:

- Add or update tests for any new endpoint behaviour or helper function.
- Ensure all existing tests pass before opening a PR.
- For caching and lock logic changes, include tests that simulate concurrent request scenarios.

---

## Commit Message Guidelines

Use imperative-mood messages with a type prefix:

```
<type>: <short summary>
```

Common types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`.

Examples:

```
feat: add /articles_ultra endpoint for article-based ESCO extension
fix: handle null skill labels in batch URI resolution
docs: document AI_SKILLS_EXTENDED list structure in README
```

---

## Pull Request Process

1. **Branch naming:** Use descriptive names, e.g. `feat/add-profiles-network-stats` or `fix/null-label-batch-resolve`.
2. **Keep PRs focused:** One logical change per PR.
3. **Fill in the PR description:** Explain what changed, why, and how it was tested.
4. **Link related issues:** Use `Closes #<issue-number>` in the PR description.
5. **Review:** At least one maintainer review is required. Be responsive to feedback.
6. **CI:** All automated checks must pass before merging.

---

## Questions

Open a [discussion or issue](https://github.com/skillab-project/escoplus-skills-extender/issues) if you have questions not covered here.
