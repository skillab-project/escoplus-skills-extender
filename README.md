# ESCOPlus Skills Extender Back-End

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/skillab-project/escoplus-skills-extender)

## Description

This project implements the backend API for the **ESCOPlus Skills Extender**, an open-source framework designed to enhance the ESCO taxonomy by identifying and proposing new skill extensions. It is built with FastAPI (Python) and exposes endpoints for:

- Fetching job postings, profiles, courses, and law/policy documents from the SkillLab Tracker API.
- Resolving ESCO skill URIs to human-readable labels via batch API calls.
- Matching ESCO skills against an extended non-ESCO skill pool using TF-IDF cosine similarity.
- Proposing high-confidence ESCO taxonomy extensions sourced from three pools: a technology skills CSV, an AI skills list, and a green/sustainability skills list.
- Building a skill co-occurrence network (ESCO ↔ non-ESCO) with NetworkX and computing explainability metrics.
- Caching completed analyses to disk and preventing duplicate concurrent runs via a lock/state mechanism.
- Exporting proposed ESCO+ extensions to CSV files.

The service is part of the broader [SkillLab](https://github.com/skillab-project) project, which analyses the European labour market using Open Job Advertisements (OJA) data.

---

## Getting Started Guide

### Prerequisites

- **Git:** Installed on your system. ([Download Git](https://git-scm.com/downloads))
- **Python:** Version 3.11 or newer is recommended. ([Download Python](https://www.python.org/downloads/))
- **Access to the SkillLab Tracker API:** Valid credentials (`TRACKER_USERNAME` and `TRACKER_PASSWORD`) for `https://skillab-tracker.csd.auth.gr/api`.

---

### Installation Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/skillab-project/escoplus-skills-extender.git
   cd escoplus-skills-extender
   ```

2. **Create and Activate a Virtual Environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate      # Windows
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**

   Create a `.env` file in the project root:

   ```env
   TRACKER_API=https://skillab-tracker.csd.auth.gr/api
   TRACKER_USERNAME=your_username
   TRACKER_PASSWORD=your_password
   ```

   > **Note:** The application validates these variables at startup and will raise a `RuntimeError` if any are missing.

5. **Technology Skills CSV (Optional):**

   The service can optionally load a local `technology_skills.csv` file as an additional non-ESCO skill source. A sample file is included in the repository. The CSV must contain an `Example` column with comma- or semicolon-separated skill entries.

---

## Running the Application

### Locally

```bash
uvicorn escoplus_skills_extender:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`. Swagger UI documentation is at `http://localhost:8000/docs`.

> When deployed behind a reverse proxy, the app is mounted under the `/escoplus-skills-extender` root path.

### With Docker

```bash
docker-compose up --build
```

Or manually:

```bash
docker build -t escoplus-skills-extender .
docker run -p 8001:8000 --env-file .env escoplus-skills-extender
```

---

## API Endpoints

All analysis endpoints are prefixed with `/api/analysis`.

### `GET /api/analysis/jobs_ultra`

Fetches all job postings (auto-paginated with retry), extends the ESCO taxonomy via TF-IDF similarity, and returns a skill network with explainability metrics. Results are cached.

| Parameter              | Type    | Default | Description                                               |
|------------------------|---------|---------|-----------------------------------------------------------|
| `keywords`             | string  | —       | Comma-separated keywords to filter jobs                   |
| `occupation_ids`       | string  | —       | Comma-separated ESCO occupation IDs                       |
| `source`               | string  | —       | Optional source filter (e.g. `linkedin`)                  |
| `min_upload_date`      | string  | —       | Filter jobs uploaded after `YYYY-MM-DD`                   |
| `max_upload_date`      | string  | —       | Filter jobs uploaded before `YYYY-MM-DD`                  |
| `similarity_threshold` | float   | `0.8`   | TF-IDF cosine similarity threshold                        |
| `confidence_threshold` | float   | `0.6`   | Confidence cutoff for proposing new ESCO+ skills          |

### `GET /api/analysis/law-policies_extend_esco`

Extends ESCO from law and policy documents (EUR-Lex or other sources).

### `GET /api/analysis/profiles_extend_esco`

Extends ESCO from professional profiles. Results cached.

### `GET /api/analysis/courses_ultra`

Extends ESCO from online courses.

---

## Non-ESCO Skill Sources

The skill extension pool is built from three sources at runtime:

- **Technology Skills CSV** (`technology_skills.csv`) — practical technology skills from a structured CSV.
- **AI Extended List** — ~250 curated AI/ML/LLM skills hardcoded in the application.
- **Green Extended List** — ~200 sustainability and green economy skills hardcoded in the application.

Each proposed extension is tagged with its source (`technology_csv`, `ai_extended`, or `green_extended`).

---

## Running the Tests

```bash
pytest tests/
```

---

## Project Structure

```
escoplus-skills-extender/
├── escoplus_skills_extender.py   # FastAPI app, analysis endpoints, skill extension logic
├── technology_skills.csv         # Optional technology skill source CSV
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container image definition
├── docker-compose.yml            # Compose configuration
├── .env                          # Environment variables (fill in before running)
├── jenkins/                      # CI/CD pipeline configuration
└── tests/                        # Test suite
```

---

## Technologies

- **Python 3.11**
- **FastAPI** — REST API framework
- **Uvicorn** — ASGI server
- **NetworkX** — Skill co-occurrence network construction and analysis
- **scikit-learn** — TF-IDF vectorisation and cosine similarity
- **pandas / NumPy** — Data processing
- **python-dotenv** — Environment variable management
- **Docker / Docker Compose** — Containerised deployment

---

## License

This project is licensed under the **Eclipse Public License 2.0 (EPL-2.0)**. See the [LICENSE](LICENSE) file for details.
