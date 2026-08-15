# ORCID Education Source Guide

![ORCID education discovery flow](../assets/orcid_education_source.gif)

Use this guide when wiring researcher education affiliations into
**scholar-rag-agent**. The connector performs deterministic public ORCID JSON
requests; optional downstream synthesis can use GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2.

## Why ORCID education

ORCID works describe author-curated research outputs and employments capture
institutional roles. The separate educations section describes degrees,
departments, institutions, dates, and organization identifiers. These records
support researcher disambiguation without treating a degree as a publication.

## Usage

```python
from ingestion.orcid_education import OrcidEducationConnector

docs = await OrcidEducationConnector().search(
    "0000-0002-1825-0097",
    max_results=5,
)
```

Bare or URL ORCID iDs fetch `GET /v3.0/{orcid}/educations` directly. Researcher
name queries first use `expanded-search`, then fetch public education summaries
for matching profiles.

## What you get

| Field | Source |
|---|---|
| `title` | Degree and organization |
| `text` | Researcher, degree, department, organization, location, dates, and identifier |
| `metadata.source_type` | `orcid_education` |
| `metadata.orcid` | Researcher's ORCID iD |
| `metadata.profile_name` | Expanded-search display name or ORCID iD |
| `metadata.put_code` | Education summary put-code |
| `metadata.degree_title` | `role-title` (degree) |
| `metadata.department` | `department-name` |
| `metadata.organization` | Organization name |
| `metadata.organization_id` | Disambiguated organization identifier, often ROR |
| `metadata.start_date` / `end_date` | Available ORCID partial dates |

## Safety notes

- Public ORCID data only; no member token or write endpoint is used.
- Education entries are researcher/member assertions and can be incomplete or
  outdated; preserve their ORCID provenance.
- Blank input and non-positive limits do not issue HTTP requests.
- Private, malformed, or unavailable records return an empty list.

## Suggested repo metadata

- **Description:** Agentic scholarly RAG with multi-source ingestion, hybrid retrieval, and GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 reasoning.
- **Topics:** `rag`, `scholarly`, `orcid`, `education`, `affiliations`, `ror`, `researchers`, `llm`, `python`
