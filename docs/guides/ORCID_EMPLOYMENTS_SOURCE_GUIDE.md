# ORCID Employments Source Guide

![ORCID employments discovery flow](../assets/orcid_employments_source.gif)

Use this guide when wiring researcher employment affiliations into
**scholar-rag-agent**. The connector performs deterministic public ORCID JSON
requests; optional downstream synthesis can use GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2.

## Why ORCID employments

ORCID works describe author-curated research outputs. The separate employments
section describes institutional affiliations, roles, departments, dates, and
organization identifiers. These records support researcher and institution
disambiguation without treating an affiliation as a publication.

## Usage

```python
from ingestion.orcid_employments import OrcidEmploymentsConnector

docs = await OrcidEmploymentsConnector().search(
    "0000-0002-1825-0097",
    max_results=5,
)
```

Bare or URL ORCID iDs fetch `GET /v3.0/{orcid}/employments` directly. Researcher
name queries first use `expanded-search`, then fetch public employment summaries
for matching profiles.

## What you get

| Field | Source |
|---|---|
| `title` | Role and organization |
| `text` | Researcher, role, department, organization, location, dates, and identifier |
| `metadata.source_type` | `orcid_employments` |
| `metadata.orcid` | Researcher's ORCID iD |
| `metadata.profile_name` | Expanded-search display name or ORCID iD |
| `metadata.put_code` | Employment summary put-code |
| `metadata.role_title` | `role-title` |
| `metadata.department` | `department-name` |
| `metadata.organization` | Organization name |
| `metadata.organization_id` | Disambiguated organization identifier, often ROR |
| `metadata.start_date` / `end_date` | Available ORCID partial dates |

## Safety notes

- Public ORCID data only; no member token or write endpoint is used.
- Employment entries are researcher/member assertions and can be incomplete or
  outdated; preserve their ORCID provenance.
- Blank input and non-positive limits do not issue HTTP requests.
- Private, malformed, or unavailable records return an empty list.

## Suggested repo metadata

- **Description:** Agentic scholarly RAG with multi-source ingestion, hybrid retrieval, and GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 reasoning.
- **Topics:** `rag`, `scholarly`, `orcid`, `employment`, `affiliations`, `ror`, `researchers`, `llm`, `python`
