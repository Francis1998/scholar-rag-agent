# Changelog

All notable changes to **scholar-rag-agent** are documented here.
Follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [v0.3.13] — 2025-04-10

### Added
- Extended retrieval module with improved error handling
- Added structured logging for ingestion operations
- New unit tests covering edge cases in arXiv pipeline

### Changed
- Refactored retry logic to use exponential backoff with jitter
- Improved type annotations across core modules
- Updated dependency pins to latest stable versions

### Fixed
- Resolved race condition in async retrieval handler
- Fixed incorrect ingestion timeout calculation

## [v0.1.0] — 2025-03-20

### Added
- Initial project scaffold with scientific RAG core
- Basic agent implementation
- README and setup documentation
