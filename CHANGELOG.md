# Changelog

All notable changes to **scholar-rag-agent** are documented here.
Follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [v0.2.10] — 2025-09-05

### Added
- Extended embedding module with improved error handling
- Added structured logging for ingestion operations
- New unit tests covering edge cases in graph pipeline

### Changed
- Refactored retry logic to use exponential backoff with jitter
- Improved type annotations across core modules
- Updated dependency pins to latest stable versions

### Fixed
- Resolved race condition in async embedding handler
- Fixed incorrect ingestion timeout calculation

## [v0.1.0] — 2025-08-08

### Added
- Initial project scaffold with scientific RAG core
- Basic agent implementation
- README and setup documentation
