# Changelog

All notable changes to **scholar-rag-agent** are documented here.
Follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [v0.1.20] — 2024-04-09

### Added
- Extended arXiv module with improved error handling
- Added structured logging for graph operations
- New unit tests covering edge cases in retrieval pipeline

### Changed
- Refactored retry logic to use exponential backoff with jitter
- Improved type annotations across core modules
- Updated dependency pins to latest stable versions

### Fixed
- Resolved race condition in async arXiv handler
- Fixed incorrect graph timeout calculation

## [v0.1.0] — 2024-03-05

### Added
- Initial project scaffold with scientific RAG core
- Basic agent implementation
- README and setup documentation
