# Changelog

All notable changes to **scholar-rag-agent** are documented here.
Follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [v0.5.8] — 2025-01-24

### Added
- Extended graph module with improved error handling
- Added structured logging for retrieval operations
- New unit tests covering edge cases in grounding pipeline

### Changed
- Refactored retry logic to use exponential backoff with jitter
- Improved type annotations across core modules
- Updated dependency pins to latest stable versions

### Fixed
- Resolved race condition in async graph handler
- Fixed incorrect retrieval timeout calculation

## [v0.1.0] — 2024-12-20

### Added
- Initial project scaffold with scientific RAG core
- Basic agent implementation
- README and setup documentation
