#!/usr/bin/env python3
"""Health check script for scholar-rag-agent."""
import sys
import requests

BASE_URL = "http://localhost:8000"

def check_health() -> bool:
    """Check if the service is healthy."""
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            print(f"✓ Service healthy: {data}")
            return True
        print(f"✗ Unhealthy response: {resp.status_code}")
        return False
    except requests.ConnectionError:
        print("✗ Service not reachable")
        return False

if __name__ == "__main__":
    sys.exit(0 if check_health() else 1)
