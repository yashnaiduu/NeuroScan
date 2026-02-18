#!/usr/bin/env python3
"""
Standalone health check script for container orchestration.
Returns exit code 0 if healthy, 1 if unhealthy.
"""
import sys
import os
import requests
from typing import Dict, Any


def check_health() -> Dict[str, Any]:
    """Check application health."""
    port = os.getenv('PORT', '5050')
    url = f'http://localhost:{port}/health'
    
    try:
        response = requests.get(url, timeout=5)
        
        if response.status_code != 200:
            return {
                'healthy': False,
                'reason': f'HTTP {response.status_code}',
                'details': response.text
            }
        
        data = response.json()
        
        # Check critical components
        if not data.get('model_loaded'):
            return {
                'healthy': False,
                'reason': 'Model not loaded',
                'details': data
            }
        
        if data.get('status') != 'healthy':
            return {
                'healthy': False,
                'reason': f"Status: {data.get('status')}",
                'details': data
            }
        
        return {
            'healthy': True,
            'reason': 'All checks passed',
            'details': data
        }
    
    except requests.exceptions.Timeout:
        return {
            'healthy': False,
            'reason': 'Request timeout',
            'details': None
        }
    except requests.exceptions.ConnectionError:
        return {
            'healthy': False,
            'reason': 'Connection refused',
            'details': None
        }
    except Exception as e:
        return {
            'healthy': False,
            'reason': f'Error: {str(e)}',
            'details': None
        }


def main():
    """Main entry point."""
    result = check_health()
    
    if result['healthy']:
        print(f"✓ Health check passed: {result['reason']}")
        sys.exit(0)
    else:
        print(f"✗ Health check failed: {result['reason']}", file=sys.stderr)
        if result['details']:
            print(f"Details: {result['details']}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
