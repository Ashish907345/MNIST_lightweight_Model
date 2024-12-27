import pytest
import sys

def main():
    # Run the architecture tests with verbose output
    exit_code = pytest.main([
        'tests/test_model_architecture.py',
        '-v'
    ])
    
    sys.exit(exit_code)

if __name__ == '__main__':
    main() 