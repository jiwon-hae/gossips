#!/usr/bin/env python3
"""
Test runner script for the chunker module.

This script provides convenient ways to run tests with different configurations
and generates coverage reports.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and handle errors."""
    print(f"{'='*60}")
    if description:
        print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description or 'Command'} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description or 'Command'} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {cmd[0]}")
        print("Please ensure pytest is installed: pip install pytest pytest-asyncio pytest-cov")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run tests for the chunker module")
    parser.add_argument(
        "--test-type", 
        choices=["unit", "integration", "all"], 
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true", 
        help="Generate coverage report"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Verbose output"
    )
    parser.add_argument(
        "--fast", 
        action="store_true", 
        help="Skip slow tests"
    )
    parser.add_argument(
        "--specific", 
        help="Run specific test file or test function"
    )
    parser.add_argument(
        "--html-coverage", 
        action="store_true", 
        help="Generate HTML coverage report"
    )
    
    args = parser.parse_args()
    
    # Ensure we're in the right directory
    project_root = Path(__file__).parent
    original_dir = Path.cwd()
    
    try:
        # Change to project root
        import os
        os.chdir(project_root)
        
        # Base pytest command
        pytest_cmd = ["python", "-m", "pytest"]
        
        # Add verbosity
        if args.verbose:
            pytest_cmd.append("-v")
        else:
            pytest_cmd.append("-q")
        
        # Add coverage if requested
        if args.coverage or args.html_coverage:
            pytest_cmd.extend([
                "--cov=ingestion.chunker",
                "--cov-report=term-missing"
            ])
            
            if args.html_coverage:
                pytest_cmd.append("--cov-report=html:htmlcov")
        
        # Handle test type selection
        if args.specific:
            pytest_cmd.append(args.specific)
        elif args.test_type == "unit":
            pytest_cmd.extend([
                "-m", "unit or not integration",
                "tests/ingestion/chunker/"
            ])
        elif args.test_type == "integration":
            pytest_cmd.extend([
                "-m", "integration",
                "tests/ingestion/chunker/"
            ])
        else:  # all tests
            pytest_cmd.append("tests/ingestion/chunker/")
        
        # Skip slow tests if requested
        if args.fast:
            pytest_cmd.extend(["-m", "not slow"])
        
        # Run the tests
        success = run_command(pytest_cmd, "Running tests")
        
        if success:
            print(f"\n{'='*60}")
            print("🎉 All tests completed successfully!")
            
            if args.html_coverage:
                print(f"📊 HTML coverage report generated in: {project_root}/htmlcov/index.html")
            
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print("❌ Some tests failed. Check the output above for details.")
            print(f"{'='*60}")
            sys.exit(1)
            
    finally:
        # Return to original directory
        os.chdir(original_dir)


def check_dependencies():
    """Check if required testing dependencies are installed."""
    required_packages = [
        "pytest",
        "pytest-asyncio", 
        "pytest-cov"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall them with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True


def show_test_info():
    """Show information about available tests."""
    print("Available test files:")
    print("  - test_config.py: Tests for ChunkingConfig")
    print("  - test_chunk.py: Tests for DocumentChunk")
    print("  - test_base_chunker.py: Tests for BaseChunker abstract class")
    print("  - test_simple_chunker.py: Tests for SimpleChunker")
    print("  - test_semantic_chunker.py: Tests for SemanticChunker")
    print("  - test_factory.py: Tests for create_chunker factory")
    print("  - test_integration.py: Integration tests")
    print("\nExample usage:")
    print("  python run_tests.py --test-type unit")
    print("  python run_tests.py --coverage --html-coverage")
    print("  python run_tests.py --specific tests/ingestion/chunker/test_config.py")
    print("  python run_tests.py --fast  # Skip slow tests")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--info":
        show_test_info()
        sys.exit(0)
    
    if not check_dependencies():
        sys.exit(1)
    
    main()