#!/usr/bin/env python3
"""
ML Model Converter - CLI Launcher
=================================

Easy launcher for the command-line interface.

Author: SamuraiSatoshi (in collaboration with Claude)
License: CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
Repository: https://github.com/samuraisatoshi/ML-Model-Converter

Copyright (c) 2024 SamuraiSatoshi
This work is licensed under Creative Commons Attribution-NonCommercial 4.0 International.
For commercial licensing, contact: samuraisatoshi@cryptoworld.8shield.net
"""

import sys
import subprocess
from pathlib import Path


def main():
    """Launch the CLI interface."""
    print("🖥️  ML Model Converter - Command Line Interface")
    print("=" * 50)
    
    # Check if we're in the right directory
    cli_path = Path(__file__).parent / "src" / "cli" / "main.py"
    if not cli_path.exists():
        print("❌ Error: Could not find the CLI application.")
        print(f"   Expected: {cli_path}")
        print("   Make sure you're running this from the project root directory.")
        sys.exit(1)
    
    try:
        # Get command line arguments (skip the script name)
        args = sys.argv[1:]
        
        if not args:
            # Show help if no arguments provided
            args = ["--help"]
        
        # Build command
        cmd = [sys.executable, str(cli_path)] + args
        
        # Run CLI
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
        
    except FileNotFoundError as e:
        if "click" in str(e):
            print("❌ Error: Click library is not installed.")
            print("   Install it with: pip install click")
        else:
            print(f"❌ Error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error launching CLI: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()