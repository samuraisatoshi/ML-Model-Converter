#!/usr/bin/env python3
"""
ML Model Converter - Web Interface Launcher
===========================================

Easy launcher for the Streamlit web interface.

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
    """Launch the Streamlit web interface."""
    print("🚀 Starting ML Model Converter Web Interface...")
    print("=" * 50)
    
    # Check if we're in the right directory
    src_path = Path(__file__).parent / "src" / "web" / "app.py"
    if not src_path.exists():
        print("❌ Error: Could not find the web application.")
        print(f"   Expected: {src_path}")
        print("   Make sure you're running this from the project root directory.")
        sys.exit(1)
    
    try:
        # Launch Streamlit
        print("🌐 Starting Streamlit server...")
        print("📱 The web interface will open in your browser automatically.")
        print("🔗 If it doesn't open, go to: http://localhost:8501")
        print("")
        print("⏹️  Press Ctrl+C to stop the server")
        print("=" * 50)
        
        # Run streamlit with cleaner UI options
        cmd = [
            sys.executable, "-m", "streamlit", "run", str(src_path),
            "--server.address", "localhost",
            "--server.port", "8501",
            "--server.headless", "true",
            "--server.runOnSave", "false",
            "--client.showErrorDetails", "false",
            "--browser.gatherUsageStats", "false"
        ]
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n⏹️  Server stopped by user")
    except FileNotFoundError:
        print("❌ Error: Streamlit is not installed.")
        print("   Install it with: pip install streamlit")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting web interface: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()