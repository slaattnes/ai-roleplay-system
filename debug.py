#!/usr/bin/env python
"""
Debug script to help diagnose import issues.
"""

import traceback
import sys

try:
    print("Attempting to import google.generativeai...")
    import google.generativeai
    print(f"Successfully imported google.generativeai version: {google.generativeai.__version__}")
except ImportError as e:
    print(f"Failed to import google.generativeai: {e}")
    traceback.print_exc()

try:
    print("\nAttempting to import src.llm.gemini...")
    import src.llm.gemini
    print("Successfully imported src.llm.gemini")
except ImportError as e:
    print(f"Failed to import src.llm.gemini: {e}")
    traceback.print_exc()
