#!/usr/bin/env python
"""
Wrapper script to run main_stage1.py with better error handling.
"""

import traceback
import sys

try:
    import main_stage1
    import asyncio
    # main doesn't take arguments directly, it reads from sys.argv
    sys.argv = [sys.argv[0]] + sys.argv[1:] if len(sys.argv) > 1 else [sys.argv[0], "AI and human consciousness"]
    asyncio.run(main_stage1.main())
except Exception as e:
    print(f"Error running main_stage1: {e}")
    traceback.print_exc()
