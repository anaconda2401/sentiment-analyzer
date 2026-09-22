"""
conftest.py — adds the project root to sys.path so tests can import `chatsense`
from any working directory.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
