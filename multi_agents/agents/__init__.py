"""
Agents package initializer.

Imports and instantiates all Tool subclasses, triggering their
self-registration into the Register singleton at import time.

To add a new agent to the system, import and instantiate it here —
the Supervisor and Graph will discover it automatically.
"""

from .directory import Directory

agents_list = [
    Directory(),
]
