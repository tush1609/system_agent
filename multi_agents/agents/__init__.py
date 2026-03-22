"""
Agents package initializer.

Imports and instantiates all Tool subclasses, triggering their
self-registration into the Register singleton at import time.

To add a new agent to the system, import and instantiate it here —
the Supervisor and Graph will discover it automatically.
"""

from .directory import Directory
from .generic import Generic
from .file import File

agents_list = [
    Directory(),
    File(),
    Generic()
]
