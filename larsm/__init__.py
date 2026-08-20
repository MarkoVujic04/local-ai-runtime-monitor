"""Local AI Runtime Security Monitor (LARSM).

Defensive tooling for the model runtime layer of local LLM setups
(Ollama, LM Studio). This package never imports, unpickles, or executes
the files it inspects, it only reads and parses bytes.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]