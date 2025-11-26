"""
Term expansion utilities for natural language search.

Provides functions for expanding natural language descriptions into
prioritized search terms using a configurable expansion dictionary.
"""

import json
import os
from pathlib import Path
from typing import Optional

from security import sanitize_search_term

# Module-level cache
_term_expansions_cache: Optional[dict[str, list[str]]] = None
_default_config_path = Path(__file__).parent / "term_expansions.json"


def load_term_expansions(config_path: Optional[str] = None) -> dict[str, list[str]]:
    """Load term expansions from JSON config file (cached)."""
    global _term_expansions_cache
    
    if _term_expansions_cache is not None and config_path is None:
        return _term_expansions_cache
    
    if config_path is None:
        config_path = str(_default_config_path)
    else:
        config_path = os.path.abspath(config_path)
        project_root = Path(__file__).parent.resolve()
        if not config_path.startswith(str(project_root)):
            raise ValueError(f"Config path must be within project directory: {config_path}")
    
    if os.path.exists(config_path) and os.path.isfile(config_path):
        try:
            with open(config_path, "r") as f:
                expansions = json.load(f)
                if not isinstance(expansions, dict):
                    raise ValueError("Term expansions file must contain a JSON object")
                if config_path == str(_default_config_path):
                    _term_expansions_cache = expansions
                return expansions
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in term expansions file: {e}")
        except Exception as e:
            raise ValueError(f"Error loading term expansions file: {e}")
    
    return {}


def expand_search_terms(
    description: str, 
    term_expansions: dict[str, list[str]], 
    max_terms: int = 50
) -> list[str]:
    """Expand natural language description into prioritized search terms."""
    if not description or not isinstance(description, str):
        return []
    
    if len(description) > 500:
        description = description[:500]
    
    words = description.lower().replace("_", " ").replace("-", " ").split()
    priority_terms = set(words)
    
    # Direct expansions
    direct_expansions = set()
    for word in words:
        if word in term_expansions:
            direct_expansions.update(term_expansions[word])
    
    # Transitive expansions
    transitive_expansions = set()
    for word in words:
        for key, expansions in term_expansions.items():
            if word in expansions or any(word in exp for exp in expansions):
                transitive_expansions.update(expansions)
    
    # Combine and prioritize
    all_terms = (
        list(priority_terms) + 
        list(direct_expansions - priority_terms) + 
        list(transitive_expansions - direct_expansions - priority_terms)
    )
    
    # Filter and sanitize
    filtered_terms = [term for term in all_terms if len(term) >= 2 or term in ["id", "pk"]]
    sanitized_terms = [sanitize_search_term(term) for term in filtered_terms]
    sanitized_terms = [term for term in sanitized_terms if term]
    
    return sanitized_terms[:max_terms]

