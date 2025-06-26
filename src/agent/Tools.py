
"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from typing import Any, Callable, List, Optional, cast


# define the tool for agent 


def search_web() : 
    """This tool can search web for information."""
    return "Hue Lightstrip Plus 2m - White and Color Ambiance - Smart LED Light Strip, Works with Alexa, Google Assistant, and Apple HomeKit (Compatible with Philips Hue Hub) - 1 Pack"

def query_database() :
    """This tool can query a database for information."""
    return "Database query result: Hue Lightstrip Plus 2m - White and Color Ambiance - Smart LED Light Strip, Works with Alexa, Google Assistant, and Apple HomeKit (Compatible with Philips Hue Hub) - 1 Pack"


TOOLS1: List[Callable[..., Any]] = [search_web] 
TOOLS2: List[Callable[..., Any]] = [query_database ]