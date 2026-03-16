"""General-purpose web scraper tool.

Three modes (tried in order for topic-based scraping):
  1. Adapter crawl   — registered adapters for known sites
  2. Search discovery — DuckDuckGo + LLM-guided regex extraction
  3. Direct URL       — generic HTML extraction on a given URL

Usage from Agent 1:
    from tools.scraper import scrape
    values = scrape(topic="usa swift code", config=llm_config)
    values = scrape(url="https://bank-code.net/country/UNITED-STATES-%28US%29/100")
"""
from tools.scraper.runner import scrape  # noqa: F401
