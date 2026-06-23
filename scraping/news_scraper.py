"""
News Scraper - CommonCrawl Pipeline Simulation.

This module simulates the CommonCrawl data acquisition pipeline at academic scale.
CommonCrawl stores petabytes of raw web pages in WARC format. Instead of processing
100+ TB, this module replicates the workflow:

    1. Discover URLs (via RSS feeds instead of WARC index)
    2. Download raw HTML (via HTTP requests instead of S3)
    3. Save raw HTML with metadata (simulating WARC records)
    4. Extract plain text (simulating WET files extraction)

Data sources:
    - RSS feeds from economic news outlets (BBC Business, Reuters, etc.)
    - Fallback to static HTML pages if RSS is unavailable

The output is raw HTML files stored in data/external/raw_html/ with metadata JSON,
ready for the text_processor.py pipeline (HTML cleaning, tokenization, TF-IDF).

Usage:
    python -m scraping.news_scraper

Output:
    data/external/raw_html/{source}_{date}_{hash}.html
    data/external/scraping_metadata.json
"""
import os
import sys
import json
import logging
import hashlib
import time
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse

import requests
import feedparser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Output directories
EXTERNAL_DIR = Path("data/external")
RAW_HTML_DIR = EXTERNAL_DIR / "raw_html"
METADATA_FILE = EXTERNAL_DIR / "scraping_metadata.json"

# RSS feeds for economic news (English and Spanish sources)
# These serve as the "URL discovery" phase, analogous to CommonCrawl's WARC index
RSS_FEEDS = {
    "bbc_business": {
        "url": "http://feeds.bbci.co.uk/news/business/rss.xml",
        "language": "en",
        "category": "business",
    },
    "reuters_business": {
        "url": "https://www.reutersagency.com/feed/?taxonomy=markets&post_type=reuters-best",
        "language": "en",
        "category": "business",
    },
    "el_pais_economia": {
        "url": "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/section/economia/portada",
        "language": "es",
        "category": "economia",
    },
    "cnn_money": {
        "url": "http://rss.cnn.com/rss/money_news_international.rss",
        "language": "en",
        "category": "business",
    },
    "ft_world": {
        "url": "https://www.ft.com/world?format=rss",
        "language": "en",
        "category": "business",
    },
}

# Fallback static pages if RSS feeds fail (for demonstration purposes)
FALLBACK_URLS = [
    "https://www.bbc.com/news/business",
    "https://www.reuters.com/business/",
    "https://edition.cnn.com/business",
    "https://www.ft.com/world",
]

# Request configuration
REQUEST_TIMEOUT = 15
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

# Scraping limits (academic scale)
MAX_ARTICLES_PER_FEED = 10
MAX_TOTAL_ARTICLES = 50
MIN_ARTICLE_AGE_DAYS = 0   # Accept articles from today
MAX_ARTICLE_AGE_DAYS = 30  # Accept articles up to 30 days old


def ensure_directories() -> None:
    """Create output directories if they do not exist."""
    RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory ensured: {RAW_HTML_DIR}")


def get_random_user_agent() -> str:
    """Return a random User-Agent string to avoid blocking."""
    return random.choice(USER_AGENTS)


def fetch_url(
    url: str,
    max_retries: int = MAX_RETRIES,
    timeout: int = REQUEST_TIMEOUT,
) -> Optional[requests.Response]:
    """
    Fetch a URL with retry logic and random User-Agent.

    Args:
        url: URL to fetch.
        max_retries: Maximum number of retries on failure.
        timeout: Request timeout in seconds.

    Returns:
        Response object if successful, None otherwise.
    """
    headers = {"User-Agent": get_random_user_agent()}

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Fetching {url} (attempt {attempt}/{max_retries})")
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed (attempt {attempt}/{max_retries}): {e}")
            if attempt == max_retries:
                logger.error(f"Max retries exceeded for {url}")
                return None
            delay = RETRY_DELAY_BASE * attempt + random.uniform(0, 1)
            logger.info(f"Waiting {delay:.1f}s before retry...")
            time.sleep(delay)

    return None


def parse_rss_feed(feed_url: str, feed_name: str) -> List[Dict]:
    """
    Parse an RSS feed and extract article URLs and metadata.

    Args:
        feed_url: URL of the RSS feed.
        feed_name: Human-readable name of the feed source.

    Returns:
        List of article dictionaries with keys: title, url, published, source.
    """
    logger.info(f"Parsing RSS feed: {feed_name} ({feed_url})")

    articles = []
    try:
        feed = feedparser.parse(feed_url)

        if feed.bozo:
            logger.warning(f"Feed parsing warning for {feed_name}: {feed.bozo_exception}")

        if not feed.entries:
            logger.warning(f"No entries found in feed: {feed_name}")
            return articles

        cutoff_date = datetime.now() - timedelta(days=MAX_ARTICLE_AGE_DAYS)

        for entry in feed.entries[:MAX_ARTICLES_PER_FEED]:
            # Extract publication date
            published = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                published = datetime(*entry.published_parsed[:6])
            elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                published = datetime(*entry.updated_parsed[:6])
            else:
                published = datetime.now()

            # Skip articles older than cutoff
            if published < cutoff_date:
                logger.debug(f"Skipping old article: {entry.get('title', 'N/A')}")
                continue

            # Extract URL
            url = None
            if hasattr(entry, "link"):
                url = entry.link
            elif hasattr(entry, "links") and entry.links:
                for link in entry.links:
                    if link.get("rel") == "alternate" or link.get("type", "").startswith("text/html"):
                        url = link.get("href")
                        break

            if not url:
                logger.warning(f"No URL found for article: {entry.get('title', 'N/A')}")
                continue

            article = {
                "title": entry.get("title", "Untitled"),
                "url": url,
                "published": published.isoformat(),
                "source": feed_name,
                "feed_url": feed_url,
            }
            articles.append(article)

        logger.info(f"Parsed {len(articles)} articles from {feed_name}")
        return articles

    except Exception as e:
        logger.error(f"Failed to parse RSS feed {feed_name}: {e}")
        return articles


def fetch_article_html(article: Dict) -> Optional[Dict]:
    """
    Download the raw HTML of an article and save it to disk.

    Args:
        article: Article dictionary with keys: title, url, published, source.

    Returns:
        Updated article dictionary with additional keys: html_path, html_size,
        content_type, fetch_timestamp, status_code. None if fetch failed.
    """
    url = article["url"]
    source = article["source"]

    response = fetch_url(url)
    if response is None:
        logger.error(f"Failed to fetch article: {article['title']}")
        return None

    # Generate filename from URL hash to avoid collisions
    url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()[:12]
    date_str = datetime.now().strftime("%Y%m%d")
    filename = f"{source}_{date_str}_{url_hash}.html"
    filepath = RAW_HTML_DIR / filename

    # Save raw HTML (this simulates storing a WARC record)
    html_content = response.text
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)

    html_size = len(html_content.encode("utf-8"))

    logger.info(
        f"Saved article HTML: {filename} ({html_size} bytes) from {source}"
    )

    article.update({
        "html_path": str(filepath),
        "html_size": html_size,
        "content_type": response.headers.get("Content-Type", "unknown"),
        "fetch_timestamp": datetime.now().isoformat(),
        "status_code": response.status_code,
    })

    return article


def scrape_rss_feeds() -> List[Dict]:
    """
    Scrape all configured RSS feeds and download article HTML.

    Returns:
        List of successfully scraped article dictionaries.
    """
    all_articles = []

    for feed_name, feed_config in RSS_FEEDS.items():
        logger.info(f"
{'='*60}")
        logger.info(f"Processing feed: {feed_name}")
        logger.info(f"{'='*60}")

        # Parse RSS feed
        articles = parse_rss_feed(feed_config["url"], feed_name)

        if not articles:
            logger.warning(f"No articles found in {feed_name}, skipping")
            continue

        # Fetch HTML for each article
        for article in articles:
            if len(all_articles) >= MAX_TOTAL_ARTICLES:
                logger.info(f"Reached maximum article limit ({MAX_TOTAL_ARTICLES})")
                break

            fetched_article = fetch_article_html(article)
            if fetched_article:
                all_articles.append(fetched_article)

            # Respectful delay between requests
            time.sleep(random.uniform(1.0, 3.0))

        # Delay between feeds
        time.sleep(random.uniform(2.0, 5.0))

        if len(all_articles) >= MAX_TOTAL_ARTICLES:
            break

    logger.info(f"
Total articles scraped: {len(all_articles)}")
    return all_articles


def scrape_fallback_pages() -> List[Dict]:
    """
    Scrape fallback static pages if RSS feeds produce insufficient data.

    Returns:
        List of page dictionaries with HTML content.
    """
    pages = []

    for url in FALLBACK_URLS:
        if len(pages) >= 5:  # Limit fallback pages
            break

        logger.info(f"Fetching fallback page: {url}")
        response = fetch_url(url)
        if response is None:
            continue

        parsed = urlparse(url)
        domain = parsed.netloc.replace("www.", "")
        url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()[:12]
        date_str = datetime.now().strftime("%Y%m%d")
        filename = f"fallback_{domain}_{date_str}_{url_hash}.html"
        filepath = RAW_HTML_DIR / filename

        html_content = response.text
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        pages.append({
            "title": f"Fallback page: {domain}",
            "url": url,
            "published": datetime.now().isoformat(),
            "source": f"fallback_{domain}",
            "html_path": str(filepath),
            "html_size": len(html_content.encode("utf-8")),
            "content_type": response.headers.get("Content-Type", "unknown"),
            "fetch_timestamp": datetime.now().isoformat(),
            "status_code": response.status_code,
        })

        logger.info(f"Saved fallback page: {filename}")
        time.sleep(random.uniform(2.0, 4.0))

    return pages


def save_metadata(articles: List[Dict], pages: List[Dict]) -> str:
    """
    Save scraping metadata to JSON for traceability.

    Args:
        articles: List of scraped article dictionaries.
        pages: List of fallback page dictionaries.

    Returns:
        Path to the saved metadata file.
    """
    metadata = {
        "scraping_timestamp": datetime.now().isoformat(),
        "total_articles": len(articles),
        "total_fallback_pages": len(pages),
        "total_records": len(articles) + len(pages),
        "rss_feeds_processed": list(RSS_FEEDS.keys()),
        "fallback_urls": FALLBACK_URLS,
        "configuration": {
            "max_articles_per_feed": MAX_ARTICLES_PER_FEED,
            "max_total_articles": MAX_TOTAL_ARTICLES,
            "max_article_age_days": MAX_ARTICLE_AGE_DAYS,
            "request_timeout": REQUEST_TIMEOUT,
            "max_retries": MAX_RETRIES,
        },
        "articles": articles,
        "fallback_pages": pages,
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"Metadata saved to {METADATA_FILE}")
    return str(METADATA_FILE)


def generate_summary(articles: List[Dict], pages: List[Dict]) -> None:
    """
    Print a summary of the scraping operation to the console.

    Args:
        articles: List of scraped articles.
        pages: List of fallback pages.
    """
    total_size = sum(a.get("html_size", 0) for a in articles)
    total_size += sum(p.get("html_size", 0) for p in pages)

    print("
" + "=" * 60)
    print("NEWS SCRAPING SUMMARY")
    print("=" * 60)
    print(f"Total articles scraped:      {len(articles)}")
    print(f"Total fallback pages:        {len(pages)}")
    print(f"Total raw HTML records:      {len(articles) + len(pages)}")
    print(f"Total raw HTML size:         {total_size / 1024:.1f} KB")
    print(f"Output directory:            {RAW_HTML_DIR.resolve()}")
    print(f"Metadata file:               {METADATA_FILE.resolve()}")
    print("=" * 60)

    if articles:
        print("
Articles by source:")
        sources = {}
        for article in articles:
            src = article.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
        for src, count in sorted(sources.items()):
            print(f"  {src}: {count} articles")

    print("
Next step: Run text_processor.py to clean HTML and extract text.")


def main():
    """
    Main execution entry point.

    Scrapes RSS feeds for economic news articles, downloads their raw HTML,
    stores the files simulating a CommonCrawl WARC pipeline, and saves
    metadata for downstream text processing.
    """
    logger.info("=" * 60)
    logger.info("NEWS SCRAPER - COMMONCRAWL PIPELINE SIMULATION")
    logger.info("=" * 60)
    logger.info("This module simulates the CommonCrawl data acquisition workflow:")
    logger.info("  1. URL discovery (RSS feeds -> WARC index analogy)")
    logger.info("  2. Raw HTML download (HTTP requests -> S3 download analogy)")
    logger.info("  3. HTML storage with metadata (WARC record analogy)")
    logger.info("=" * 60)

    ensure_directories()

    # Step 1: Scrape RSS feeds
    logger.info("
Phase 1: RSS Feed Scraping (URL Discovery)")
    articles = scrape_rss_feeds()

    # Step 2: Fallback pages if needed
    pages = []
    if len(articles) < 10:
        logger.info("
Phase 2: Fallback Page Scraping")
        logger.info("Insufficient articles from RSS, fetching fallback pages...")
        pages = scrape_fallback_pages()
    else:
        logger.info("
Phase 2: Fallback scraping skipped (sufficient RSS data)")

    # Step 3: Save metadata
    logger.info("
Phase 3: Saving Metadata")
    save_metadata(articles, pages)

    # Step 4: Generate summary
    generate_summary(articles, pages)

    logger.info("
" + "=" * 60)
    logger.info("SCRAPING COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Raw HTML files saved to: {RAW_HTML_DIR}")
    logger.info(f"Metadata saved to: {METADATA_FILE}")
    logger.info("Next: Run python -m scraping.text_processor")


if __name__ == "__main__":
    main()
