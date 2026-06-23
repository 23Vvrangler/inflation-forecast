"""
World Bank API Scraper for Inflation Data.

Downloads economic indicators from the World Bank Open Data API for multiple
Latin American countries. Indicators fetched:
    - FP.CPI.TOTL.ZG : Inflation, consumer prices (annual %)
    - NY.GDP.MKTP.KD.ZG : GDP growth (annual %)
    - FR.INR.RINR : Real interest rate (%)

The World Bank API is free, requires no authentication, and returns JSON.
Documentation: https://datahelpdesk.worldbank.org/knowledgebase/articles/898599-api-documentation

Usage:
    python -m scraping.worldbank_scraper

Output:
    data/raw/inflation_{country_code}.csv
    data/raw/gdp_growth_{country_code}.csv
    data/raw/interest_rate_{country_code}.csv
"""
import os
import sys
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# World Bank API configuration
WB_API_BASE = "https://api.worldbank.org/v2/country"
WB_API_FORMAT = "json"
WB_API_PER_PAGE = 500  # Max records per page

# Countries to scrape (ISO 3166-1 alpha-3 codes)
COUNTRIES = {
    "PER": "Peru",
    "ARG": "Argentina",
    "BRA": "Brazil",
    "CHL": "Chile",
    "COL": "Colombia",
    "MEX": "Mexico",
}

# World Bank indicator codes
INDICATORS = {
    "inflation": {
        "code": "FP.CPI.TOTL.ZG",
        "name": "Inflation, consumer prices (annual %)",
        "column_name": "inflation_rate",
    },
    "gdp_growth": {
        "code": "NY.GDP.MKTP.KD.ZG",
        "name": "GDP growth (annual %)",
        "column_name": "gdp_growth",
    },
    "interest_rate": {
        "code": "FR.INR.RINR",
        "name": "Real interest rate (%)",
        "column_name": "interest_rate",
    },
}

# Date range for scraping
START_YEAR = 2000
END_YEAR = 2024

# Output directories
RAW_DATA_DIR = Path("data/raw")


def ensure_directories() -> None:
    """Create output directories if they do not exist."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory ensured: {RAW_DATA_DIR}")


def fetch_indicator(
    country_code: str,
    indicator_code: str,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> List[Dict]:
    """
    Fetch all data pages for a single indicator from the World Bank API.

    The World Bank API uses pagination. This function follows the
    `page` and `pages` fields in the response to retrieve all records.

    Args:
        country_code: ISO 3166-1 alpha-3 country code (e.g., "PER").
        indicator_code: World Bank indicator code (e.g., "FP.CPI.TOTL.ZG").
        start_year: Start year for the query (inclusive).
        end_year: End year for the query (inclusive).
        max_retries: Maximum number of retries on network errors.
        retry_delay: Delay in seconds between retries.

    Returns:
        List of raw data dictionaries from the API. Each dict contains
        fields: indicator, country, countryiso3code, date, value, unit, obs_status, decimal.
    """
    url = (
        f"{WB_API_BASE}/{country_code}/indicator/{indicator_code}"
        f"?format={WB_API_FORMAT}"
        f"&date={start_year}:{end_year}"
        f"&per_page={WB_API_PER_PAGE}"
    )

    all_records = []
    current_page = 1
    total_pages = 1

    while current_page <= total_pages:
        page_url = f"{url}&page={current_page}"
        logger.info(
            f"Fetching {country_code} | {indicator_code} | page {current_page}/{total_pages}"
        )

        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get(page_url, timeout=30)
                response.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"Request failed (attempt {attempt}/{max_retries}): {e}"
                )
                if attempt == max_retries:
                    logger.error(f"Max retries exceeded for {page_url}")
                    raise
                time.sleep(retry_delay * attempt)

        data = response.json()

        # The World Bank API returns a list where:
        #   data[0] is metadata (pagination info)
        #   data[1] is the list of records
        if not isinstance(data, list) or len(data) < 2:
            logger.error(f"Unexpected API response format: {data}")
            raise ValueError("Unexpected API response format")

        metadata = data[0]
        records = data[1]

        total_pages = metadata.get("pages", 1)
        current_page = metadata.get("page", 1) + 1

        if records:
            all_records.extend(records)
            logger.info(
                f"Retrieved {len(records)} records (total so far: {len(all_records)})"
            )
        else:
            logger.info("No records on this page")
            break

        # Respectful delay between pages
        time.sleep(0.5)

    logger.info(
        f"Total records fetched for {country_code} | {indicator_code}: {len(all_records)}"
    )
    return all_records


def parse_records_to_dataframe(
    records: List[Dict],
    indicator_config: Dict,
    country_code: str,
    country_name: str,
) -> pd.DataFrame:
    """
    Convert raw World Bank API records into a clean pandas DataFrame.

    Args:
        records: List of raw record dictionaries from the API.
        indicator_config: Dictionary with keys: code, name, column_name.
        country_code: ISO country code.
        country_name: Human-readable country name.

    Returns:
        DataFrame with columns: date, country_code, country_name, {indicator_value}.
    """
    if not records:
        logger.warning(f"No records to parse for {country_code}")
        return pd.DataFrame()

    rows = []
    for record in records:
        value = record.get("value")
        date_str = record.get("date")

        # Skip records with missing values (World Bank returns null for some years)
        if value is None or date_str is None:
            continue

        rows.append(
            {
                "date": date_str,
                "country_code": country_code,
                "country_name": country_name,
                indicator_config["column_name"]: float(value),
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        logger.warning(f"All records had null values for {country_code}")
        return df

    # Convert date to datetime (World Bank returns year as string)
    df["date"] = pd.to_datetime(df["date"], format="%Y")
    df = df.sort_values("date").reset_index(drop=True)

    logger.info(
        f"Parsed {len(df)} records for {country_name} | {indicator_config['name']}"
    )
    return df


def save_dataframe(df: pd.DataFrame, filename: str) -> str:
    """
    Save a DataFrame to CSV in the raw data directory.

    Args:
        df: DataFrame to save.
        filename: Output filename (e.g., "inflation_PER.csv").

    Returns:
        Full path to the saved file.
    """
    filepath = RAW_DATA_DIR / filename
    df.to_csv(filepath, index=False)
    logger.info(f"Saved {len(df)} rows to {filepath}")
    return str(filepath)


def scrape_country_indicator(
    country_code: str,
    country_name: str,
    indicator_key: str,
    indicator_config: Dict,
) -> Optional[pd.DataFrame]:
    """
    Scrape a single indicator for a single country.

    Args:
        country_code: ISO country code.
        country_name: Human-readable country name.
        indicator_key: Key name for the indicator (e.g., "inflation").
        indicator_config: Configuration dict for the indicator.

    Returns:
        DataFrame with the scraped data, or None if failed.
    """
    try:
        records = fetch_indicator(country_code, indicator_config["code"])
        df = parse_records_to_dataframe(
            records, indicator_config, country_code, country_name
        )
        if not df.empty:
            filename = f"{indicator_key}_{country_code}.csv"
            save_dataframe(df, filename)
            return df
        return None
    except Exception as e:
        logger.error(
            f"Failed to scrape {indicator_key} for {country_name} ({country_code}): {e}"
        )
        return None


def scrape_all_countries() -> Dict[str, Dict[str, Optional[pd.DataFrame]]]:
    """
    Scrape all indicators for all configured countries.

    Returns:
        Nested dictionary: {country_code: {indicator_key: DataFrame or None}}.
    """
    results = {}

    for country_code, country_name in COUNTRIES.items():
        logger.info(f"
{'='*60}")
        logger.info(f"Scraping data for {country_name} ({country_code})")
        logger.info(f"{'='*60}")

        country_results = {}
        for indicator_key, indicator_config in INDICATORS.items():
            df = scrape_country_indicator(
                country_code, country_name, indicator_key, indicator_config
            )
            country_results[indicator_key] = df

            # Respectful delay between indicators to avoid rate limiting
            time.sleep(1.0)

        results[country_code] = country_results

        # Respectful delay between countries
        time.sleep(2.0)

    return results


def merge_country_data(country_code: str) -> Optional[pd.DataFrame]:
    """
    Merge all indicators for a single country into one DataFrame.

    Args:
        country_code: ISO country code.

    Returns:
        Merged DataFrame with columns: date, country_code, country_name,
        inflation_rate, gdp_growth, interest_rate.
        Returns None if inflation data is missing (required base indicator).
    """
    inflation_path = RAW_DATA_DIR / f"inflation_{country_code}.csv"
    gdp_path = RAW_DATA_DIR / f"gdp_growth_{country_code}.csv"
    interest_path = RAW_DATA_DIR / f"interest_rate_{country_code}.csv"

    if not inflation_path.exists():
        logger.error(f"Inflation data missing for {country_code}, cannot merge")
        return None

    df = pd.read_csv(inflation_path, parse_dates=["date"])

    if gdp_path.exists():
        df_gdp = pd.read_csv(gdp_path, parse_dates=["date"])
        df = df.merge(
            df_gdp[["date", "gdp_growth"]],
            on="date",
            how="left",
        )
        logger.info(f"Merged GDP growth data for {country_code}")
    else:
        logger.warning(f"GDP growth data missing for {country_code}")
        df["gdp_growth"] = pd.NA

    if interest_path.exists():
        df_interest = pd.read_csv(interest_path, parse_dates=["date"])
        df = df.merge(
            df_interest[["date", "interest_rate"]],
            on="date",
            how="left",
        )
        logger.info(f"Merged interest rate data for {country_code}")
    else:
        logger.warning(f"Interest rate data missing for {country_code}")
        df["interest_rate"] = pd.NA

    # Reorder columns
    column_order = [
        "date",
        "country_code",
        "country_name",
        "inflation_rate",
        "gdp_growth",
        "interest_rate",
    ]
    df = df[[col for col in column_order if col in df.columns]]

    merged_path = RAW_DATA_DIR / f"merged_{country_code}.csv"
    df.to_csv(merged_path, index=False)
    logger.info(f"Saved merged data for {country_code}: {merged_path} ({len(df)} rows)")

    return df


def merge_all_countries() -> pd.DataFrame:
    """
    Merge all country data into a single master DataFrame.

    Returns:
        Combined DataFrame with all countries and all indicators.
    """
    all_dfs = []
    for country_code in COUNTRIES.keys():
        df = merge_country_data(country_code)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        logger.error("No data available to merge")
        return pd.DataFrame()

    master_df = pd.concat(all_dfs, ignore_index=True)
    master_df = master_df.sort_values(["country_code", "date"]).reset_index(drop=True)

    master_path = RAW_DATA_DIR / "master_inflation_data.csv"
    master_df.to_csv(master_path, index=False)
    logger.info(
        f"Saved master dataset: {master_path} ({len(master_df)} rows, {master_df['country_code'].nunique()} countries)"
    )

    return master_df


def generate_scraping_report(results: Dict) -> str:
    """
    Generate a summary report of the scraping operation.

    Args:
        results: Nested dictionary from scrape_all_countries().

    Returns:
        JSON string with the report.
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "countries_scraped": list(COUNTRIES.keys()),
        "indicators_scraped": list(INDICATORS.keys()),
        "date_range": f"{START_YEAR}-{END_YEAR}",
        "details": {},
    }

    for country_code, indicators in results.items():
        country_report = {}
        for indicator_key, df in indicators.items():
            if df is not None and not df.empty:
                country_report[indicator_key] = {
                    "status": "success",
                    "records": len(df),
                    "date_min": df["date"].min().strftime("%Y-%m-%d"),
                    "date_max": df["date"].max().strftime("%Y-%m-%d"),
                    "value_min": float(df[INDICATORS[indicator_key]["column_name"]].min()),
                    "value_max": float(df[INDICATORS[indicator_key]["column_name"]].max()),
                    "value_mean": float(df[INDICATORS[indicator_key]["column_name"]].mean()),
                    "null_count": int(df[INDICATORS[indicator_key]["column_name"]].isna().sum()),
                }
            else:
                country_report[indicator_key] = {
                    "status": "failed_or_empty",
                    "records": 0,
                }
        report["details"][country_code] = country_report

    report_path = RAW_DATA_DIR / "scraping_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Scraping report saved to {report_path}")
    return json.dumps(report, indent=2)


def main():
    """
    Main execution entry point.

    Scrapes all indicators for all countries, merges per-country data,
    creates a master dataset, and generates a scraping report.
    """
    logger.info("=" * 60)
    logger.info("WORLD BANK API SCRAPER - INFLATION DATA")
    logger.info("=" * 60)
    logger.info(f"Countries: {list(COUNTRIES.values())}")
    logger.info(f"Indicators: {[v['name'] for v in INDICATORS.values()]}")
    logger.info(f"Date range: {START_YEAR}-{END_YEAR}")
    logger.info("=" * 60)

    ensure_directories()

    # Step 1: Scrape all indicators for all countries
    results = scrape_all_countries()

    # Step 2: Merge per-country data
    logger.info("
" + "=" * 60)
    logger.info("MERGING COUNTRY DATA")
    logger.info("=" * 60)
    master_df = merge_all_countries()

    # Step 3: Generate report
    logger.info("
" + "=" * 60)
    logger.info("GENERATING SCRAPING REPORT")
    logger.info("=" * 60)
    report = generate_scraping_report(results)
    print("
" + report)

    logger.info("
" + "=" * 60)
    logger.info("SCRAPING COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Output directory: {RAW_DATA_DIR.resolve()}")
    logger.info(f"Files generated: {len(list(RAW_DATA_DIR.glob('*.csv')))} CSV files")
    logger.info(f"Master dataset: {len(master_df)} rows across {master_df['country_code'].nunique()} countries")


if __name__ == "__main__":
    main()
