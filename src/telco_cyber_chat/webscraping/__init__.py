# telco_cyber_chat/webscraping/__init__.py

from .scrape_core import url_already_ingested

from .nokia_scraper import scrape_nokia
from .cisco_scraper import scrape_cisco
from .variot_scraper import scrape_variot
from .ericsson_scraper import scrape_ericsson
from .huawei_scraper import scrape_huawei

__all__ = [
    "url_already_ingested",
    "scrape_nokia",
    "scrape_cisco",
    "scrape_variot",
    "scrape_ericsson",
    "scrape_huawei",
]
