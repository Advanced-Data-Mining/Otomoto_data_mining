import requests, bs4
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm
import sys
import re
import pandas as pd
import os
import pyarrow
import argparse


MAX_THREADS = 50
links = []


def get_testid_text(soup, testid):
    el = soup.find(attrs={"data-testid": testid})
    if not el:
        return None
    val = el.find("p", class_=re.compile("ugve4x"))
    return val.get_text(strip=True) if val else None

def extract_car_properties(soup):
    data = {}

    # Core attributes
    data["brand"] = get_testid_text(soup, "make")
    data["model"] = get_testid_text(soup, "model")
    data["color"] = get_testid_text(soup, "color")
    data["seats"] = get_testid_text(soup, "nr_seats")
    data["year"] = get_testid_text(soup, "year")
    data["fuel"] = get_testid_text(soup, "fuel_type")
    data["capacity"] = get_testid_text(soup, "engine_capacity")
    data["power"] = get_testid_text(soup, "engine_power")
    data["body_type"] = get_testid_text(soup, "body_type")
    data["gearbox"] = get_testid_text(soup, "gearbox")
    data["mileage"] = get_testid_text(soup, "mileage")
    data["condition"] = get_testid_text(soup, "new_used")
    data["accident_free"] = get_testid_text(soup, "no_accident")
    data["country_of_origin"] = get_testid_text(soup, "country_origin")

    # Title
    h1 = soup.find("h1", class_=re.compile("offer-title"))
    data["title"] = h1.get_text(strip=True) if h1 else None

    # Prices
    price = soup.find("span", class_="offer-price__number")
    data["price_pln"] = price.get_text(strip=True) if price else None

    net_info = soup.find("p", class_=re.compile("5jshcp"))
    data["price_net_info"] = net_info.get_text(strip=True) if net_info else None

    # Location
    location = soup.find("a", href=re.compile("google.com/maps"))
    data["location"] = location.get_text(strip=True) if location else None

    # Additional equipment
    equipment_section = soup.find("div", id="content-equipments-section")
    equipment = []
    if equipment_section:
        for p in equipment_section.find_all("p", class_=re.compile("iqjlxi")):
            equipment.append(p.get_text(strip=True))
    data["equipment"] = equipment

    # Posted date
    posted_date = soup.find("p", class_=re.compile("zv8mpc"))
    data["posted_date"] = posted_date.get_text(strip=True) if posted_date else None
    
    # Description ("Opis")
    desc_section = soup.find(attrs={"data-testid": "content-description-section"}).get_text(separator="\n", strip=True)

    data["description"] = desc_section

    return data

def get_number_of_pages(path) -> int:
    res = requests.get(path, headers = {
        "Content-Type": "application/json",
        "User-Agent": "Safari/537.36"
    })
    res.raise_for_status()
    car_soup = bs4.BeautifulSoup(res.text, "html.parser")
    last_page_button = car_soup.find_all("button", {"class": "ooa-13ptg7a"})[-1]
    number_of_pages = last_page_button.get_text().strip()
    return int(number_of_pages)


def get_car_details(path):
    res = requests.get(path, headers = {
        "Content-Type": "application/json",
        "User-Agent": "Safari/537.36"
    })
    res.raise_for_status()
    car_soup = bs4.BeautifulSoup(res.text, "html.parser")
    car_data = extract_car_properties(car_soup)
    car_data["url"] = path
    return car_data

def get_cars_in_page2(path, page_num):
    # Check if parquet file already exists
    filename = f"data/page_{page_num:03d}.parquet"
    if os.path.exists(filename):
        return page_num, []

    res = requests.get(path + '?page=' + str(page_num), headers = {
        "Content-Type": "application/json",
        "User-Agent": "Safari/537.36"
    })
    res.raise_for_status()
    currentPage = bs4.BeautifulSoup(res.text, "html.parser")

    page_cars = []
    for x in currentPage.find_all("h2", {"class": "etydmma0 ooa-ezpr21"}):
        x = x.find('a', href=True)
        links.append(x['href'])
        car_details = get_car_details(x['href'])
        page_cars.append(car_details)

    return page_num, page_cars


def write_page_to_parquet(page_num, page_cars):
    if not page_cars:
        return

    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame(page_cars)
    filename = f"data/page_{page_num:03d}.parquet"
    df.to_parquet(filename, index=False)
    print(f"Saved {len(page_cars)} cars to {filename}")


def parse_args():
    parser = argparse.ArgumentParser(description='Scrape car data from otomoto.pl')
    parser.add_argument('--number-of-pages', type=int, default=50,
                       help='Number of pages to scrape (default: 50)')
    parser.add_argument('--start-from-page', type=int, default=1,
                       help='Starting page number (default: 1)')
    parser.add_argument('--max-threads', type=int, default=MAX_THREADS,
                       help=f'Maximum number of threads (default: {MAX_THREADS})')
    parser.add_argument('--url', type=str, default='https://www.otomoto.pl/dostawcze',
                       help='Base URL to scrape (default: otomoto.pl commercial vehicles)')
    return parser.parse_args()

def scrap_model(number_of_pages=50, start_from_page=1, max_threads=MAX_THREADS, url='https://www.otomoto.pl/dostawcze'):
    path = url

    try:
        res = requests.get(path, headers = {
            "Content-Type": "application/json",
            "User-Agent": "Safari/537.36"
        })
        res.raise_for_status()
    except Exception as e:
        raise Exception(f"Failed to access {path}: {e}")

    try:
        pages_count = get_number_of_pages(path)
        print(f"Total pages available: {pages_count}")
    except Exception as e:
        print("Number of pages not found: ",e)
        pages_count = 1

    if start_from_page + number_of_pages - 1 > pages_count:
        print(f"Warning: Requested range ({start_from_page}-{start_from_page + number_of_pages - 1}) exceeds available pages ({pages_count})")
        number_of_pages = max(1, pages_count - start_from_page + 1)
        print(f"Adjusted to scrape {number_of_pages} pages")

    threads = min(max_threads, number_of_pages)

    pages_range = range(start_from_page, start_from_page + number_of_pages)

    links.clear()

    with ThreadPoolExecutor(max_workers=threads) as executor:
        # Submit all tasks
        future_to_page = {
            executor.submit(get_cars_in_page2, path, page_num): page_num
            for page_num in pages_range
        }

        # Process results as they complete
        for future in tqdm(
                as_completed(future_to_page),
                total=len(future_to_page),
                desc="🔍 Fetching car data",
                colour='cyan',
                ncols=100,
                smoothing=0.3,
                unit="page",
                dynamic_ncols=True,
                file=sys.stdout
        ):
            page_num, page_cars = future.result()
            write_page_to_parquet(page_num, page_cars)

    print(f"Total links collected: {len(links)}")
    print(f"Data saved to parquet files in 'data/' directory")


    time.sleep(0.25)


if __name__ == "__main__":
    args = parse_args()

    if args.number_of_pages <= 0:
        print("Error: Number of pages must be positive")
        sys.exit(1)

    if args.start_from_page <= 0:
        print("Error: Start page must be positive")
        sys.exit(1)

    if args.max_threads <= 0:
        print("Error: Max threads must be positive")
        sys.exit(1)

    print(f"Scraping {args.number_of_pages} pages starting from page {args.start_from_page}")
    print(f"Using {args.max_threads} max threads")
    print(f"URL: {args.url}")

    scrap_model(
        number_of_pages=args.number_of_pages,
        start_from_page=args.start_from_page,
        max_threads=args.max_threads,
        url=args.url
    )
