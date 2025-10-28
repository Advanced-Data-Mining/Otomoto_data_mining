import requests, bs4
from concurrent.futures import ThreadPoolExecutor
import time
from tqdm import tqdm
import sys
import re
import pandas as pd
import os
import pyarrow


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


def scrap_model():
    path = 'https://www.otomoto.pl/dostawcze'
    
    try:
        res = requests.get(path, headers = {
            "Content-Type": "application/json",
            "User-Agent": "Safari/537.36"
        })
        res.raise_for_status()
    except Exception as e:
        raise Exception(e)

    try:
        number_of_pages = get_number_of_pages(path)
    except Exception as e:
        print("Number of pages not found: ",e)
        number_of_pages = 1

    number_of_pages = 1
    threads = min(MAX_THREADS, number_of_pages)
    path_array = [path]*number_of_pages
    pages_range = range(1, number_of_pages + 1)

    links.clear()

    with ThreadPoolExecutor(max_workers=threads) as executor:
        for page_num, page_cars in tqdm(
                executor.map(get_cars_in_page2, path_array, pages_range),
                total=len(path_array),  # or len(pages_range)
                desc="🔍 Fetching car data",  # Custom description
                colour='cyan',  # Progress bar color (in terminals that support it)
                ncols=100,  # Width of the progress bar
                smoothing=0.3,  # Smoothing for speed estimate
                unit="req",  # Label per iteration
                dynamic_ncols=True,  # Dynamically fit to terminal
                file=sys.stdout
        ):
            write_page_to_parquet(page_num, page_cars)

    print(f"Total links collected: {len(links)}")
    print(f"Data saved to parquet files in 'data/' directory")


    time.sleep(0.25)


scrap_model()
