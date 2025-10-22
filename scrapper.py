import requests, bs4
from concurrent.futures import ThreadPoolExecutor
import time
from tqdm import tqdm
import sys



MAX_THREADS = 50
links = []

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



def get_cars_in_page2(path, i):
    res = requests.get(path + '?page=' + str(i), headers = {
        "Content-Type": "application/json",
        "User-Agent": "Safari/537.36"
    })
    res.raise_for_status()
    currentPage = bs4.BeautifulSoup(res.text, "html.parser")

    # print(currentPage.find_all("h2", {"class": "etydmma0 ooa-ezpr21"}))
    for x in currentPage.find_all("h2", {"class": "etydmma0 ooa-ezpr21"}):
        x = x.find('a', href=True)
        links.append(x['href'])


def scrap_model():
    path = 'https://www.otomoto.pl/dostawcze'
    print(path)
    try:
        res = requests.get(path, headers = {
            "Content-Type": "application/json",
            "User-Agent": "Safari/537.36"
        })
        res.raise_for_status()
        car_soup = bs4.BeautifulSoup(res.text, "html.parser")
    except Exception as e:
        raise Exception(e)

    try:
        number_of_pages = get_number_of_pages(path)
    except Exception as e:
        print("Number of pages not found: ",e)
        number_of_pages = 1

    threads = min(MAX_THREADS, number_of_pages)
    path_array = [path]*number_of_pages
    pages_range = range(1, number_of_pages + 1)

    links.clear()

    with ThreadPoolExecutor(max_workers=threads) as executor:
        for _ in tqdm(
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
            pass

    print(links)
    print(len(links))


    time.sleep(0.25)


scrap_model()
