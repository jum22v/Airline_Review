import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

reviews = []

for pageNum in range (1, 21):

    url = f"https://www.airlinequality.com/airline-reviews/southwest-airlines/page/{pageNum}/"

    print(f"Scraping page: {pageNum}")

    response = requests.get(url)

    if response.status_code != 200:
        print(f"Failed to retrieve page {pageNum}. Skipping.")
        continue

    parser = BeautifulSoup(response.text, 'html.parser')

    reviewContainers = parser.find_all('div', class_='text_content')

    for container in reviewContainers:
        reviewText = container.text.strip()
        reviews.append(reviewText)

    time.sleep(1)

    dataFrame = pd.DataFrame(reviews, columns = ['Review'])
    dataFrame.to_csv('reviews.csv', index = False)

    print(f"Succesfully saved {len(reviews)} reviews to file")

