import time
import requests
import json
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from fp.fp import FreeProxy

# Параметри підключення до MongoDB
MONGO_URI = "mongodb://localhost:27017"
DATABASE_NAME = "dota"
COLLECTION_NAME = "matches"

# Налаштовуємо MongoDB
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
matches_collection = db[COLLECTION_NAME]

# Додаємо індекс на поле match_id, щоб уникнути дублювання записів
matches_collection.create_index("match_id", unique=True)

# Файл для збереження останнього обробленого match_id
LAST_MATCH_FILE = "last_match_id.txt"


# Функція для отримання рандомного проксі-сервера через FreeProxy
def get_random_proxy():
    try:
        proxy = FreeProxy(timeout=1, rand=True, https=True).get()
        print(f"Using proxy: {proxy}")
        return {"http": proxy, "https": proxy}
    except Exception as e:
        print(f"Failed to get proxy: {e}")
        return None


# Функція для збереження останнього обробленого match_id у файл
def save_last_match_id(match_id):
    with open(LAST_MATCH_FILE, "w") as f:
        f.write(str(match_id))
        print(f"Saved last processed match_id: {match_id}")


# Функція для завантаження останнього обробленого match_id з файлу
def load_last_match_id():
    try:
        with open(LAST_MATCH_FILE, "r") as f:
            match_id = int(f.read().strip())
            print(f"Loaded last processed match_id: {match_id}")
            return match_id
    except FileNotFoundError:
        print("No last match_id found, starting from the latest matches.")
        return None


# Функція для парсингу матчів з OpenDota API
def fetch_high_rank_matches(limit=1000, min_mmr=6000):
    last_match_id = load_last_match_id()

    query = f"""
    SELECT match_id, start_time
    FROM public_matches
    WHERE match_id > {last_match_id if last_match_id else 0}
    ORDER BY match_id ASC
    LIMIT {limit};
    """

    url = "https://api.opendota.com/api/explorer"
    params = {"sql": query}

    while True:
        proxy = get_random_proxy()
        try:
            response = requests.get(url, params=params, proxies=proxy, timeout=10)
            if response.status_code == 200:
                data = response.json()
                matches = data.get("rows", [])

                for match in matches:
                    try:
                        matches_collection.insert_one(match)
                        save_last_match_id(match['match_id'])
                    except DuplicateKeyError:
                        print(f"Match {match['match_id']} already exists in the database.")

                print(f"Successfully saved {len(matches)} matches to MongoDB")
                break
            elif response.status_code == 429:
                print("Rate limit exceeded, waiting 60 seconds...")
                time.sleep(60)
            else:
                print(f"Error {response.status_code}: {response.text}")
        except requests.RequestException as e:
            print(f"Proxy failed: {e}")
            time.sleep(5)  # Затримка перед вибором нового проксі
        time.sleep(1)


# Функція для парсингу деталей матчів з OpenDota API
def fetch_match_details():
    last_match_id = load_last_match_id()
    query = {"match_id": {"$gt": last_match_id}} if last_match_id else {}
    matches = matches_collection.find(query, {"match_id": 1}).sort("match_id", 1)

    for match in matches:
        match_id = match.get("match_id")
        if match_id:
            url = f"https://api.opendota.com/api/matches/{match_id}"
            print(f'Parsing match {url}')

            while True:
                proxy = get_random_proxy()
                try:
                    response = requests.get(url, proxies=proxy, timeout=10)
                    if response.status_code == 200:
                        match_data = response.json()
                        try:
                            matches_collection.update_one(
                                {"match_id": match_id},
                                {"$set": match_data},
                                upsert=True
                            )
                            save_last_match_id(match_id)
                            print(f"Match {match_id} saved to MongoDB")
                        except Exception as e:
                            print(f"Failed to save match {match_id}: {e}")
                        break
                    elif response.status_code == 429:
                        print("Rate limit exceeded, waiting 60 seconds...")
                        time.sleep(60)
                    else:
                        print(f"Failed to fetch match {match_id}: {response.status_code}")
                        break
                except requests.RequestException as e:
                    print(f"Proxy failed: {e}")
                    time.sleep(0.1)  # Затримка перед вибором нового проксі

                # time.sleep(0.5)


if __name__ == "__main__":
    # fetch_high_rank_matches()
    fetch_match_details()
