import time
import requests
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

MONGO_URI = "mongodb://localhost:27017"
DATABASE_NAME = "dota"
MATCHES_COLLECTION = "matches"
MATCHES_INFO_COLLECTION = "matches_info"
BENCHMARKS_COLLECTION = "heroes"

client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
matches_collection = db[MATCHES_COLLECTION]
matches_info_collection = db[MATCHES_INFO_COLLECTION]
heroes_collection = db[BENCHMARKS_COLLECTION]

matches_info_collection.create_index("match_id", unique=True)
heroes_collection.create_index("hero_id", unique=True)

LAST_MATCH_FILE = "last_match_id_info.txt"

API_KEY = "API_KEY"


def save_last_match_id(match_id):
    with open(LAST_MATCH_FILE, "w") as f:
        f.write(str(match_id))
        print(f"Saved last processed match_id: {match_id}")


def load_last_match_id():
    try:
        with open(LAST_MATCH_FILE, "r") as f:
            match_id = int(f.read().strip())
            print(f"Loaded last processed match_id: {match_id}")
            return match_id
    except FileNotFoundError:
        print("No last match_id found, starting from the latest matches.")
        return None


def fetch_match_details():
    last_match_id = load_last_match_id()
    query = {"match_id": {"$gt": last_match_id}} if last_match_id else {}

    matches = matches_collection.find(query, {"match_id": 1}).sort("match_id", 1)
    total_matches = matches_collection.count_documents(query)
    print(f"Found {total_matches} matches to process.")

    processed_count = 0

    for match in matches:
        match_id = match.get("match_id")
        if match_id:
            existing_info = matches_info_collection.find_one({"match_id": match_id})
            if existing_info:
                print(f"Match {match_id} already exists in matches_info. Skipping...")
                continue

            url = f"https://api.opendota.com/api/matches/{match_id}"
            params = {"api_key": API_KEY}
            print(f'Fetching detailed info for match {match_id}...')

            try:
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    match_data = response.json()
                    match_data["match_id"] = match_id

                    try:
                        matches_info_collection.insert_one(match_data)
                        save_last_match_id(match_id)
                        processed_count += 1
                        print(f"Match {match_id} saved to matches_info. Total processed: {processed_count}/{total_matches}")

                    except DuplicateKeyError:
                        print(f"Match {match_id} already in matches_info. Skipping...")

                elif response.status_code == 429:
                    print("Rate limit exceeded, waiting 60 seconds...")
                    time.sleep(60)

                else:
                    print(f"Failed to fetch match {match_id}: {response.status_code}")

            except requests.RequestException as e:
                print(f"Request failed for match {match_id}: {e}")
                time.sleep(5)


    print(f"Finished processing. Total matches processed: {processed_count}/{total_matches}")

def fetch_heroes_benchmarks(hero_id):
    params = {"api_key": API_KEY}
    url = f"https://api.opendota.com/api/benchmarks?hero_id={hero_id}"
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        result = data.get("result", {})
        result["hero_id"] = hero_id
        return result
    else:
        raise Exception(f"Failed to fetch benchmarks for hero_id: {hero_id}")



def get_heroes_with_benchmarks():
    params = {"api_key": API_KEY}
    url = f"https://api.opendota.com/api/heroes"
    response = requests.get(url, params=params)
    result = []
    if response.status_code == 200:
        heroes_data = response.json()
        for hero in heroes_data:
            benchmark = fetch_heroes_benchmarks(hero["id"])
            combined = {**hero, **benchmark}
            result.append(combined)

        return result
    else:
        raise Exception(f"Failed to get heroes data: {response.status_code}")

def write_heroes_to_collection(data):
    heroes_collection.insert_many(data)

if __name__ == "__main__":
    # fetch_match_details()
    # heroes = get_heroes_with_benchmarks()
    # write_heroes_to_collection(heroes)
    import requests
    from datetime import datetime

    url = "https://api.opendota.com/api/constants/patch"
    data = requests.get(url).json()
    print(data)
    for x in data:
        print(x)
    # for patch_id, info in sorted(data.items(), key=lambda x: int(x[0]), reverse=True):
    #     name = info['name']
    #     date = datetime.fromtimestamp(info['release_date']).strftime('%Y-%m-%d')
    #     print(f"{patch_id}: {name} — {date}")

