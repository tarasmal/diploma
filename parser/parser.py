from pymongo import MongoClient
import requests

MONGO_URI = "mongodb://localhost:27017"
DATABASE_NAME = "dota"
COLLECTION_NAME = "heroes"

client = MongoClient(MONGO_URI)
collection = client[DATABASE_NAME][COLLECTION_NAME]

response = requests.get("https://api.opendota.com/api/heroStats")
hero_stats = response.json()

for hero in hero_stats:
    hero_id = hero["id"]

    fields_to_unset = {}

    for i in range(1, 9):
        fields_to_unset[f"{i}_pick"] = ""
        fields_to_unset[f"{i}_win"] = ""
        fields_to_unset[f"{i}_pick_winrate"] = ""

    for key in [
        "phase_1_pick_winrate",
        "phase_2_pick_winrate",
        "pro_pick",
        "pro_win",
        "pro_ban",
        "turbo_picks",
        "turbo_wins",
    ]:
        fields_to_unset[key] = ""

    updates = {}
    pro_pick = hero.get("pro_pick", 0)
    pro_win = hero.get("pro_win", 0)
    updates["pro_winrate"] = pro_win / pro_pick if pro_pick > 0 else None

    collection.update_one(
        {"hero_id": hero_id},
        {"$unset": fields_to_unset, "$set": updates}
    )
