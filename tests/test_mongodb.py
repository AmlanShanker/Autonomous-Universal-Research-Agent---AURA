from config.settings import MONGO_URL
from storage.database.mongo import MongoDatabase


def main():
    database = MongoDatabase(
        connection_string=MONGO_URL,
        database_name="aura"
    )

    if database.ping():
        print("MongoDB connection successful!")
    else:
        print("MongoDB connection failed.")


if __name__ == "__main__":
    main()