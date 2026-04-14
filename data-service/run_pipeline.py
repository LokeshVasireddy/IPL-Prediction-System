from pipeline.clean_deliveries import clean_deliveries
from pipeline.clean_matches import clean_matches


def run():

    print("\nSTEP 1: CLEAN MATCHES")
    clean_matches()

    print("\nSTEP 2: CLEAN DELIVERIES")
    clean_deliveries()

    print("\nPipeline completed successfully")


if __name__ == "__main__":
    run()
