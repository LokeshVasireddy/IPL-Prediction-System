from pipeline.build_features import build_features
from pipeline.clean_deliveries import clean_deliveries
from pipeline.clean_matches import clean_matches


def run():

    print("\nSTEP 1: CLEAN MATCHES")
    clean_matches()

    print("\nSTEP 2: CLEAN DELIVERIES")
    clean_deliveries()

    print("\nSTEP 3: BUILD FEATURES")
    build_features()

    print("\nPipeline completed successfully")


if __name__ == "__main__":
    run()
