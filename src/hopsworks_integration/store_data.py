import hopsworks
# from src.config import settings

# project = hopsworks.login(
#     api_key_value = settings.HOPSWORKS_API_KEY
# )

# def upload_data():
#     # Get dataset API object
#     dataset_api = project.get_dataset_api()

#     # Upload your local CSV file to Hopsworks
#     dataset_api.upload(
#         # str(settings.SOLAR_GEN_DIR / "data" / "raw" / "generation"/ "Apr_08_2019.csv"),
#         "D:/Projects/ML/solar generation/data/raw/generation/Apr_08_2019.csv",
#         "Resources",
#         overwrite=True
#     )

# # # List files in your project
# # files = dataset_api.list("destination/path/in/hopsworks/")
# # print(files)