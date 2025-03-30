from src import logger
import pandas as pd
# from src.data.make_dataset import MakeDataset 
# from src.data.make_dataframe import MakeDataframe
from src.hopsworks_integration.feature_store import HopsworksFeatureStore
from src.features.remove_outliers import OutlierDetection

# STAGE_NAME = "Make dataset"
# try:
#     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#     make_dataset = MakeDataset()
#     make_dataset.make_generation_csv()
#     make_dataset.make_weather_csv()
#     logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logger.exception(e)
#         raise e


# STAGE_NAME = "Make dataframe"
# try:
#     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#     make_dataframe = MakeDataframe()
#     make_dataframe.make_generation_dataframe()
#     make_dataframe.make_weather_dataframe()
#     make_dataframe.make_combined_dataframe()
#     logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logger.exception(e)
#         raise e


# STAGE_NAME = "Make feature group for raw data"
# try:
#     logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
#     hopsworks_feature_store = HopsworksFeatureStore()
#     hopsworks_feature_store.upload_fg()
    
#     logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#     logger.exception(e)
#     raise e

STAGE_NAME = "Outlier detection"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
    hopsworks_feature_store = HopsworksFeatureStore()
    project, fs = hopsworks_feature_store.get_feature_store()
    raw_data_fg = fs.get_feature_group("raw_data", version=1)
    raw_data_df = raw_data_fg.read()
    output_dir = 'reports/figures' 
    outlier_detector = OutlierDetection(raw_data_df, output_dir)
    outlier_detector.save_boxplots()
    outlier_detector.detect_and_save_outliers()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

