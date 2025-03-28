# import hopsworks
# import pandas as pd
# import polars as pl
# from hsfs import embedding
# from loguru import logger
# from src.config import settings

# class HopsworksFeatureStore:
#     def __init__(self):
#         """Initialize Hopsworks Feature Store connection."""
#         # self.project, self.feature_store = self._get_feature_store()
#         pass

#     def get_feature_store(self):
#         """Authenticate and connect to Hopsworks Feature Store."""
#         if settings.HOPSWORKS_API_KEY:
#             logger.info("Logging into Hopsworks using HOPSWORKS_API_KEY env var.")
#             project = hopsworks.login(
#                 api_key_value=settings.HOPSWORKS_API_KEY.get_secret_value()
#             )
#         else:
#             logger.info("Logging into Hopsworks using cached API key.")
#             project = hopsworks.login()

#         return project, project.get_feature_store()

#     def create_raw_data_feature_group(self, df: pd.DataFrame, online_enabled: bool = True):
#         """Create or get the raw_data feature group and insert data."""
#         raw_data_fg = self.feature_store.get_or_create_feature_group(
#             name="raw_data",
#             description="Raw data including generation and weather data",
#             version=1,
#             online_enabled=online_enabled,
#         )

#         # Insert Data
#         raw_data_fg.insert(df, wait=True)

#         return raw_data_fg
    
#     def upload_fg():
#         project, fs = feature_store.get_feature_store()
#         raw_df = pd.read_csv("../../data/processed/01_combined_data.csv")
#         logger.info("Uploading 'raw_data' Feature Group to Hopsworks.")
#         raw_data_fg = feature_store.create_customers_feature_group(
#             fs, df=raw_df, online_enabled=True
#         )

#         logger.info("✅ Uploaded 'raw_data' Feature Group to Hopsworks!")

# import hopsworks
# import pandas as pd
# from hsfs import embedding
# from loguru import logger
# from src.config import settings


# class HopsworksFeatureStore:
#     def __init__(self):
#         """Initialize Hopsworks Feature Store connection."""
#         self.project, self.feature_store = self.get_feature_store()
#         self.raw_df = pd.read_csv("D:/Projects/ML/solar generation/data/processed/01_combined_data.csv")  

#     def get_feature_store(self):
#         """Authenticate and connect to Hopsworks Feature Store."""
#         if settings.HOPSWORKS_API_KEY:
#             logger.info("Logging into Hopsworks using HOPSWORKS_API_KEY env var.")
#             project = hopsworks.login(
#                 api_key_value=settings.HOPSWORKS_API_KEY.get_secret_value()
#             )
#         else:
#             logger.info("Logging into Hopsworks using cached API key.")
#             project = hopsworks.login()

#         return project, project.get_feature_store()

#     def create_raw_data_feature_group(fs, df: pd.DataFrame, online_enabled: bool = True):
#         """Create or get the raw_data feature group and insert data."""
#         raw_data_fg = fs.get_or_create_feature_group(
#             name="raw_data",
#             description="Raw data including generation and weather data",
#             primary_key=["datetime"],
#             version=1,
#             online_enabled=online_enabled,
#         )

#         # Insert Data
#         raw_data_fg.insert(df, wait=True)

#         return raw_data_fg

#     def upload_fg(self):
#         """Uploads a CSV dataset as a feature group."""
#         logger.info("Uploading 'raw_data' Feature Group to Hopsworks.")

#         # Use self.raw_df instead of raw_df
#         raw_data_fg = self.create_raw_data_feature_group(df=self.raw_df, online_enabled=True)

#         logger.info("✅ Uploaded 'raw_data' Feature Group to Hopsworks!")  

import hopsworks
import pandas as pd
from hsfs import embedding
from loguru import logger
from src.config import settings

class HopsworksFeatureStore:
    def __init__(self):
        """Initialize Hopsworks Feature Store connection."""
        self.project, self.feature_store = self.get_feature_store()
        self.raw_df = pd.read_csv("data/processed/01_combined_data.csv").head(10050)
        
    def get_feature_store(self):
        """Authenticate and connect to Hopsworks Feature Store."""
        if settings.HOPSWORKS_API_KEY:
            logger.info("Logging into Hopsworks using HOPSWORKS_API_KEY env var.")
            project = hopsworks.login(
                api_key_value=settings.HOPSWORKS_API_KEY.get_secret_value()
            )
        else:
            logger.info("Logging into Hopsworks using cached API key.")
            project = hopsworks.login()
            
        return project, project.get_feature_store()
    
    def create_raw_data_feature_group(self, df: pd.DataFrame, online_enabled: bool = True):
        """Create or get the raw_data feature group and insert data."""
        # Use self.feature_store instead of fs parameter
        raw_data_fg = self.feature_store.get_or_create_feature_group(
            name="raw_data",
            description="Raw data including generation and weather data",
            primary_key=["datetime"],
            version=1,
            online_enabled=online_enabled,
        )
        
        # Insert Data
        raw_data_fg.insert(df, wait=True)
        
        return raw_data_fg
    
    def upload_fg(self):
        """Uploads a CSV dataset as a feature group."""
        logger.info("Uploading 'raw_data' Feature Group to Hopsworks.")
        
        # Call create_raw_data_feature_group as an instance method
        raw_data_fg = self.create_raw_data_feature_group(df=self.raw_df, online_enabled=True)
        
        logger.info("✅ Uploaded 'raw_data' Feature Group to Hopsworks!")

    def get_feature_group(self, name, version=1):
        """
        Get a feature group from the feature store.
        
        Args:
            name (str): Name of the feature group
            version (int): Version of the feature group
            
        Returns:
            Feature group object
        """
        try:
            return self.feature_store.get_feature_group(name=name, version=version)
        except Exception as e:
            logger.error(f"Error retrieving feature group '{name}': {str(e)}")
            raise

    def read_feature_data(self, name, version=1):
        """
        Read data from a feature group.
        
        Args:
            name (str): Name of the feature group
            version (int): Version of the feature group
            
        Returns:
            pd.DataFrame: DataFrame containing the feature data
        """
        try:
            fg = self.get_feature_group(name, version)
            return fg.read()
        except Exception as e:
            logger.error(f"Error reading data from feature group '{name}': {str(e)}")
            raise