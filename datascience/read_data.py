from asyncio.log import logger
import pandas as pd

def read_csv(file_path):
    if file_path is not None:
        logger.debug("File is being read! Let us relax like a Boss!")
        return pd.read_csv(file_path)
    else:
        msg="File Not Found"
        return ValueError(msg)
    