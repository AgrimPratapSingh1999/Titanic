import logging 
import os
from datetime import datetime


#LOG_DIR ='logs'
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_path = os.path.join (os.getcwd(),"logs",LOG_FILE)
os.makedirs(os.path.dirname(log_path),exist_ok=True)

#LOG_FILE_PATH = os.path.join(log_path,LOG_FILE)
logging.basicConfig(
    format= "[%(asctime)s]%(lineno)d %(name)s - %(levelname)s - %(message)s",
    level = logging.INFO,
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)
