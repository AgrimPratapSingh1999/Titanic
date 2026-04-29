import logging 
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime


LOG_DIR ='logs'
LOG_FILE = f"{datetime.now().strptime('%m_%d_%Y_%H_%M_%S')}.log"
log_path = os.path.join (os.getcwd(),LOG_DIR,LOG_FILE)
os.makedirs(log_path,exist_ok=True)

LOG_FILE_PATH = os.path.join(log_path,LOG_FILE)
logging.basicConfig(
    filename= LOG_FILE_PATH,
    format= "[%(asctime)s]%(lineno)d %(name)s - %(levelname)s - %(message)s",
    level = logging.INFO,
    
)