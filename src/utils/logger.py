import logging
import os
from logging.handlers import RotatingFileHandler
from src.config import settings

# 确保日志目录存在
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)

# 日志文件路径
log_file = os.path.join(log_dir, 'rag_agent.log')

# 创建日志记录器
logger = logging.getLogger('rag_agent')
logger.setLevel(logging.INFO)

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 创建文件处理器（使用RotatingFileHandler以防止日志文件过大）
file_handler = RotatingFileHandler(
    log_file,
    maxBytes=settings.log_file_max_bytes,  # 从配置中读取，默认10MB
    backupCount=settings.log_file_backup_count,  # 从配置中读取，默认5
    encoding='utf-8'
)
file_handler.setLevel(logging.INFO)

# 创建格式化器
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# 添加处理器到记录器
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# 提供不同级别的日志函数
def debug(msg, stacklevel=2, *args, **kwargs):
    logger.debug(msg=msg, stacklevel=stacklevel, *args, **kwargs)

def info(msg, stacklevel=2, *args, **kwargs):
    logger.info(msg=msg, stacklevel=stacklevel, *args, **kwargs)

def warning(msg, stacklevel=2, *args, **kwargs):
    logger.warning(msg=msg, stacklevel=stacklevel, *args, **kwargs)

def error(msg, stacklevel=2, *args, **kwargs):
    logger.error(msg=msg, stacklevel=stacklevel, *args, **kwargs)

def critical(msg, stacklevel=2, *args, **kwargs):
    logger.critical(msg=msg, stacklevel=stacklevel, *args, **kwargs)