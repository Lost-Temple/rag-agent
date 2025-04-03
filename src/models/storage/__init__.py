# 存储模块初始化文件
from src.models.storage.peewee_store import PeeweeStore as SQLiteStore

# 导出主要类，便于其他模块直接从storage导入