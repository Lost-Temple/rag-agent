# 存储模块初始化文件
# 导入所有存储实现
from src.models.storage.sqlite_store import SQLiteStore
from src.models.storage.mysql_store import MySQLStore

# 导出所有存储类，便于使用
__all__ = ['SQLiteStore', 'MySQLStore']

# 如果需要默认导出SQLiteStore，可以保留这行
# 或者根据配置决定使用哪个存储实现
from src.models.storage.sqlite_store import SQLiteStore as DefaultStore