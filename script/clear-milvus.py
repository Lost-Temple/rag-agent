import sys
import os
# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pymilvus import connections, Collection
from src.config.config import settings

# 连接到 Milvus 服务
connections.connect(alias="default", host=settings.milvus_host, port=settings.milvus_port)

# 指定集合名称
collection_name = settings.milvus_collection

# 获取集合对象
collection = Collection(name=collection_name)

# 删除集合中的所有实体
collection.delete(expr="pk >= 0")

# 提交删除操作
collection.flush()

# 可选：执行压缩操作以永久删除数据
collection.compact()