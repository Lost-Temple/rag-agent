import sys
import os
# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neo4j import GraphDatabase
from src.config.config import settings 

# 创建驱动程序实例
driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))

# 指定要操作的数据库名称
database_name = settings.neo4j_database

# 开始会话并指定数据库
with driver.session(database=database_name) as session:
    # 执行删除所有节点和关系的操作
    session.run("MATCH (n) DETACH DELETE n")

# 关闭驱动程序
driver.close()