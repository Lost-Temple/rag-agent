import sqlite3

# 连接到 SQLite 数据库
connection = sqlite3.connect('data/database/document_summaries.db')
cursor = connection.cursor()

# 要清空的表名
table_name = 'document_summaries'

# 构建并执行删除语句
delete_query = f'DELETE FROM {table_name};'
cursor.execute(delete_query)

# 提交更改并关闭连接
connection.commit()
connection.close()