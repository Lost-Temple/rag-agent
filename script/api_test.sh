#!/bin/bash
cd /Users/mao/work/yunphant/rag_agent

# 检查并安装必要的pytest插件
echo "检查必要的pytest插件..."
if ! pip list | grep -q pytest-html; then
    echo "安装pytest-html插件..."
    pip install pytest-html
fi

# 创建必要的目录
mkdir -p tests/api tests/models tests/integration

# 创建输出目录
mkdir -p tests/output

# 获取当前时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_NAME="api_test_${TIMESTAMP}.html"
REPORT_PATH="tests/output/${REPORT_NAME}"

# 运行所有测试
echo "运行所有测试..."
pytest -v

# 运行特定模块的测试
echo "运行API模块测试..."
pytest -v tests/api/

# 运行特定测试文件
echo "运行文档上传测试..."
pytest -v tests/api/test_document_upload.py

# 运行特定测试函数
echo "运行特定测试函数..."
pytest -v tests/api/test_document_upload.py::test_upload_document

# 运行并生成HTML报告
echo "运行测试并生成HTML报告..."
echo "报告将保存到: ${REPORT_PATH}"
pytest --html="${REPORT_PATH}"

echo "测试完成，HTML报告已生成: ${REPORT_PATH}"