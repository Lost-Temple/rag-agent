#!/bin/bash

# 设置镜像名称和标签
IMAGE_NAME="rag-agent"
TAG="latest"

# 获取当前脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# 获取项目根目录
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 创建 .dockerignore 文件以排除不需要的目录
cat > "$PROJECT_ROOT/.dockerignore" << 'EOF'
.idea/
.pytest_cache/
.venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
.git/
.gitignore
EOF

# 构建Docker镜像，指定Dockerfile路径
docker build -t $IMAGE_NAME:$TAG -f "$SCRIPT_DIR/Dockerfile" .

# 检查构建是否成功
if [ $? -eq 0 ]; then
    echo "Docker镜像构建成功: $IMAGE_NAME:$TAG"
    echo "运行以下命令启动容器:"
    echo "docker run -p 8000:8000 -v $PROJECT_ROOT/.env:/app/.env $IMAGE_NAME:$TAG"
else
    echo "Docker镜像构建失败"
    exit 1
fi