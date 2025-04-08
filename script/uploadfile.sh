#!/bin/bash

# 检查是否提供了文件路径
if [ -z "$1" ]; then
  echo "Usage: $0 /path/to/file_full_name"
  exit 1
fi

FILE_PATH="$1"

# 使用 curl 上传文件
curl -X POST http://localhost:8009/documents/upload \
  -F "file=@${FILE_PATH}"