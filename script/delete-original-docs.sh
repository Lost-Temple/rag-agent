#!/bin/bash

# 指定要清除的目标目录
TARGET_DIR="../data/original_documents/"

# 检查目标目录是否存在
if [ -d "$TARGET_DIR" ]; then
    # 删除目标目录下的所有文件和子目录
    rm -rf "$TARGET_DIR"/*
    echo "已清除 $TARGET_DIR 下的所有文件和文件夹。"
else
    echo "目录 $TARGET_DIR 不存在。"
fi