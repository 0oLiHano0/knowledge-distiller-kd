#!/bin/bash
# 脚本用于完全重置数据库和迁移状态
# 使用方法：从项目根目录运行 ./scripts/reset_database.sh

set -e  # 遇到错误立即退出

# 删除现有数据库
echo "删除现有数据库..."
rm -f data/kd_tool.db

# 确保数据目录存在
mkdir -p data

# 创建一个空的数据库
echo "创建新的空数据库..."
touch data/kd_tool.db

# 重置Alembic版本
echo "重置Alembic基础版本..."
alembic stamp base

# 执行迁移
echo "执行数据库迁移..."
alembic upgrade head

# 检查表是否存在
echo "验证数据库表结构..."
TABLES=$(sqlite3 data/kd_tool.db ".tables" | tr -s ' ' '\n' | sort)
EXPECTED_TABLES=("alembic_version" "analyses" "blocks" "decisions" "documents")

# 将表名输出到控制台
echo "数据库中的表:"
echo "$TABLES"

# 验证所有预期的表都存在
for table in "${EXPECTED_TABLES[@]}"; do
    if ! echo "$TABLES" | grep -q "$table"; then
        echo "错误: 缺少表 $table"
        exit 1
    fi
done

echo "✅ 数据库重置和迁移成功! 所有表都已正确创建。" 