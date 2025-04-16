# server.py
from mcp.server.fastmcp import FastMCP
from typing import Dict, Union
import asyncio
# 创建MCP服务器实例
mcp = FastMCP("ScoreQueryServer")

# 学生成绩数据
scores: Dict[str, int] = {
    "语文": 85,
    "数学": 90,
    "英语": 88,
}

@mcp.tool()
async def get_score(subject: str) -> Union[str, Dict[str, int]]:
    """
    查询指定科目的分数
    
    Args:
        subject: 要查询的科目名称
        
    Returns:
        如果科目存在，返回包含分数的字典
        如果科目不存在，返回错误信息字符串
    """
    if subject in scores:
        return {subject: scores[subject]}
    return f"错误：没有找到科目 '{subject}' 的成绩"

if __name__ == "__main__":
    try:
        print("成绩查询服务正在启动，等待连接...")
        mcp.run(transport='stdio')
    except Exception as e:
        print(f"服务器启动失败: {str(e)}")