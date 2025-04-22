
import os
import time
import asyncio
import argparse
import pathlib
import appbuilder
import dotenv
from appbuilder.core.console.appbuilder_client.async_event_handler import (
    AsyncToolCallEventHandler,
)
import urllib.parse
from appbuilder.mcp_server.client import MCPClient
import base64

dotenv.load_dotenv()

def print_green(text):
    """print text in green color"""
    print(f"\033[92m{text}\033[0m")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", nargs="+", dest="files", help="输入要整理的发票文件，支持jpg/png/jpeg格式")
    return parser.parse_args()


class AsyncToolCallEventHandlerAIP(AsyncToolCallEventHandler):
    """
    工具调用事件处理类
    """
    
    async def interrupt(self, run_context, run_response):
        """
        工具调用事件处理
        """
        print_green("Agent 中间思考: {}\n".format(run_context.current_thought))
        # 遍历工具调用参数，将<<{file_path}>>替换为文件的base64编码
        for tool_call in run_context.current_tool_calls:
            function_arguments = tool_call.function.arguments
            print_green("工具名称: {}, 工具参数: {}".format(tool_call.function.name, function_arguments))
            for key, value in function_arguments.items():
                if value.startswith("<<") and value.endswith(">>"):
                    file_path = value[2:-2]
                    # 获取文件内容
                    with open(file_path, "rb") as f:
                        file_content = f.read()
                    # 将文件内容转换为base64编码
                    file_content_base64 = base64.b64encode(file_content).decode("utf-8")    
                    function_arguments[key] = file_content_base64
        tool_output = await super().interrupt(run_context, run_response)
        print_green("工具输出: {}".format(tool_output))
        return tool_output
    
    def success(self, run_context, run_response):
        """
        agent执行成功回调，输出结果
        """
        print_green("Agent 回答: " + run_response.answer)
        return super().success(run_context, run_response)

async def main():
    """
    主函数入口
    """
    # 获取MCP Server地址、API_KEY和APP_ID
    MCP_SERVER_BASE_URL = os.getenv("MCP_SERVER_BASE_URL")
    API_KEY = os.getenv("API_KEY")
    APP_ID = os.getenv("APP_ID")

    # 检查是否设置了环境变量
    if not MCP_SERVER_BASE_URL or not API_KEY or not APP_ID:
        print("Please set APP_ID, API_KEY and MCP_SERVER_BASE_URL in .env file or system environment variables")
        return

    args = get_args()
    # 创建app builder client
    appbuilder_client = appbuilder.AsyncAppBuilderClient(APP_ID)
    mcp_client = MCPClient()
    # server.py 是上述步骤中下载的mcp组件文件
    # 对API_KEY进行url encode,并拼接到MCP_SERVER_BASE_URL上
    service_url = MCP_SERVER_BASE_URL +  "?Authorization=" + "Bearer+" + API_KEY
    await mcp_client.connect_to_server(service_url=service_url)
    conversation_id = await appbuilder_client.create_conversation()

    files = []
    for f in args.files:
        p = pathlib.Path(f)
        if p.suffix in [".jpg", ".png", ".jpeg"]:
            files.append(f)
        else:
            print(f"文件{f}不是图片文件，跳过")
    if len(files) == 0:
        print("没有上传任何文件")
        return

    tools = mcp_client.tools
    event_handler = AsyncToolCallEventHandlerAIP(mcp_client)
    with await appbuilder_client.run_with_handler(
        conversation_id=conversation_id,
        query="帮我整理这些发票，发票的文件路径分别是：" +
                ", ".join(files) +
                "。 在生成工具调用参数时，如果需要用到文件的base64编码，" +
                """请使用"<<{file_path}>>"来替代, 其中{file_path}为对应文件的文件路径""",
        tools=tools,
        event_handler=event_handler,
    ) as run:
        await run.until_done()

    await appbuilder_client.http_client.session.close()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())