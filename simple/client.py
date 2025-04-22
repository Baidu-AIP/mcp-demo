#!/usr/bin/env python3
# -*-coding:utf-8 -*-
"""
@Time    :   2025/04/08 13:27:54
@Desc    :   borrowed many code from https://github.com/modelcontextprotocol/python-sdk/blob/main/examples/clients/simple-chatbot/mcp_simple_chatbot/main.py
"""

import asyncio
import json
import logging
import os
import shutil
from contextlib import AsyncExitStack
from typing import Any
import base64

import httpx
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult
from mcp.client.sse import sse_client
from pathlib import Path
from typing import Union, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

class ToolNotFoundError(Exception):
    """Exception raised when a tool is not found."""
    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        super().__init__(f"Tool {tool_name} not found")


class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        self.api_key = os.getenv("LLM_API_KEY")
        self.base_url = os.getenv("LLM_BASE_URL")
        self.model = os.getenv("LLM_MODEL")

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str) -> dict[str, Any]:
        """Load server configuration from JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
        """
        with open(file_path, "r") as f:
            return json.load(f)

    @property
    def llm_api_key(self) -> str:
        """Get the LLM API key.

        Returns:
            The API key as a string.

        Raises:
            ValueError: If the API key is not found in environment variables.
        """
        if not self.api_key:
            raise ValueError("LLM_API_KEY not found in environment variables")
        return self.api_key
    
    @property
    def llm_base_url(self) -> str:
        """Get the LLM base URL.

        Returns:
            The base URL as a string.
        """
        if not self.base_url:
            raise ValueError("LLM_BASE_URL not found in environment variables")
        return self.base_url
    
    @property
    def llm_model(self) -> str:
        """Get the LLM model.

        Returns:
            The model as a string.
        """
        if not self.model:
            raise ValueError("LLM_MODEL not found in environment variables")
        return self.model

class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name: str = name
        self.config: dict[str, Any] = config
        self.stdio_context: Any | None = None
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        """Initialize the server connection."""
        if self.config.get("command"): 
            command = (
                shutil.which("npx")
                if self.config["command"] == "npx"
                else self.config["command"]
            )
            if command is None:
                raise ValueError("The command must be a valid string and cannot be None.")

            server_params = StdioServerParameters(
                command=command,
                args=self.config["args"],
                env={**os.environ, **self.config["env"]}
                if self.config.get("env")
                else None,
            )
            try:
                stdio_transport = await self.exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
                read, write = stdio_transport
                session = await self.exit_stack.enter_async_context(
                    ClientSession(read, write)
                )
                await session.initialize()
                self.session = session
            except Exception as e:
                logging.error(f"Error initializing server {self.name}: {e}")
                await self.cleanup()
                raise
        elif self.config.get("url"):
            try:
                sse_transport = await self.exit_stack.enter_async_context(
                    sse_client(self.config["url"])
                )
                read, write = sse_transport
                session = await self.exit_stack.enter_async_context(
                    ClientSession(read, write)
                )
                await session.initialize()
                self.session = session
            except Exception as e:
                logging.error(f"Error initializing server {self.name}: {e}")
                await self.cleanup()
                raise
        else:
            raise ValueError("Either 'command' or 'url' must be specified in the configuration.")

    async def list_tools(self) -> list[Any]:
        """List available tools from the server.

        Returns:
            A list of available tools.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        tools_response = await self.session.list_tools()
        tools = []

        for item in tools_response:
            if isinstance(item, tuple) and item[0] == "tools":
                for tool in item[1]:
                    tools.append(Tool(tool.name, tool.description, tool.inputSchema))

        return tools

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        retries: int = 2,
        delay: float = 1.0,
    ) -> Any:
        """Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        attempt = 0
        while attempt < retries:
            try:
                logging.info(f"Executing {tool_name}...")
                result = await self.session.call_tool(tool_name, arguments)

                return result

            except Exception as e:
                attempt += 1
                logging.warning(
                    f"Error executing tool: {e}. Attempt {attempt} of {retries}."
                )
                if attempt < retries:
                    logging.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logging.error("Max retries reached. Failing.")
                    raise

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
                self.stdio_context = None
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")


class Tool:
    """Represents a tool with its properties and formatting."""

    def __init__(
        self, name: str, description: str, input_schema: dict[str, Any]
    ) -> None:
        self.name: str = name
        self.description: str = description
        self.input_schema: dict[str, Any] = input_schema

    def format_for_llm(self) -> str:
        """Format tool information for LLM.

        Returns:
            A formatted string describing the tool.
        """
        args_desc = []
        if "properties" in self.input_schema:
            for param_name, param_info in self.input_schema["properties"].items():
                arg_desc = (
                    f"- {param_name}: {param_info.get('description', 'No description')}"
                )
                if param_name in self.input_schema.get("required", []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)

        return f"""
Tool: {self.name}
Description: {self.description}
Arguments:
{chr(10).join(args_desc)}
"""

class LLMClient:
    """Manages communication with the LLM provider."""

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.api_key: str = api_key
        self.base_url: str = base_url
        self.model: str = model

    def get_response(self, messages: list[dict[str, str]]) -> str:
        """Get a response from the LLM.

        Args:
            messages: A list of message dictionaries.

        Returns:
            The LLM's response as a string.

        Raises:
            httpx.RequestError: If the request to the LLM fails.
        """
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        payload = {
            "messages": messages,
            "model": self.model,
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 1,
            "stream": False,
            "stop": None,
        }

        try:
            response = client.chat.completions.create(**payload)
            return response.choices[0].message.content
        except Exception as e:
            error_message = f"Error getting LLM response: {str(e)}"
            logging.error(error_message)
            return error_message


class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    # placeholder for file bytes and file base64 in tool call arguments
    PLACEHOLDER_FILE_BYTES = "<<file_bytes>>"
    PLACEHOLDER_FILE_BASE64 = "<<file_base64>>"

    # system prompt template
    # this prompt does not require the llm has the capability of tool calling, feel free to use all kind of models
    SYSTEM_PROMPT_TEMPLATE = (
                    "你是一个智能的助手，可以使用以下工具：\n\n"
                    "{tools_description}\n\n"
                    "请根据用户的问题选择合适的工具, 每次最多只能输出一个工具, 请严格按照上面工具描述中的参数要求来填写参数"
                    "如果不需要使用工具，请直接回答。\n\n"
                    "注意：当你需要使用工具时，你必须只响应以下JSON对象格式，不要添加其他内容：\n"
                    "{{\n"
                    '    "tool": "tool-name",\n'
                    '    "arguments": {{\n'
                    '        "argument-name": "value"\n'
                    "    }}\n"
                    "}}\n\n"
                    "当收到工具的执行结果时：\n"
                    "1. 将原始数据转换为自然、流畅的对话式回答\n"
                    "2. 保持回答简洁但信息丰富\n"
                    "3. 专注于最相关信息\n"
                    "4. 基于用户问题的上下文来回答\n"
                    "5. 避免简单重复原始数据\n\n"
                    "请只使用上面明确提供的工具，不要编造工具"
                )

    def __init__(self, servers: list[Server], llm_client: LLMClient) -> None:
        self.servers: list[Server] = servers
        self.llm_client: LLMClient = llm_client
        self.history: list[dict] = []
        self.messages: list[dict] = []
        self.tools: dict[str, list[Tool]] = {}

        self.attached_file: bytes = None
    
    @classmethod
    def create(cls, config_file: Union[str, Path], servers: list[str]) -> "ChatSession":
        """Create a ChatSession instance from a configuration file and a list of servers.

        Args:
            config_file: The path to the configuration file.
            servers: A list of server names. If not provided, all servers will be used.

        Returns:
            A ChatSession instance.
        """
        config = Configuration()
        server_config = config.load_config(config_file)
        servers = [
            Server(name, srv_config)
            for name, srv_config in server_config["mcpServers"].items() 
            if not servers or name in servers  
        ]
        llm_client = LLMClient(config.llm_api_key, config.base_url, config.model)
        return cls(servers, llm_client)

    async def refresh_tools(self):
        """Refresh tools from all servers."""
        self.tools = {}
        for server in self.servers:
            self.tools[server.name] = await server.list_tools()
    
    async def reset_session(self):
        """reset session history and messages
        Args:
            attach_file: file bytes to be attached to the session
        """
        self.history = []
        self.messages = []
        # format tools description
        descriptions = []
        for server in self.servers:
            for tool in self.tools[server.name]:
                descriptions.append(tool.format_for_llm())
        tools_description = "\n".join(descriptions)
        # construct system message
        system_message = self.SYSTEM_PROMPT_TEMPLATE.format(tools_description=tools_description)
        self.messages = [{"role": "system", "content": system_message}]
    
    async def start_session(self):
        """start a new session
        """
        for server in self.servers:
            await server.initialize()
        # refresh tools if not initialized
        if not self.tools:
            await self.refresh_tools()
        await self.reset_session()

    async def close_session(self):
        """close session
        """
        for server in self.servers:
            await server.cleanup()
        self.history.extend(self.messages)
        self.messages = []

    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        cleanup_tasks = []
        for server in self.servers:
            cleanup_tasks.append(asyncio.create_task(server.cleanup()))

        if cleanup_tasks:
            try:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            except Exception as e:
                logging.warning(f"Warning during final cleanup: {e}")

    async def execute_tool(
            self,
            tool_name: str,
            arguments: dict[str, Any],
            retries: int = 2,
            delay: float = 1.0,
        ) -> Any:
        """Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.
        """
        arguments = arguments or {}
        for server in self.servers:
            tools = self.tools[server.name]
            if any(tool.name == tool_name for tool in tools):
                result = await server.execute_tool(tool_name, arguments, retries, delay)
                if isinstance(result, dict) and "progress" in result:
                    progress = result["progress"]
                    total = result["total"]
                    percentage = (progress / total) * 100
                    logging.info(
                        f"Progress: {progress}/{total} "
                        f"({percentage:.1f}%)"
                    )
                return result
        raise ToolNotFoundError(tool_name)

    async def process_llm_response(self, 
                                   llm_response: str, 
                                   refresh_tools: bool = False,
                                   file_bytes: bytes = None) -> Union[str, CallToolResult]:
        """Process the LLM response and execute tools if needed.

        Args:
            llm_response: The response from the LLM.
            refresh_tools: Whether to refresh tools before executing the tool.

        Returns:
            The result of tool execution or the original response.
        """
        if refresh_tools:
            await self.refresh_tools()
        try:
            # if json decode success, it means the response is a tool call
            tool_call = json.loads(llm_response)
            if "tool" in tool_call:
                logging.info(f"Executing tool: {tool_call['tool']}")
                logging.info(f"With arguments: {tool_call['arguments']}")
                for key, value in tool_call["arguments"].items():
                    if value == self.PLACEHOLDER_FILE_BYTES:
                        if file_bytes:
                            tool_call["arguments"][key] = file_bytes
                        else:
                            logging.error("No file bytes provided")
                            return "No file provided"
                    elif value == self.PLACEHOLDER_FILE_BASE64:
                        if file_bytes:
                            tool_call["arguments"][key] = base64.b64encode(file_bytes)
                        else:
                            logging.error("No file provided")
                            return "No file provided"
                try:
                    result = await self.execute_tool(tool_call["tool"], tool_call["arguments"])
                    logging.info(f"Tool execution result: {result}")
                    return result
                except ToolNotFoundError as e:
                    logging.error(f"Tool {tool_call['tool']} not found")
                    return f"Tool {tool_call['tool']} not found"
                except Exception as e:
                    error_msg = f"Error executing tool: {str(e)}"
                    logging.error(error_msg)
                    return error_msg
            return llm_response
        except json.JSONDecodeError:
            # if not tool call, return the original response
            return llm_response