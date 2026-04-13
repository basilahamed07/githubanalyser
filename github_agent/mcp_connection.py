import asyncio
import logging
from langchain_mcp_adapters.client import MultiServerMCPClient
from runtime_context import resolve_git_connection

logger = logging.getLogger(__name__)


def get_mcp_client(mcp_url: str, mcp_api_key: str):
    logger.info(f"[MCP] Creating client for URL: {mcp_url}")
    client = MultiServerMCPClient(
        {
            "github": {
                "url": mcp_url,
                "transport": "streamable_http",
                "headers": {
                    "Authorization": f"Bearer {mcp_api_key}"
                }
            }
        }
    )
    logger.info("[MCP] Client created successfully")
    return client


def get_mcp_client_from_connection(connection, *, mcp_url: str | None = None, github_token: str | None = None):
    resolved = resolve_git_connection(connection, github_token=github_token, mcp_url=mcp_url)
    return get_mcp_client(resolved.mcp_url, resolved.access_token)


async def get_tools(mcp_url: str, mcp_api_key: str):
    logger.info("[MCP] Fetching tools from MCP server...")
    client = get_mcp_client(mcp_url, mcp_api_key)
    tools = await client.get_tools()
    logger.info(f"[MCP] Tools loaded: {[t.name for t in tools]}")
    return tools


async def get_tools_for_connection(connection, *, mcp_url: str | None = None, github_token: str | None = None):
    logger.info("[MCP] Fetching tools using GitConnection context...")
    client = get_mcp_client_from_connection(connection, mcp_url=mcp_url, github_token=github_token)
    tools = await client.get_tools()
    logger.info(f"[MCP] Tools loaded: {[t.name for t in tools]}")
    return tools


def get_tools_sync(mcp_url: str, mcp_api_key: str):
    logger.info("[MCP] Running async get_tools in sync context")
    return asyncio.run(get_tools(mcp_url, mcp_api_key))


def get_tools_for_connection_sync(connection, *, mcp_url: str | None = None, github_token: str | None = None):
    logger.info("[MCP] Running async GitConnection tool load in sync context")
    return asyncio.run(get_tools_for_connection(connection, mcp_url=mcp_url, github_token=github_token))
