from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
from urllib.parse import urlparse


DEFAULT_GITHUB_MCP_URL = "https://api.githubcopilot.com/mcp/"


@dataclass(frozen=True)
class AzureOpenAIConfig:
    azure_endpoint: str
    azure_api_key: str
    deployment_name: str
    api_version: str
    model_name: str


@dataclass(frozen=True)
class LLMConfig:
    llm_provider: str
    model_name: str
    azure_endpoint: str | None = None
    azure_api_key: str | None = None
    deployment_name: str | None = None
    api_version: str | None = None
    groq_api_key: str | None = None


@dataclass(frozen=True)
class GitHubConnectionConfig:
    git_id: int | None
    application_id: int | None
    customer_id: int | None
    group_id: int | None
    repo_url: str
    repo_owner: str
    repo_name: str
    auth_type: str | None
    username: str | None
    access_token: str | None
    default_branch: str | None
    branch: str | None
    cloned_path: str | None
    cloned_status: int | None
    mcp_url: str

    @property
    def repo_full_name(self) -> str:
        return f"{self.repo_owner}/{self.repo_name}"

    @property
    def active_branch(self) -> str | None:
        return self.branch or self.default_branch

    def safe_metadata(self) -> dict[str, Any]:
        return {
            "git_id": self.git_id,
            "application_id": self.application_id,
            "customer_id": self.customer_id,
            "group_id": self.group_id,
            "repo_url": self.repo_url,
            "repo_owner": self.repo_owner,
            "repo_name": self.repo_name,
            "repo_full_name": self.repo_full_name,
            "auth_type": self.auth_type,
            "github_username": self.username,
            "default_branch": self.default_branch,
            "branch": self.branch,
            "cloned_path": self.cloned_path,
            "cloned_status": self.cloned_status,
            "mcp_url": self.mcp_url,
        }


def _read_value(source: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(source, dict):
        return source.get(field_name, default)
    return getattr(source, field_name, default)


def parse_github_repo_url(repo_url: str) -> tuple[str, str]:
    normalized = (repo_url or "").strip()
    if not normalized:
        raise ValueError("repo_url is required to resolve the repository")

    if normalized.endswith(".git"):
        normalized = normalized[:-4]

    if normalized.startswith("git@github.com:"):
        normalized = normalized.replace("git@github.com:", "https://github.com/", 1)

    parsed = urlparse(normalized)
    if (parsed.hostname or "").lower() != "github.com":
        raise ValueError(f"Unsupported GitHub URL: {repo_url}")

    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        raise ValueError(f"Unable to parse owner/repo from repo_url: {repo_url}")

    return parts[0], parts[1]


def resolve_azure_openai_config(config: Any) -> AzureOpenAIConfig:
    azure_endpoint = _read_value(config, "azure_endpoint")
    azure_api_key = _read_value(config, "azure_api_key")
    deployment_name = _read_value(config, "deployment_name")
    api_version = _read_value(config, "api_version")
    model_name = _read_value(config, "model_name") or deployment_name

    missing = [
        name for name, value in [
            ("azure_endpoint", azure_endpoint),
            ("azure_api_key", azure_api_key),
            ("deployment_name", deployment_name),
            ("api_version", api_version),
        ]
        if not value
    ]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing Azure OpenAI settings: {joined}")

    return AzureOpenAIConfig(
        azure_endpoint=azure_endpoint,
        azure_api_key=azure_api_key,
        deployment_name=deployment_name,
        api_version=api_version,
        model_name=model_name,
    )


def resolve_llm_config(config: Any) -> LLMConfig:
    llm_provider = (_read_value(config, "llm_provider") or _read_value(config, "provider") or "azure_openai").strip().lower()

    if llm_provider == "azure_openai":
        azure = resolve_azure_openai_config(config)
        return LLMConfig(
            llm_provider=llm_provider,
            model_name=azure.model_name,
            azure_endpoint=azure.azure_endpoint,
            azure_api_key=azure.azure_api_key,
            deployment_name=azure.deployment_name,
            api_version=azure.api_version,
        )

    if llm_provider == "groq":
        groq_api_key = _read_value(config, "groq_api_key") or _read_value(config, "api_key")
        model_name = _read_value(config, "model_name")
        missing = [
            name for name, value in [
                ("groq_api_key", groq_api_key),
                ("model_name", model_name),
            ]
            if not value
        ]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"Missing Groq settings: {joined}")

        return LLMConfig(
            llm_provider=llm_provider,
            model_name=model_name,
            groq_api_key=groq_api_key,
        )

    raise ValueError("Unsupported llm_provider. Supported values: azure_openai, groq")


def resolve_git_connection(
    connection: Any,
    *,
    github_token: str | None = None,
    mcp_url: str | None = None,
    repo_owner: str | None = None,
    repo_name: str | None = None,
) -> GitHubConnectionConfig:
    repo_url = _read_value(connection, "repo_url", "")
    parsed_owner, parsed_repo = parse_github_repo_url(repo_url)
    resolved_owner = repo_owner or parsed_owner
    resolved_repo = repo_name or parsed_repo

    access_token = github_token or _read_value(connection, "access_token")
    auth_type = _read_value(connection, "auth_type")
    if not access_token and auth_type != "ssh":
        raise ValueError("GitHub access token is required for MCP access")

    return GitHubConnectionConfig(
        git_id=_read_value(connection, "git_id"),
        application_id=_read_value(connection, "application_id"),
        customer_id=_read_value(connection, "customer_id"),
        group_id=_read_value(connection, "group_id"),
        repo_url=repo_url,
        repo_owner=resolved_owner,
        repo_name=resolved_repo,
        auth_type=auth_type,
        username=_read_value(connection, "username"),
        access_token=access_token,
        default_branch=_read_value(connection, "default_branch"),
        branch=_read_value(connection, "branch"),
        cloned_path=_read_value(connection, "cloned_path"),
        cloned_status=_read_value(connection, "cloned_status"),
        mcp_url=mcp_url or DEFAULT_GITHUB_MCP_URL,
    )


def select_git_connection(
    connections: Iterable[Any],
    *,
    git_id: int | None = None,
    application_id: int | None = None,
    customer_id: int | None = None,
    group_id: int | None = None,
    username: str | None = None,
    repo_url: str | None = None,
    repo_owner: str | None = None,
    repo_name: str | None = None,
) -> Any:
    matches = []
    normalized_repo_url = (repo_url or "").strip().lower()
    normalized_username = (username or "").strip().lower()
    normalized_repo_owner = (repo_owner or "").strip().lower()
    normalized_repo_name = (repo_name or "").strip().lower()

    for connection in connections:
        if git_id is not None and _read_value(connection, "git_id") != git_id:
            continue
        if application_id is not None and _read_value(connection, "application_id") != application_id:
            continue
        if customer_id is not None and _read_value(connection, "customer_id") != customer_id:
            continue
        if group_id is not None and _read_value(connection, "group_id") != group_id:
            continue

        connection_username = str(_read_value(connection, "username", "")).strip().lower()
        if normalized_username and connection_username != normalized_username:
            continue

        connection_repo_url = str(_read_value(connection, "repo_url", "")).strip().lower()
        if normalized_repo_url and connection_repo_url != normalized_repo_url:
            continue

        if normalized_repo_owner or normalized_repo_name:
            owner, repo = parse_github_repo_url(_read_value(connection, "repo_url", ""))
            if normalized_repo_owner and owner.lower() != normalized_repo_owner:
                continue
            if normalized_repo_name and repo.lower() != normalized_repo_name:
                continue

        matches.append(connection)

    if not matches:
        raise ValueError("No GitConnection row matched the provided filters")

    if len(matches) > 1:
        raise ValueError(
            "Multiple GitConnection rows matched. Filter with git_id or a unique combination like "
            "(customer_id, application_id, repo_url)."
        )

    return matches[0]


def load_git_connection_from_session(
    session: Any,
    git_connection_model: Any,
    *,
    git_id: int | None = None,
    application_id: int | None = None,
    customer_id: int | None = None,
    group_id: int | None = None,
    username: str | None = None,
    repo_url: str | None = None,
    repo_owner: str | None = None,
    repo_name: str | None = None,
) -> Any:
    if not hasattr(session, "query"):
        raise TypeError("session must expose a SQLAlchemy-style query() method")

    query = session.query(git_connection_model)
    if git_id is not None:
        query = query.filter(git_connection_model.git_id == git_id)
    if application_id is not None:
        query = query.filter(git_connection_model.application_id == application_id)
    if customer_id is not None:
        query = query.filter(git_connection_model.customer_id == customer_id)
    if group_id is not None:
        query = query.filter(git_connection_model.group_id == group_id)

    return select_git_connection(
        query.all(),
        git_id=git_id,
        application_id=application_id,
        customer_id=customer_id,
        group_id=group_id,
        username=username,
        repo_url=repo_url,
        repo_owner=repo_owner,
        repo_name=repo_name,
    )
