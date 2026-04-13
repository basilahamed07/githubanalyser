import logging
from functools import partial
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_openai import AzureChatOpenAI

logger = logging.getLogger(__name__)


def get_llm(azure_endpoint, azure_api_key, deployment_name, api_version):
    logger.info(f"[LLM] Creating AzureChatOpenAI — deployment: {deployment_name}")
    return AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        azure_deployment=deployment_name,
        api_version=api_version,
        temperature=0,
        streaming=False   # must be False for tool calling
    )


def filter_tools(all_tools: list, tool_names: list):
    filtered = [t for t in all_tools if t.name in tool_names]
    logger.debug(f"[TOOLS] Filtered tools: {[t.name for t in filtered]}")
    return filtered


# ─────────────────────────────────────────────
# ASYNC TOOL EXECUTOR HELPER
# ─────────────────────────────────────────────
async def execute_tool_calls(response: AIMessage, all_tools: list) -> list[ToolMessage]:
    """
    Executes all tool calls in an AIMessage asynchronously.
    Returns list of ToolMessages with results.
    """
    tool_map = {t.name: t for t in all_tools}
    tool_messages = []

    if not hasattr(response, "tool_calls") or not response.tool_calls:
        logger.info("[TOOL EXEC] No tool calls in response")
        return tool_messages

    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id   = tool_call["id"]

        logger.info(f"[TOOL EXEC] Calling tool: {tool_name} | args: {tool_args}")

        if tool_name not in tool_map:
            logger.warning(f"[TOOL EXEC] Tool not found: {tool_name}")
            tool_messages.append(ToolMessage(
                content=f"Tool {tool_name} not found",
                tool_call_id=tool_id
            ))
            continue

        try:
            tool = tool_map[tool_name]
            result = await tool.ainvoke(tool_args)   # ASYNC invoke
            logger.info(f"[TOOL EXEC] Tool {tool_name} result length: {len(str(result))}")
            logger.debug(f"[TOOL EXEC] Tool {tool_name} result: {str(result)[:300]}")
            tool_messages.append(ToolMessage(
                content=str(result),
                tool_call_id=tool_id
            ))
        except Exception as e:
            logger.error(f"[TOOL EXEC] Tool {tool_name} failed: {e}")
            tool_messages.append(ToolMessage(
                content=f"Tool {tool_name} error: {str(e)}",
                tool_call_id=tool_id
            ))

    return tool_messages


# ─────────────────────────────────────────────
# NODE: INTENT CLASSIFIER
# ─────────────────────────────────────────────
async def intent_classifier_node(state: dict, llm) -> dict:
    logger.info(f"[INTENT] Classifying query: {state['user_query'][:100]}")

    import json
    system = SystemMessage(content="""
You are an intent classifier for a GitHub repository analysis agent.
Classify the user query into:
1. intent: READ or WRITE or HYBRID
2. domain: list from [REPO, CODE, PR, ISSUES, TEAM, SECURITY, COPILOT, META]
3. complexity: SIMPLE or COMPLEX
4. needs_clarification: true if repo owner/name is missing

Respond ONLY in this exact JSON format, no extra text:
{
  "intent": "READ",
  "domain": ["CODE", "REPO"],
  "complexity": "COMPLEX",
  "needs_clarification": false
}
""")
    human = HumanMessage(content=f"""
Query: {state['user_query']}
Repo Owner: {state.get('repo_owner', 'NOT PROVIDED')}
Repo Name: {state.get('repo_name', 'NOT PROVIDED')}
""")

    response = await llm.ainvoke([system, human])
    logger.info(f"[INTENT] Raw response: {response.content}")

    try:
        result = json.loads(response.content)
    except Exception as e:
        logger.warning(f"[INTENT] JSON parse failed: {e} — using defaults")
        result = {"intent": "READ", "domain": ["CODE"], "complexity": "COMPLEX", "needs_clarification": False}

    logger.info(f"[INTENT] Classified → intent={result.get('intent')} domain={result.get('domain')}")

    return {
        **state,
        "intent": result.get("intent", "READ"),
        "domain": result.get("domain", ["CODE"]),
        "needs_clarification": result.get("needs_clarification", False),
        "loop_count": 0,
        "is_complete": False,
        "tool_calls_made": [],
        "intermediate_results": {},
        "current_step": 0,
        "plan": []
    }


# ─────────────────────────────────────────────
# NODE: ORCHESTRATOR (SUPER AGENT)
# ─────────────────────────────────────────────
async def orchestrator_node(state: dict, llm) -> dict:
    loop_count = state.get("loop_count", 0)
    logger.info(f"[ORCHESTRATOR] Loop #{loop_count} | step={state.get('current_step')} | results={list(state.get('intermediate_results', {}).keys())}")

    if loop_count >= 15:
        logger.warning("[ORCHESTRATOR] Max loops reached — forcing synthesizer")
        return {**state, "next_node": "synthesizer", "is_complete": True}

    import json

    intermediate = state.get("intermediate_results", {})
    has_tree = any("tree_scan" in k for k in intermediate.keys())

    if not has_tree and loop_count == 0:
        logger.info("[ORCHESTRATOR] No tree yet — forcing tree scan first")
        return {
            **state,
            "plan": [{
                "step": 1,
                "node": "code_analyst",
                "action": "Get the full repository file tree by calling get_file_contents with path='' (empty string) for the root directory. List ALL files and folders.",
                "status": "pending",
                "purpose": "tree_scan"
            }],
            "current_step": 1,
            "next_node": "code_analyst",
            "loop_count": loop_count + 1
        }

    system = SystemMessage(content="""
You are the Orchestrator — the Super Agent that controls all other nodes.

CRITICAL RULES:
1. ALWAYS carry forward the FULL plan — never drop previous steps
2. Mark completed steps as "done" — never remove them
3. Add NEW pending steps at the end of the plan
4. current_step must match the step number of the NEXT pending step
5. Use EXACT file paths from the tree scan result — never guess paths
6. For FastAPI: look for files in routers/, api/, endpoints/ folders
7. Read actual .py files to extract @router.get/@router.post decorators
8. Route to synthesizer ONLY when you have enough data

Available nodes:
- code_analyst   → get_file_contents, search_code
- repo_analyst   → branches, commits, tags, releases
- pr_node        → pull requests
- issues_node    → issues
- team_node      → team members
- security_node  → secret scanning
- write_ops      → create/update/delete
- meta_node      → repo discovery
- synthesizer    → final answer

Respond ONLY in this exact JSON, no extra text:
{
  "plan": [
    {"step": 1, "node": "code_analyst", "action": "...", "status": "done", "purpose": "tree_scan"},
    {"step": 2, "node": "code_analyst", "action": "Read backend/README.md", "status": "done", "purpose": "api_docs"},
    {"step": 3, "node": "code_analyst", "action": "Read backend/app/api/v1/endpoints/submissions.py to extract all @router endpoints", "status": "pending", "purpose": "api_routes"}
  ],
  "current_step": 3,
  "next_node": "code_analyst",
  "reasoning": "Tree shows backend/app/api/v1/endpoints/ — reading each router file"
}
""")

    last_result = ""
    if intermediate:
        last_val = list(intermediate.values())[-1]
        last_result = str(last_val)[:2000]

    # ── FIX: pass full existing plan so orchestrator never loses context ──
    existing_plan = state.get("plan", [])

    context = f"""
User Query: {state['user_query']}
Repo: {state.get('repo_owner')}/{state.get('repo_name')}
Intent: {state.get('intent')}
Loop Count: {loop_count}

EXISTING PLAN (carry this forward, mark done steps, add new pending steps):
{existing_plan}

Results Collected (keys): {list(intermediate.keys())}

Last Tool Result (use this to decide next step — contains actual file contents):
{last_result}
"""

    response = await llm.ainvoke([system, HumanMessage(content=context)])
    logger.info(f"[ORCHESTRATOR] Response: {response.content[:800]}")

    try:
        # ── FIX: strip markdown code fences if LLM wraps in ```json ──
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
    except Exception as e:
        logger.warning(f"[ORCHESTRATOR] JSON parse failed: {e} — routing to synthesizer")
        result = {
            "plan": existing_plan,
            "current_step": state.get("current_step", 0),
            "next_node": "synthesizer",
            "reasoning": f"JSON parse error: {e}"
        }

    # ── FIX: validate current_step matches a real pending step in plan ──
    new_plan = result.get("plan", existing_plan)
    new_step = result.get("current_step", 0)
    next_node = result.get("next_node", "synthesizer")

    # Double-check: if next_node is not synthesizer, the step must exist in plan
    if next_node != "synthesizer":
        step_exists = any(item.get("step") == new_step for item in new_plan)
        if not step_exists:
            logger.warning(f"[ORCHESTRATOR] current_step={new_step} not found in plan — routing to synthesizer")
            next_node = "synthesizer"

    logger.info(f"[ORCHESTRATOR] → next_node={next_node} | step={new_step} | reasoning={result.get('reasoning')}")

    return {
        **state,
        "plan": new_plan,
        "current_step": new_step,
        "next_node": next_node,
        "loop_count": loop_count + 1
    }

# ─────────────────────────────────────────────
# GENERIC SPECIALIST NODE RUNNER
# ─────────────────────────────────────────────
async def run_specialist_node(
    state: dict,
    llm,
    all_tools: list,
    node_name: str,
    tool_names: list,
    system_prompt: str
) -> dict:
    logger.info(f"[{node_name.upper()}] Starting | step={state.get('current_step')}")

    node_tools = filter_tools(all_tools, tool_names)
    llm_with_tools = llm.bind_tools(node_tools)

    current_step = state.get("current_step", 0)
    plan = state.get("plan", [])

    # ── FIX: find plan item by step number, not by index ──
    current_plan_item = {}
    for item in plan:
        if item.get("step") == current_step:
            current_plan_item = item
            break

    action  = current_plan_item.get("action", "")
    purpose = current_plan_item.get("purpose", "")

    # Only fall back to user_query if action is truly missing
    if not action:
        logger.warning(f"[{node_name.upper()}] No plan action found for step {current_step} — falling back to user_query")
        action = state["user_query"]

    logger.info(f"[{node_name.upper()}] Action : {action}")
    logger.info(f"[{node_name.upper()}] Purpose: {purpose}")

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=action)]

    response = await llm_with_tools.ainvoke(messages)
    logger.info(f"[{node_name.upper()}] LLM response — tool_calls: {len(getattr(response, 'tool_calls', []))}")

    tool_messages = await execute_tool_calls(response, all_tools)

    all_content = response.content or ""
    for tm in tool_messages:
        all_content += f"\n\nTool Result:\n{tm.content}"
        logger.info(f"[{node_name.upper()}] Tool result collected, length={len(tm.content)}")

    result_key = f"{purpose}_{node_name}_step_{current_step}" if purpose else f"{node_name}_step_{current_step}"

    results = state.get("intermediate_results", {})
    results[result_key] = all_content

    logger.info(f"[{node_name.upper()}] Done | stored key: {result_key} | content length: {len(all_content)}")

    return {
        **state,
        "messages": state.get("messages", []) + [response] + tool_messages,
        "intermediate_results": results,
        "tool_calls_made": state.get("tool_calls_made", []) + [node_name]
    }
# ─────────────────────────────────────────────
# ALL SPECIALIST NODES
# ─────────────────────────────────────────────
async def repo_analyst_node(state: dict, llm, all_tools: list) -> dict:
    return await run_specialist_node(
        state, llm, all_tools,
        node_name="repo_analyst",
        tool_names=["list_branches", "list_commits", "get_commit",
                    "list_tags", "list_releases", "get_latest_release",
                    "get_release_by_tag", "get_tag"],
        system_prompt=f"""You are the Repo Analyst specialist.
Repo: {state.get('repo_owner')}/{state.get('repo_name')}
Tools: list_branches, list_commits, get_commit, list_tags,
       list_releases, get_latest_release, get_release_by_tag, get_tag
Use the most appropriate tool for the given action."""
    )


async def code_analyst_node(state: dict, llm, all_tools: list) -> dict:
    current_step = state.get("current_step", 0)
    plan = state.get("plan", [])
    current_plan_item = plan[current_step - 1] if plan and current_step > 0 and current_step <= len(plan) else {}
    purpose = current_plan_item.get("purpose", "")

    # Special system prompt for tree scan step
    if purpose == "tree_scan":
        system_prompt = f"""You are the Code Analyst. Your ONLY job right now is to get the full repo file tree.

Repo: {state.get('repo_owner')}/{state.get('repo_name')}

Call get_file_contents with:
- owner: {state.get('repo_owner')}
- repo: {state.get('repo_name')}
- path: ""   (empty string = root)

This will return ALL files and folders. List everything you see — folder names and file names.
This is critical so the orchestrator can plan which files to read next."""
    else:
        system_prompt = f"""You are the Code Analyst specialist.
Repo: {state.get('repo_owner')}/{state.get('repo_name')}

Tools available:
- get_file_contents(owner, repo, path): Read a file or list a directory
- search_code(query, owner, repo): Search patterns across codebase

IMPORTANT:
- Use EXACT file paths from the repo tree
- For directories, call get_file_contents to list their contents first
- For API endpoints: read the actual router/route files line by line
- Do NOT summarize — return the full raw content"""

    return await run_specialist_node(
        state, llm, all_tools,
        node_name="code_analyst",
        tool_names=["get_file_contents", "search_code"],
        system_prompt=system_prompt
    )


async def pr_node(state: dict, llm, all_tools: list) -> dict:
    return await run_specialist_node(
        state, llm, all_tools,
        node_name="pr_node",
        tool_names=["list_pull_requests", "pull_request_read", "search_pull_requests",
                    "create_pull_request", "merge_pull_request", "update_pull_request",
                    "update_pull_request_branch", "pull_request_review_write",
                    "add_comment_to_pending_review", "add_issue_comment",
                    "add_reply_to_pull_request_comment", "request_copilot_review"],
        system_prompt=f"""You are the PR specialist.
Repo: {state.get('repo_owner')}/{state.get('repo_name')}
Handle all pull request operations using the appropriate tool."""
    )


async def issues_node(state: dict, llm, all_tools: list) -> dict:
    return await run_specialist_node(
        state, llm, all_tools,
        node_name="issues_node",
        tool_names=["list_issues", "issue_read", "search_issues",
                    "list_issue_types", "issue_write", "sub_issue_write"],
        system_prompt=f"""You are the Issues specialist.
Repo: {state.get('repo_owner')}/{state.get('repo_name')}
Handle bugs, features, roadmap, and issue tracking."""
    )


async def team_node(state: dict, llm, all_tools: list) -> dict:
    return await run_specialist_node(
        state, llm, all_tools,
        node_name="team_node",
        tool_names=["get_me", "get_team_members", "get_teams", "search_users"],
        system_prompt=f"""You are the Team specialist.
Repo: {state.get('repo_owner')}/{state.get('repo_name')}
Handle people, org structure, and contributor queries."""
    )


async def security_node(state: dict, llm, all_tools: list) -> dict:
    return await run_specialist_node(
        state, llm, all_tools,
        node_name="security_node",
        tool_names=["run_secret_scanning", "search_code", "get_file_contents"],
        system_prompt=f"""You are the Security Audit specialist.
Repo: {state.get('repo_owner')}/{state.get('repo_name')}
Scan for secrets, hardcoded credentials, vulnerabilities.
Search patterns: "password =", "api_key", "token =", filename:.env"""
    )


async def copilot_node(state: dict, llm, all_tools: list) -> dict:
    return await run_specialist_node(
        state, llm, all_tools,
        node_name="copilot_node",
        tool_names=["create_pull_request_with_copilot", "assign_copilot_to_issue",
                    "get_copilot_job_status", "request_copilot_review"],
        system_prompt=f"""You are the Copilot Automation specialist.
Repo: {state.get('repo_owner')}/{state.get('repo_name')}
Handle AI automation tasks using GitHub Copilot."""
    )


async def write_ops_node(state: dict, llm, all_tools: list) -> dict:
    return await run_specialist_node(
        state, llm, all_tools,
        node_name="write_ops",
        tool_names=["create_branch", "create_repository", "fork_repository",
                    "create_or_update_file", "delete_file", "push_files"],
        system_prompt=f"""You are the Write Operations specialist.
Repo: {state.get('repo_owner')}/{state.get('repo_name')}
Handle all create, update, delete operations on files, branches, repos."""
    )


async def meta_node(state: dict, llm, all_tools: list) -> dict:
    return await run_specialist_node(
        state, llm, all_tools,
        node_name="meta_node",
        tool_names=["search_repositories", "search_users", "get_label", "list_issue_types"],
        system_prompt=f"""You are the Meta Discovery specialist.
Repo: {state.get('repo_owner')}/{state.get('repo_name')}
Handle repo discovery, labels, metadata queries."""
    )


# ─────────────────────────────────────────────
# NODE: SYNTHESIZER
# ─────────────────────────────────────────────
async def synthesizer_node(state: dict, llm) -> dict:
    logger.info(f"[SYNTHESIZER] Building final answer from {len(state.get('intermediate_results', {}))} results")

    system = SystemMessage(content="""
You are the Synthesizer. Create the perfect final answer.
Rules:
1. Combine ALL collected data into one coherent answer
2. Use markdown formatting (tables, lists, headers)
3. Cite sources (mention which file/PR/issue data came from)
4. Be comprehensive but clear
5. If data is incomplete, mention what could not be found
""")

    context = f"""
Original Query: {state['user_query']}
Repo: {state.get('repo_owner')}/{state.get('repo_name')}
Tools Used: {state.get('tool_calls_made', [])}

All Collected Data:
{state.get('intermediate_results', {})}
"""

    response = await llm.ainvoke([system, HumanMessage(content=context)])
    logger.info(f"[SYNTHESIZER] Final answer length: {len(response.content)}")

    return {
        **state,
        "final_answer": response.content,
        "is_complete": True
    }