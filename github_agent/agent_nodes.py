import logging
from functools import partial
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_openai import AzureChatOpenAI

logger = logging.getLogger(__name__)


def get_llm(azure_endpoint, azure_api_key, deployment_name, api_version, model_name):
    tracked_model_name = model_name or deployment_name
    logger.info(
        f"[LLM] Creating AzureChatOpenAI for deployment={deployment_name} model={tracked_model_name}"
    )
    return AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        azure_deployment=deployment_name,
        api_version=api_version,
        model=tracked_model_name,
        temperature=0,
        streaming=False
    )


def filter_tools(all_tools: list, tool_names: list):
    filtered = [t for t in all_tools if t.name in tool_names]
    logger.debug(f"[TOOLS] Filtered tools: {[t.name for t in filtered]}")
    return filtered


# ─────────────────────────────────────────────
# ASYNC TOOL EXECUTOR HELPER
# ─────────────────────────────────────────────
async def execute_tool_calls(response: AIMessage, all_tools: list) -> list[ToolMessage]:
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
            tool_messages.append(ToolMessage(content=f"Tool {tool_name} not found", tool_call_id=tool_id))
            continue

        try:
            result = await tool_map[tool_name].ainvoke(tool_args)
            logger.info(f"[TOOL EXEC] Tool {tool_name} result length: {len(str(result))}")
            tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))
        except Exception as e:
            logger.error(f"[TOOL EXEC] Tool {tool_name} failed: {e}")
            tool_messages.append(ToolMessage(content=f"Tool {tool_name} error: {str(e)}", tool_call_id=tool_id))

    return tool_messages


# ─────────────────────────────────────────────
# NODE: INTENT CLASSIFIER
# ─────────────────────────────────────────────
async def intent_classifier_node(state: dict, llm) -> dict:
    logger.info(f"[INTENT] Classifying query: {state['user_query'][:100]}")
    import json

    system = SystemMessage(content="""\
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

    human = HumanMessage(content=f"""\
Query: {state['user_query']}
Repo Owner: {state.get('repo_owner', 'NOT PROVIDED')}
Repo Name:  {state.get('repo_name',  'NOT PROVIDED')}
""")

    response = await llm.ainvoke([system, human])
    logger.info(f"[INTENT] Raw response: {response.content}")

    try:
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
    except Exception as e:
        logger.warning(f"[INTENT] JSON parse failed: {e} — using defaults")
        result = {"intent": "READ", "domain": ["CODE"], "complexity": "COMPLEX", "needs_clarification": False}

    logger.info(f"[INTENT] Classified → intent={result.get('intent')} domain={result.get('domain')}")

    return {
        **state,
        "intent":               result.get("intent", "READ"),
        "domain":               result.get("domain", ["CODE"]),
        "needs_clarification":  result.get("needs_clarification", False),
        "loop_count":           0,
        "is_complete":          False,
        "tool_calls_made":      [],
        "intermediate_results": {},
        "current_step":         0,
        "plan":                 [],
        "_prev_step":           -1,
        "_stuck_count":         0,
    }


# ─────────────────────────────────────────────
# NODE: ORCHESTRATOR (SUPER AGENT)
# ─────────────────────────────────────────────
async def orchestrator_node(state: dict, llm) -> dict:
    import json

    loop_count = state.get("loop_count", 0)
    logger.info(f"[ORCHESTRATOR] Loop #{loop_count} | step={state.get('current_step')} | results={list(state.get('intermediate_results', {}).keys())}")

    if loop_count >= 15:
        logger.warning("[ORCHESTRATOR] Max loops reached — forcing synthesizer")
        return {**state, "next_node": "synthesizer", "is_complete": True}

    intermediate  = state.get("intermediate_results", {})
    existing_plan = state.get("plan", [])

    # ── STEP 0: force root-only tree scan before anything else ──
    has_tree = any("tree_scan" in k for k in intermediate.keys())
    if not has_tree and loop_count == 0:
        logger.info("[ORCHESTRATOR] No tree yet — forcing root tree scan first")
        return {
            **state,
            "plan": [{
                "step": 1,
                "node": "code_analyst",
                "action": (
                    f"Get the repository file tree. "
                    f"First call get_file_contents with path='' (empty string) to list root level folders and files. "
                    f"Then based on what folders you ACTUALLY FIND in root, call get_file_contents on the 2-3 "
                    f"most relevant subfolders (use ONLY folder names that appear in the root result — "
                    f"do NOT guess paths like 'backend' or 'app' if they are not listed). "
                    f"owner={state.get('repo_owner')} repo={state.get('repo_name')}"
                ),
                "status": "pending",
                "purpose": "tree_scan"
            }],
            "current_step": 1,
            "next_node": "code_analyst",
            "loop_count": loop_count + 1,
            "_prev_step": -1,
            "_stuck_count": 0,
        }

    # ── Stuck-step detection: same current_step for 2+ consecutive loops → bail out ──
    current_step_now = state.get("current_step", 0)
    prev_step        = state.get("_prev_step", -1)
    stuck_count      = state.get("_stuck_count", 0)

    if current_step_now == prev_step and current_step_now > 0:
        stuck_count += 1
    else:
        stuck_count = 0

    if stuck_count >= 2:
        logger.warning(f"[ORCHESTRATOR] Step {current_step_now} stuck for {stuck_count + 1} loops — forcing synthesizer")
        return {
            **state,
            "next_node":    "synthesizer",
            "is_complete":  True,
            "_prev_step":   current_step_now,
            "_stuck_count": stuck_count,
        }

    # ── Extract full tree result (never truncate it) ──
    tree_result = ""
    for k, v in intermediate.items():
        if "tree_scan" in k:
            tree_result = str(v)
            break

    # ── Last tool result for context ──
    last_result = ""
    if intermediate:
        last_val = list(intermediate.values())[-1]
        last_result = str(last_val)[:4000]

    # ── Collect failed result keys so orchestrator knows what not to retry ──
    failed_keys = [
        k for k, v in intermediate.items()
        if any(marker in str(v).lower()[:300] for marker in
               ["error", "does not exist", "failed", "not found", "404"])
    ]

    system = SystemMessage(content="""\
You are the Orchestrator — the Super Agent that controls all other nodes.

CRITICAL RULES:
1. The REPO FILE TREE is provided — READ IT CAREFULLY before planning ANY file path
2. ONLY use file paths that EXIST in the tree — NEVER invent or guess paths
3. If a path returned a 404/error, it does NOT exist — do NOT retry it
4. ALWAYS carry forward the FULL plan — never drop previous steps
5. Mark completed steps as "done" — add NEW pending steps at the end
6. current_step must be the step number of the NEXT pending step
7. Route to synthesizer ONLY when you have enough real data to answer
8. NEVER re-read a file that already has a result key — check "Results Already Collected" first
9. NEVER retry a path listed under FAILED RESULT KEYS — those paths do not exist

FINDING CODE/ENDPOINTS — FOLLOW THIS ORDER:
  Step A: Deep tree scan (root + real subfolders found in root) — get REAL paths
  Step B: From the tree, identify EXACT files relevant to the query
  Step C: Call get_file_contents on each EXACT file path from the tree
  Step D: Synthesize from the actual file contents

NEVER search for paths not in the tree.
NEVER repeat a step that already failed.
If search_code returns >100k chars it is raw JSON — do NOT use it as final data,
instead use the file names it mentions to call get_file_contents on those files.

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

Respond ONLY in this exact JSON, no extra text, no markdown fences:
{
  "plan": [
    {"step": 1, "node": "code_analyst", "action": "...", "status": "done", "purpose": "tree_scan"},
    {"step": 2, "node": "code_analyst", "action": "Read src/Jellyfin.Api/Controllers/UserController.cs", "status": "pending", "purpose": "read_controller"}
  ],
  "current_step": 2,
  "next_node": "code_analyst",
  "reasoning": "Tree shows src/Jellyfin.Api/Controllers/UserController.cs — reading it directly"
}
""")

    context = f"""\
User Query: {state['user_query']}
Repo: {state.get('repo_owner')}/{state.get('repo_name')}
Intent: {state.get('intent')}
Loop Count: {loop_count}

REPO FILE TREE (FULL — use ONLY these exact paths):
{tree_result}

EXISTING PLAN (carry forward ALL steps, mark done, add new pending):
{existing_plan}

Results Already Collected (DO NOT re-read these — data already available):
{list(intermediate.keys())}

FAILED RESULT KEYS — DO NOT RETRY THESE PATHS (they returned errors or 404):
{failed_keys}

Last Tool Result (most recent — read carefully to decide next step):
{last_result}
"""

    response = await llm.ainvoke([system, HumanMessage(content=context)])
    logger.info(f"[ORCHESTRATOR] Response: {response.content[:800]}")

    try:
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
    except Exception as e:
        logger.warning(f"[ORCHESTRATOR] JSON parse failed: {e} — routing to synthesizer")
        result = {
            "plan":         existing_plan,
            "current_step": state.get("current_step", 0),
            "next_node":    "synthesizer",
            "reasoning":    f"JSON parse error: {e}"
        }

    new_plan  = result.get("plan", existing_plan)
    new_step  = result.get("current_step", 0)
    next_node = result.get("next_node", "synthesizer")

    # ── Validate: if next_node is not synthesizer, the step must exist in plan ──
    if next_node != "synthesizer":
        step_exists = any(item.get("step") == new_step for item in new_plan)
        if not step_exists:
            logger.warning(f"[ORCHESTRATOR] current_step={new_step} not in plan — routing to synthesizer")
            next_node = "synthesizer"

    logger.info(f"[ORCHESTRATOR] → next_node={next_node} | step={new_step} | reasoning={result.get('reasoning')}")

    return {
        **state,
        "plan":         new_plan,
        "current_step": new_step,
        "next_node":    next_node,
        "loop_count":   loop_count + 1,
        "_prev_step":   current_step_now,
        "_stuck_count": stuck_count,
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

    node_tools     = filter_tools(all_tools, tool_names)
    llm_with_tools = llm.bind_tools(node_tools)

    current_step = state.get("current_step", 0)
    loop_count   = state.get("loop_count", 0)
    plan         = state.get("plan", [])

    # ── Find plan item by step NUMBER not by list index ──
    current_plan_item = {}
    for item in plan:
        if item.get("step") == current_step:
            current_plan_item = item
            break

    action  = current_plan_item.get("action", "")
    purpose = current_plan_item.get("purpose", "")

    if not action:
        logger.warning(f"[{node_name.upper()}] No plan action for step {current_step} — falling back to user_query")
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

    # ── Append loop_count so retried steps don't overwrite previous results ──
    if purpose:
        result_key = f"{purpose}_{node_name}_step_{current_step}_loop_{loop_count}"
    else:
        result_key = f"{node_name}_step_{current_step}_loop_{loop_count}"

    results = state.get("intermediate_results", {})
    results[result_key] = all_content

    logger.info(f"[{node_name.upper()}] Done | stored key: {result_key} | content length: {len(all_content)}")

    return {
        **state,
        "messages":             state.get("messages", []) + [response] + tool_messages,
        "intermediate_results": results,
        "tool_calls_made":      state.get("tool_calls_made", []) + [node_name]
    }


# ─────────────────────────────────────────────
# ALL SPECIALIST NODES
# ─────────────────────────────────────────────
async def repo_analyst_node(state: dict, llm, all_tools: list) -> dict:
    return await run_specialist_node(
        state, llm, all_tools,
        node_name="repo_analyst",
        tool_names=["list_branches", "list_commits", "get_commit", "list_tags",
                    "list_releases", "get_latest_release", "get_release_by_tag", "get_tag"],
        system_prompt=f"""You are the Repo Analyst specialist.
Repo: {state.get('repo_owner')}/{state.get('repo_name')}
Tools: list_branches, list_commits, get_commit, list_tags, list_releases,
       get_latest_release, get_release_by_tag, get_tag
Use the most appropriate tool for the given action."""
    )


async def code_analyst_node(state: dict, llm, all_tools: list) -> dict:
    current_step = state.get("current_step", 0)
    plan         = state.get("plan", [])

    # ── Find plan item by step number ──
    current_plan_item = {}
    for item in plan:
        if item.get("step") == current_step:
            current_plan_item = item
            break

    purpose = current_plan_item.get("purpose", "")

    if purpose == "tree_scan":
        system_prompt = f"""\
You are the Code Analyst. Your ONLY job right now is to get the FULL repo file tree.

Repo: {state.get('repo_owner')}/{state.get('repo_name')}

STEP 1: Call get_file_contents with path="" (empty string) to list root-level folders and files.

STEP 2: Look at what folders ACTUALLY EXIST in the root result.
Then call get_file_contents on the 2-3 most relevant subfolders based on what you see.
Examples of what to look for:
  - .NET/C# repos: look for 'src', then the main project folder inside src
  - Python repos:  look for 'backend', 'app', 'src', or similar — whichever EXISTS in root
  - Node.js repos: look for 'src', 'lib', 'packages'

NEVER call get_file_contents on a path that was NOT listed in the root result.
owner={state.get('repo_owner')}  repo={state.get('repo_name')}

Return ALL file and folder names found across all calls.
This is critical — the orchestrator uses this to plan exact file paths."""

    else:
        system_prompt = f"""\
You are the Code Analyst specialist.
Repo: {state.get('repo_owner')}/{state.get('repo_name')}

Tools available:
- get_file_contents(owner, repo, path): Read a file or list a directory
- search_code(query, owner, repo): Search patterns across codebase

IMPORTANT:
- Use EXACT file paths — only paths confirmed to exist in the repo tree
- For directories, call get_file_contents to list contents first
- For API endpoints: read the actual route/controller files using get_file_contents
- Do NOT use search_code as a substitute for reading files
- Return the full raw file content — do NOT summarize"""

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

    system = SystemMessage(content="""\
You are the Synthesizer. Create the perfect final answer.

Rules:
1. Combine ALL collected data into one coherent answer
2. Use markdown formatting (tables, lists, headers)
3. Cite sources (mention which file/PR/issue the data came from)
4. Be comprehensive but clear
5. If data is incomplete, mention what could not be found
6. For API endpoints: present as a clean table with Method | Path | Description
""")

    # ── Pass full intermediate results but cap per-result to avoid token overflow ──
    all_data = state.get("intermediate_results", {})
    data_str = ""
    for k, v in all_data.items():
        chunk = str(v)[:8000]
        data_str += f"\n\n--- {k} ---\n{chunk}"

    context = f"""\
Original Query: {state['user_query']}
Repo: {state.get('repo_owner')}/{state.get('repo_name')}
Tools Used: {state.get('tool_calls_made', [])}

All Collected Data:
{data_str}
"""

    response = await llm.ainvoke([system, HumanMessage(content=context)])
    logger.info(f"[SYNTHESIZER] Final answer length: {len(response.content)}")

    return {
        **state,
        "final_answer": response.content,
        "is_complete":  True
    }