import os
import datetime
from pathlib import Path

def append_to_agent_log(user_prompt: str, response_summary: str, actions: list[str], repo_root: str):
    """
    Appends a formatted turn entry to the log.txt file located in the user's home directory
    under the 'hackerrank_orchestrate' folder, exactly as required by AGENTS.md.
    """
    # 1. Determine log file path based on OS
    home_dir = Path.home()
    log_dir = home_dir / "hackerrank_orchestrate"
    log_path = log_dir / "log.txt"

    # Ensure the directory exists
    log_dir.mkdir(parents=True, exist_ok=True)

    # 2. Get current ISO-8601 timestamp with timezone
    # For simplicity, using UTC or local time. Local time is preferred.
    now = datetime.datetime.now().astimezone()
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%S%z")
    # Insert colon in timezone offset if missing (e.g. +0530 -> +05:30)
    if len(timestamp) > 5 and timestamp[-5] in "+-":
        timestamp = timestamp[:-2] + ":" + timestamp[-2:]

    # 3. Format the log entry
    short_title = (user_prompt[:75] + "...") if len(user_prompt) > 75 else user_prompt
    
    actions_str = "\n".join(f"* {action}" for action in actions)

    log_entry = f"""
## [{timestamp}] {short_title}

User Prompt (verbatim, secrets redacted):
{user_prompt}

Agent Response Summary:
{response_summary}

Actions:
{actions_str}

Context:
tool=PythonScript
branch=main
repo_root={repo_root}
worktree=main
parent_agent=None
"""

    # 4. Append to file
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(log_entry)
        
    print(f"Successfully appended log entry to {log_path}")


if __name__ == "__main__":
    # Example Usage:
    repo_path = str(Path(__file__).parent.parent.resolve())
    
    append_to_agent_log(
        user_prompt="koi code add kro jo log.txt bana ke de",
        response_summary="Generated a Python script to programmatically create and append to the AGENTS.md log.txt file in the user's home directory.",
        actions=["Created code/generate_agent_log.py"],
        repo_root=repo_path
    )
