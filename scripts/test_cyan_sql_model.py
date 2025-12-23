#!/usr/bin/env python3
"""
Test harness for cyan-sql model with playbook correction simulation.

Tests:
1. Basic queries (SELECT)
2. Mutations (INSERT/UPDATE/DELETE)
3. Workflows (multi-step)
4. Playbook injection (hints improve output)
5. Error recovery simulation
"""

import subprocess
import json
import re
import sys

MODEL = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER = "adapters/cyan-sql-v4"

SYSTEM_PROMPT = """You are CyanLens, an AI assistant for Cyan workspace management.
You help users search, create, modify, and organize their workspace content.

## Schema
- groups(id, name, icon, color, created_at)
- workspaces(id, group_id, name, description, created_at)
- objects(id, workspace_id, type, name, board_mode, archived, created_at, updated_at)
  - type: 'whiteboard', 'chat', 'file'
  - board_mode: 'freeform' or 'notebook'
- notebook_cells(id, board_id, cell_type, content, cell_order)
  - cell_type: 'markdown', 'mermaid', 'code', 'image'
- board_labels(id, board_id, label)
- board_metadata(board_id, starred, template, rating)

## Key Relationships
- Objects belong to workspaces, workspaces belong to groups
- To find boards in a group: JOIN objects → workspaces → groups
- Labels and metadata are separate tables joined by board_id

## Output Format
- For QUERIES: Output SQL in ```sql``` block
- For MUTATIONS: Output JSON action plan with confirmation
- For WORKFLOWS: Output step-by-step action plan"""


def build_prompt(query: str, playbook_hints: list[str] = None) -> str:
    """Build prompt with optional playbook hints."""
    prompt = f"<|system|>\n{SYSTEM_PROMPT}\n<|end|>\n"
    prompt += "<|user|>\n"

    if playbook_hints:
        prompt += "Learned patterns:\n"
        for hint in playbook_hints:
            prompt += f"- {hint}\n"
        prompt += "\n"

    prompt += f"{query}\n<|end|>\n<|assistant|>\n"
    return prompt


def run_inference(prompt: str, max_tokens: int = 300) -> str:
    """Run model inference and return output."""
    cmd = [
        "python", "-m", "mlx_lm", "generate",
        "--model", MODEL,
        "--adapter-path", ADAPTER,
        "--max-tokens", str(max_tokens),
        "--prompt", prompt
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout

    # Extract generated text between ========== markers
    match = re.search(r'==========\n(.*?)\n==========', output, re.DOTALL)
    if match:
        return match.group(1).strip()
    return output


def extract_sql(response: str) -> str:
    """Extract SQL from response."""
    match = re.search(r'```sql\n(.*?)\n```', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_json(response: str) -> dict:
    """Extract JSON from response."""
    match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass

    # Try to find raw JSON
    try:
        # Find first { to last }
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(response[start:end])
    except:
        pass

    return None


def test_query(name: str, query: str, expected_contains: list[str], playbook_hints: list[str] = None):
    """Test a query and check output contains expected strings."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"Query: {query}")
    if playbook_hints:
        print(f"Playbook hints: {playbook_hints}")
    print("-" * 60)

    prompt = build_prompt(query, playbook_hints)
    response = run_inference(prompt)

    print(f"Response:\n{response}")
    print("-" * 60)

    # Check for SQL
    sql = extract_sql(response)
    if sql:
        print(f"✓ Extracted SQL: {sql[:100]}...")

        passed = True
        for expected in expected_contains:
            if expected.lower() in sql.lower():
                print(f"  ✓ Contains: {expected}")
            else:
                print(f"  ✗ Missing: {expected}")
                passed = False
        return passed

    # Check for JSON
    json_data = extract_json(response)
    if json_data:
        print(f"✓ Extracted JSON action plan")
        if isinstance(json_data, dict):
            print(f"  Intent: {json_data.get('intent', 'unknown')}")
        else:
            print(f"  (List of {len(json_data)} actions)")

        passed = True
        json_str = json.dumps(json_data).lower()
        for expected in expected_contains:
            if expected.lower() in json_str:
                print(f"  ✓ Contains: {expected}")
            else:
                print(f"  ✗ Missing: {expected}")
                passed = False
        return passed

    print("✗ Could not extract SQL or JSON from response")
    return False


def main():
    results = []

    # === BASIC QUERIES ===
    print("\n" + "="*60)
    print("SECTION 1: BASIC QUERIES")
    print("="*60)

    results.append(("Basic: find boards", test_query(
        "Find Design boards",
        "find Design boards",
        ["SELECT", "objects", "groups", "Design"]
    )))

    results.append(("Basic: list groups", test_query(
        "List all groups",
        "list all groups",
        ["SELECT", "groups"]
    )))

    results.append(("Basic: notebooks", test_query(
        "Show notebooks in Engineering",
        "show notebooks in Engineering",
        ["SELECT", "notebook", "Engineering"]
    )))

    # === LABEL QUERIES ===
    print("\n" + "="*60)
    print("SECTION 2: LABEL QUERIES")
    print("="*60)

    results.append(("Labels: find labeled", test_query(
        "Find reviewed boards",
        "boards labeled reviewed",
        ["SELECT", "board_labels", "reviewed"]
    )))

    results.append(("Labels: starred", test_query(
        "Find starred boards",
        "show starred boards",
        ["SELECT", "board_metadata", "starred"]
    )))

    # === CELL TYPE QUERIES ===
    print("\n" + "="*60)
    print("SECTION 3: CELL TYPE QUERIES")
    print("="*60)

    results.append(("Cells: mermaid", test_query(
        "Find boards with mermaid",
        "boards with mermaid cells",
        ["SELECT", "notebook_cells", "mermaid"]
    )))

    results.append(("Cells: code", test_query(
        "Find boards with code",
        "boards with code cells",
        ["SELECT", "notebook_cells", "code"]
    )))

    # === MUTATIONS ===
    print("\n" + "="*60)
    print("SECTION 4: MUTATIONS")
    print("="*60)

    results.append(("Mutation: create workspace", test_query(
        "Create workspace",
        "create a workspace called API Docs in Engineering",
        ["INSERT", "workspaces", "API Docs"]
    )))

    results.append(("Mutation: add label", test_query(
        "Add label",
        "add label 'reviewed' to this board",
        ["INSERT", "board_labels", "reviewed"]
    )))

    results.append(("Mutation: star board", test_query(
        "Star board",
        "star this board",
        ["board_metadata", "starred"]
    )))

    results.append(("Mutation: rename", test_query(
        "Rename board",
        "rename this board to Final Version",
        ["UPDATE", "objects", "Final Version"]
    )))

    # === WORKFLOWS ===
    print("\n" + "="*60)
    print("SECTION 5: WORKFLOWS")
    print("="*60)

    results.append(("Workflow: copy board", test_query(
        "Copy board",
        "copy this board to Marketing workspace",
        ["SELECT", "INSERT", "Marketing"]
    )))

    results.append(("Workflow: archive old", test_query(
        "Archive old boards",
        "archive all boards older than 90 days",
        ["UPDATE", "archived", "90"]
    )))

    # === PLAYBOOK INJECTION TEST ===
    print("\n" + "="*60)
    print("SECTION 6: PLAYBOOK INJECTION")
    print("="*60)

    # Without playbook - might generate suboptimal query
    results.append(("Playbook: without hint", test_query(
        "Content search (no hint)",
        "find boards about authentication",
        ["SELECT", "content", "auth"]
    )))

    # With playbook hint - should generate better query
    results.append(("Playbook: with hint", test_query(
        "Content search (with hint)",
        "find boards about authentication",
        ["SELECT", "notebook_cells", "content", "LIKE", "auth"],
        playbook_hints=[
            "To search content, query notebook_cells.content with LIKE",
            "Use LOWER() for case-insensitive search"
        ]
    )))

    # === SUMMARY ===
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    print(f"\nResults: {passed}/{total} tests passed\n")
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())