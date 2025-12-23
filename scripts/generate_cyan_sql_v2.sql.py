#!/usr/bin/env python3
"""
Generate comprehensive training data for cyan-sql model.

Categories:
1. QUERIES (SELECT) - ~150 examples
2. MUTATIONS (INSERT/UPDATE/DELETE) - ~100 examples
3. WORKFLOWS (multi-step) - ~50 examples

Total: ~300 examples
"""

import json
import random
from pathlib import Path

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
- whiteboard_elements(id, board_id, element_type, x, y, width, height)
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

# Groups and workspaces for variety
GROUPS = ["Design", "Engineering", "Marketing", "Sales", "Product", "Management"]
WORKSPACES = ["Brand Kit", "API Docs", "Q1 Planning", "Customer Research", "Sprint Board", "Templates", "Archive"]
LABELS = ["reviewed", "draft", "needs-work", "approved", "archived", "priority", "template", "wip"]
CELL_TYPES = ["markdown", "mermaid", "code", "image"]
BOARD_MODES = ["notebook", "freeform"]

def make_example(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant}
        ]
    }

def generate_query_examples() -> list:
    """Generate SELECT query examples."""
    examples = []

    # === Simple retrieval ===
    examples.append(make_example(
        "list all groups",
        "```sql\nSELECT id, name, icon, color FROM groups ORDER BY name;\n```"
    ))

    examples.append(make_example(
        "show me all workspaces",
        "```sql\nSELECT w.id, w.name, g.name as group_name FROM workspaces w JOIN groups g ON w.group_id = g.id ORDER BY g.name, w.name;\n```"
    ))

    for group in GROUPS[:4]:
        examples.append(make_example(
            f"find {group} boards",
            f"```sql\nSELECT o.id, o.name, o.board_mode FROM objects o JOIN workspaces w ON o.workspace_id = w.id JOIN groups g ON w.group_id = g.id WHERE g.name = '{group}' AND o.type = 'whiteboard' AND o.archived = 0;\n```"
        ))

        examples.append(make_example(
            f"show me notebooks in {group}",
            f"```sql\nSELECT o.id, o.name FROM objects o JOIN workspaces w ON o.workspace_id = w.id JOIN groups g ON w.group_id = g.id WHERE g.name = '{group}' AND o.board_mode = 'notebook' AND o.archived = 0;\n```"
        ))

    # === Workspace queries ===
    examples.append(make_example(
        "what workspaces are in Engineering",
        "```sql\nSELECT w.id, w.name, w.description FROM workspaces w JOIN groups g ON w.group_id = g.id WHERE g.name = 'Engineering';\n```"
    ))

    examples.append(make_example(
        "list workspaces with their board counts",
        "```sql\nSELECT w.name, g.name as group_name, COUNT(o.id) as board_count FROM workspaces w JOIN groups g ON w.group_id = g.id LEFT JOIN objects o ON o.workspace_id = w.id AND o.type = 'whiteboard' GROUP BY w.id ORDER BY board_count DESC;\n```"
    ))

    # === Label queries ===
    for label in LABELS[:5]:
        examples.append(make_example(
            f"find boards labeled '{label}'",
            f"```sql\nSELECT o.id, o.name, w.name as workspace FROM objects o JOIN workspaces w ON o.workspace_id = w.id JOIN board_labels bl ON bl.board_id = o.id WHERE bl.label = '{label}' AND o.archived = 0;\n```"
        ))

    examples.append(make_example(
        "boards with label draft in Design",
        "```sql\nSELECT o.id, o.name FROM objects o JOIN workspaces w ON o.workspace_id = w.id JOIN groups g ON w.group_id = g.id JOIN board_labels bl ON bl.board_id = o.id WHERE g.name = 'Design' AND bl.label = 'draft' AND o.archived = 0;\n```"
    ))

    examples.append(make_example(
        "what labels does this board have",
        "```sql\nSELECT bl.label FROM board_labels bl WHERE bl.board_id = :board_id;\n```\n\nReplace `:board_id` with the current board's ID."
    ))

    examples.append(make_example(
        "boards with multiple labels",
        "```sql\nSELECT o.id, o.name, COUNT(bl.id) as label_count FROM objects o JOIN board_labels bl ON bl.board_id = o.id GROUP BY o.id HAVING label_count > 1;\n```"
    ))

    # === Metadata queries (starred, template, rating) ===
    examples.append(make_example(
        "show starred boards",
        "```sql\nSELECT o.id, o.name, w.name as workspace FROM objects o JOIN workspaces w ON o.workspace_id = w.id JOIN board_metadata bm ON bm.board_id = o.id WHERE bm.starred = 1 AND o.archived = 0;\n```"
    ))

    examples.append(make_example(
        "find template boards",
        "```sql\nSELECT o.id, o.name, w.name as workspace FROM objects o JOIN workspaces w ON o.workspace_id = w.id JOIN board_metadata bm ON bm.board_id = o.id WHERE bm.template = 1;\n```"
    ))

    examples.append(make_example(
        "highly rated boards",
        "```sql\nSELECT o.id, o.name, bm.rating FROM objects o JOIN board_metadata bm ON bm.board_id = o.id WHERE bm.rating >= 4 ORDER BY bm.rating DESC;\n```"
    ))

    examples.append(make_example(
        "starred notebooks in Engineering",
        "```sql\nSELECT o.id, o.name FROM objects o JOIN workspaces w ON o.workspace_id = w.id JOIN groups g ON w.group_id = g.id JOIN board_metadata bm ON bm.board_id = o.id WHERE g.name = 'Engineering' AND bm.starred = 1 AND o.board_mode = 'notebook';\n```"
    ))

    # === Cell type queries ===
    for cell_type in CELL_TYPES:
        examples.append(make_example(
            f"boards with {cell_type} cells",
            f"```sql\nSELECT DISTINCT o.id, o.name FROM objects o JOIN notebook_cells nc ON nc.board_id = o.id WHERE nc.cell_type = '{cell_type}' AND o.archived = 0;\n```"
        ))

    examples.append(make_example(
        "find boards with mermaid diagrams",
        "```sql\nSELECT DISTINCT o.id, o.name, w.name as workspace FROM objects o JOIN workspaces w ON o.workspace_id = w.id JOIN notebook_cells nc ON nc.board_id = o.id WHERE nc.cell_type = 'mermaid' AND o.archived = 0;\n```"
    ))

    examples.append(make_example(
        "boards with both code and mermaid",
        "```sql\nSELECT o.id, o.name FROM objects o WHERE o.id IN (SELECT board_id FROM notebook_cells WHERE cell_type = 'code') AND o.id IN (SELECT board_id FROM notebook_cells WHERE cell_type = 'mermaid');\n```"
    ))

    examples.append(make_example(
        "count cells by type in Design",
        "```sql\nSELECT nc.cell_type, COUNT(*) as count FROM notebook_cells nc JOIN objects o ON nc.board_id = o.id JOIN workspaces w ON o.workspace_id = w.id JOIN groups g ON w.group_id = g.id WHERE g.name = 'Design' GROUP BY nc.cell_type;\n```"
    ))

    # === Content search ===
    search_terms = ["authentication", "login", "payment", "user", "api", "database", "error handling"]
    for term in search_terms[:4]:
        examples.append(make_example(
            f"find boards mentioning {term}",
            f"```sql\nSELECT DISTINCT o.id, o.name FROM objects o JOIN notebook_cells nc ON nc.board_id = o.id WHERE LOWER(nc.content) LIKE '%{term}%' AND o.archived = 0;\n```"
        ))

    examples.append(make_example(
        "search for 'OAuth' in code cells",
        "```sql\nSELECT DISTINCT o.id, o.name, nc.content FROM objects o JOIN notebook_cells nc ON nc.board_id = o.id WHERE nc.cell_type = 'code' AND LOWER(nc.content) LIKE '%oauth%';\n```"
    ))

    examples.append(make_example(
        "mermaid diagrams showing login flow",
        "```sql\nSELECT o.id, o.name, nc.content FROM objects o JOIN notebook_cells nc ON nc.board_id = o.id WHERE nc.cell_type = 'mermaid' AND LOWER(nc.content) LIKE '%login%';\n```"
    ))

    # === Temporal queries ===
    examples.append(make_example(
        "recent boards",
        "```sql\nSELECT o.id, o.name, datetime(o.updated_at, 'unixepoch') as updated FROM objects o WHERE o.type = 'whiteboard' AND o.archived = 0 ORDER BY o.updated_at DESC LIMIT 10;\n```"
    ))

    examples.append(make_example(
        "boards modified today",
        "```sql\nSELECT o.id, o.name FROM objects o WHERE o.type = 'whiteboard' AND date(o.updated_at, 'unixepoch') = date('now') AND o.archived = 0;\n```"
    ))

    examples.append(make_example(
        "boards not accessed in 90 days",
        "```sql\nSELECT o.id, o.name, datetime(o.updated_at, 'unixepoch') as last_updated FROM objects o WHERE o.type = 'whiteboard' AND o.updated_at < strftime('%s', 'now', '-90 days') AND o.archived = 0;\n```"
    ))

    examples.append(make_example(
        "oldest notebooks",
        "```sql\nSELECT o.id, o.name, datetime(o.created_at, 'unixepoch') as created FROM objects o WHERE o.board_mode = 'notebook' ORDER BY o.created_at ASC LIMIT 10;\n```"
    ))

    examples.append(make_example(
        "boards created this week",
        "```sql\nSELECT o.id, o.name, datetime(o.created_at, 'unixepoch') as created FROM objects o WHERE o.type = 'whiteboard' AND o.created_at >= strftime('%s', 'now', '-7 days') ORDER BY o.created_at DESC;\n```"
    ))

    # === Aggregation queries ===
    examples.append(make_example(
        "how many boards per group",
        "```sql\nSELECT g.name, COUNT(o.id) as board_count FROM groups g LEFT JOIN workspaces w ON w.group_id = g.id LEFT JOIN objects o ON o.workspace_id = w.id AND o.type = 'whiteboard' GROUP BY g.id ORDER BY board_count DESC;\n```"
    ))

    examples.append(make_example(
        "groups with no boards",
        "```sql\nSELECT g.name FROM groups g WHERE g.id NOT IN (SELECT DISTINCT w.group_id FROM workspaces w JOIN objects o ON o.workspace_id = w.id WHERE o.type = 'whiteboard');\n```"
    ))

    examples.append(make_example(
        "workspace with most notebooks",
        "```sql\nSELECT w.name, g.name as group_name, COUNT(o.id) as notebook_count FROM workspaces w JOIN groups g ON w.group_id = g.id JOIN objects o ON o.workspace_id = w.id WHERE o.board_mode = 'notebook' GROUP BY w.id ORDER BY notebook_count DESC LIMIT 1;\n```"
    ))

    examples.append(make_example(
        "average cells per notebook",
        "```sql\nSELECT AVG(cell_count) as avg_cells FROM (SELECT o.id, COUNT(nc.id) as cell_count FROM objects o LEFT JOIN notebook_cells nc ON nc.board_id = o.id WHERE o.board_mode = 'notebook' GROUP BY o.id);\n```"
    ))

    # === Complex joins ===
    examples.append(make_example(
        "all content in Design group",
        "```sql\nSELECT o.id, o.name, o.type, o.board_mode, w.name as workspace FROM objects o JOIN workspaces w ON o.workspace_id = w.id JOIN groups g ON w.group_id = g.id WHERE g.name = 'Design' AND o.archived = 0 ORDER BY o.type, o.name;\n```"
    ))

    examples.append(make_example(
        "boards with templates in their workspace",
        "```sql\nSELECT DISTINCT o.id, o.name FROM objects o JOIN workspaces w ON o.workspace_id = w.id WHERE w.id IN (SELECT o2.workspace_id FROM objects o2 JOIN board_metadata bm ON bm.board_id = o2.id WHERE bm.template = 1);\n```"
    ))

    # === Informal variations ===
    examples.append(make_example(
        "gimme all the design stuff",
        "```sql\nSELECT o.id, o.name, o.type FROM objects o JOIN workspaces w ON o.workspace_id = w.id JOIN groups g ON w.group_id = g.id WHERE g.name = 'Design' AND o.archived = 0;\n```"
    ))

    examples.append(make_example(
        "whats in engineering",
        "```sql\nSELECT o.id, o.name, w.name as workspace FROM objects o JOIN workspaces w ON o.workspace_id = w.id JOIN groups g ON w.group_id = g.id WHERE g.name = 'Engineering' AND o.archived = 0;\n```"
    ))

    examples.append(make_example(
        "any boards with diagrams?",
        "```sql\nSELECT DISTINCT o.id, o.name FROM objects o JOIN notebook_cells nc ON nc.board_id = o.id WHERE nc.cell_type = 'mermaid' AND o.archived = 0;\n```"
    ))

    examples.append(make_example(
        "can i get boards with code",
        "```sql\nSELECT DISTINCT o.id, o.name FROM objects o JOIN notebook_cells nc ON nc.board_id = o.id WHERE nc.cell_type = 'code' AND o.archived = 0;\n```"
    ))

    # === More query variations ===
    for group in GROUPS:
        examples.append(make_example(
            f"what's in {group} group",
            f"```sql\nSELECT o.id, o.name, w.name as workspace FROM objects o JOIN workspaces w ON o.workspace_id = w.id JOIN groups g ON w.group_id = g.id WHERE g.name = '{group}' AND o.archived = 0;\n```"
        ))

        examples.append(make_example(
            f"count boards in {group}",
            f"```sql\nSELECT COUNT(*) as board_count FROM objects o JOIN workspaces w ON o.workspace_id = w.id JOIN groups g ON w.group_id = g.id WHERE g.name = '{group}' AND o.type = 'whiteboard' AND o.archived = 0;\n```"
        ))

    # More label variations
    examples.append(make_example(
        "boards without any labels",
        "```sql\nSELECT o.id, o.name FROM objects o WHERE o.type = 'whiteboard' AND o.id NOT IN (SELECT board_id FROM board_labels);\n```"
    ))

    examples.append(make_example(
        "all labels used",
        "```sql\nSELECT DISTINCT label FROM board_labels ORDER BY label;\n```"
    ))

    examples.append(make_example(
        "most used labels",
        "```sql\nSELECT label, COUNT(*) as count FROM board_labels GROUP BY label ORDER BY count DESC;\n```"
    ))

    # File and chat queries
    examples.append(make_example(
        "find all files",
        "```sql\nSELECT o.id, o.name, o.mime_type, w.name as workspace FROM objects o JOIN workspaces w ON o.workspace_id = w.id WHERE o.type = 'file' AND o.archived = 0;\n```"
    ))

    examples.append(make_example(
        "show chat threads",
        "```sql\nSELECT o.id, o.name, w.name as workspace FROM objects o JOIN workspaces w ON o.workspace_id = w.id WHERE o.type = 'chat' AND o.archived = 0;\n```"
    ))

    examples.append(make_example(
        "files in Design",
        "```sql\nSELECT o.id, o.name FROM objects o JOIN workspaces w ON o.workspace_id = w.id JOIN groups g ON w.group_id = g.id WHERE g.name = 'Design' AND o.type = 'file';\n```"
    ))

    # Specific workspace queries
    for ws in WORKSPACES[:4]:
        examples.append(make_example(
            f"boards in {ws}",
            f"```sql\nSELECT o.id, o.name, o.board_mode FROM objects o JOIN workspaces w ON o.workspace_id = w.id WHERE w.name = '{ws}' AND o.type = 'whiteboard' AND o.archived = 0;\n```"
        ))

    # Content search variations
    examples.append(make_example(
        "find anything about users",
        "```sql\nSELECT DISTINCT o.id, o.name FROM objects o JOIN notebook_cells nc ON nc.board_id = o.id WHERE LOWER(nc.content) LIKE '%user%' AND o.archived = 0;\n```"
    ))

    examples.append(make_example(
        "code cells with import statements",
        "```sql\nSELECT o.name, nc.content FROM objects o JOIN notebook_cells nc ON nc.board_id = o.id WHERE nc.cell_type = 'code' AND nc.content LIKE '%import%';\n```"
    ))

    examples.append(make_example(
        "mermaid diagrams with sequenceDiagram",
        "```sql\nSELECT o.name, nc.content FROM objects o JOIN notebook_cells nc ON nc.board_id = o.id WHERE nc.cell_type = 'mermaid' AND nc.content LIKE '%sequenceDiagram%';\n```"
    ))

    examples.append(make_example(
        "flowcharts in notebooks",
        "```sql\nSELECT o.name, nc.content FROM objects o JOIN notebook_cells nc ON nc.board_id = o.id WHERE nc.cell_type = 'mermaid' AND nc.content LIKE '%flowchart%';\n```"
    ))

    # Archived content
    examples.append(make_example(
        "show archived boards",
        "```sql\nSELECT o.id, o.name, w.name as workspace FROM objects o JOIN workspaces w ON o.workspace_id = w.id WHERE o.archived = 1;\n```"
    ))

    examples.append(make_example(
        "archived notebooks in Engineering",
        "```sql\nSELECT o.id, o.name FROM objects o JOIN workspaces w ON o.workspace_id = w.id JOIN groups g ON w.group_id = g.id WHERE g.name = 'Engineering' AND o.board_mode = 'notebook' AND o.archived = 1;\n```"
    ))

    # Complex filters
    examples.append(make_example(
        "starred notebooks in Design with mermaid",
        "```sql\nSELECT DISTINCT o.id, o.name FROM objects o JOIN workspaces w ON o.workspace_id = w.id JOIN groups g ON w.group_id = g.id JOIN board_metadata bm ON bm.board_id = o.id JOIN notebook_cells nc ON nc.board_id = o.id WHERE g.name = 'Design' AND o.board_mode = 'notebook' AND bm.starred = 1 AND nc.cell_type = 'mermaid';\n```"
    ))

    examples.append(make_example(
        "highly rated boards with code cells",
        "```sql\nSELECT DISTINCT o.id, o.name, bm.rating FROM objects o JOIN board_metadata bm ON bm.board_id = o.id JOIN notebook_cells nc ON nc.board_id = o.id WHERE bm.rating >= 4 AND nc.cell_type = 'code';\n```"
    ))

    examples.append(make_example(
        "templates with mermaid diagrams",
        "```sql\nSELECT DISTINCT o.id, o.name FROM objects o JOIN board_metadata bm ON bm.board_id = o.id JOIN notebook_cells nc ON nc.board_id = o.id WHERE bm.template = 1 AND nc.cell_type = 'mermaid';\n```"
    ))

    # Board details
    examples.append(make_example(
        "show cells in this board",
        "```sql\nSELECT nc.id, nc.cell_type, nc.content, nc.cell_order FROM notebook_cells nc WHERE nc.board_id = :current_board_id ORDER BY nc.cell_order;\n```"
    ))

    examples.append(make_example(
        "how many cells does this board have",
        "```sql\nSELECT COUNT(*) as cell_count FROM notebook_cells WHERE board_id = :current_board_id;\n```"
    ))

    examples.append(make_example(
        "cell types in this notebook",
        "```sql\nSELECT cell_type, COUNT(*) as count FROM notebook_cells WHERE board_id = :current_board_id GROUP BY cell_type;\n```"
    ))

    # Whiteboard elements
    examples.append(make_example(
        "elements on this whiteboard",
        "```sql\nSELECT id, element_type, x, y, width, height FROM whiteboard_elements WHERE board_id = :current_board_id ORDER BY z_index;\n```"
    ))

    examples.append(make_example(
        "how many shapes on this board",
        "```sql\nSELECT COUNT(*) as element_count FROM whiteboard_elements WHERE board_id = :current_board_id;\n```"
    ))

    # Cross-group queries
    examples.append(make_example(
        "boards shared between Design and Marketing",
        "```sql\nSELECT o.id, o.name FROM objects o JOIN workspaces w ON o.workspace_id = w.id JOIN groups g ON w.group_id = g.id WHERE g.name IN ('Design', 'Marketing') AND o.type = 'whiteboard' AND o.archived = 0;\n```"
    ))

    # Empty/null checks
    examples.append(make_example(
        "boards with no description",
        "```sql\nSELECT w.name FROM workspaces w WHERE w.description IS NULL OR w.description = '';\n```"
    ))

    examples.append(make_example(
        "notebooks with no cells",
        "```sql\nSELECT o.id, o.name FROM objects o WHERE o.board_mode = 'notebook' AND o.id NOT IN (SELECT DISTINCT board_id FROM notebook_cells);\n```"
    ))

    # Size/count queries
    examples.append(make_example(
        "largest notebooks by cell count",
        "```sql\nSELECT o.id, o.name, COUNT(nc.id) as cell_count FROM objects o JOIN notebook_cells nc ON nc.board_id = o.id WHERE o.board_mode = 'notebook' GROUP BY o.id ORDER BY cell_count DESC LIMIT 10;\n```"
    ))

    examples.append(make_example(
        "groups by workspace count",
        "```sql\nSELECT g.name, COUNT(w.id) as workspace_count FROM groups g LEFT JOIN workspaces w ON w.group_id = g.id GROUP BY g.id ORDER BY workspace_count DESC;\n```"
    ))

    return examples


def generate_mutation_examples() -> list:
    """Generate INSERT/UPDATE/DELETE mutation examples."""
    examples = []

    # === CREATE WORKSPACE ===
    examples.append(make_example(
        "create a workspace called API Docs in Engineering",
        """I'll create a new workspace in the Engineering group.

```json
{
  "intent": "create_workspace",
  "requires_confirmation": true,
  "actions": [
    {
      "op": "SELECT",
      "purpose": "Find Engineering group ID",
      "sql": "SELECT id FROM groups WHERE name = 'Engineering'"
    },
    {
      "op": "INSERT",
      "table": "workspaces",
      "values": {
        "id": "{{generate_uuid}}",
        "group_id": "{{result[0].id}}",
        "name": "API Docs",
        "created_at": "{{now}}"
      }
    }
  ],
  "confirmation": "Create workspace 'API Docs' in Engineering group?"
}
```"""
    ))

    examples.append(make_example(
        "add a new workspace for Q1 Planning under Product",
        """I'll create a Q1 Planning workspace in the Product group.

```json
{
  "intent": "create_workspace",
  "requires_confirmation": true,
  "actions": [
    {
      "op": "SELECT",
      "purpose": "Find Product group ID",
      "sql": "SELECT id FROM groups WHERE name = 'Product'"
    },
    {
      "op": "INSERT",
      "table": "workspaces",
      "values": {
        "id": "{{generate_uuid}}",
        "group_id": "{{result[0].id}}",
        "name": "Q1 Planning",
        "created_at": "{{now}}"
      }
    }
  ],
  "confirmation": "Create workspace 'Q1 Planning' in Product group?"
}
```"""
    ))

    # === CREATE BOARD ===
    examples.append(make_example(
        "create a new notebook called Auth Flow in Engineering",
        """I'll create a new notebook board.

```json
{
  "intent": "create_board",
  "requires_confirmation": true,
  "actions": [
    {
      "op": "SELECT",
      "purpose": "Find first Engineering workspace",
      "sql": "SELECT w.id FROM workspaces w JOIN groups g ON w.group_id = g.id WHERE g.name = 'Engineering' LIMIT 1"
    },
    {
      "op": "INSERT",
      "table": "objects",
      "values": {
        "id": "{{generate_uuid}}",
        "workspace_id": "{{result[0].id}}",
        "type": "whiteboard",
        "name": "Auth Flow",
        "board_mode": "notebook",
        "archived": 0,
        "created_at": "{{now}}",
        "updated_at": "{{now}}"
      }
    }
  ],
  "confirmation": "Create notebook 'Auth Flow' in Engineering?"
}
```"""
    ))

    examples.append(make_example(
        "add a freeform board to Design > Brand Kit",
        """I'll create a freeform board in the Brand Kit workspace.

```json
{
  "intent": "create_board",
  "requires_confirmation": true,
  "actions": [
    {
      "op": "SELECT",
      "purpose": "Find Brand Kit workspace in Design",
      "sql": "SELECT w.id FROM workspaces w JOIN groups g ON w.group_id = g.id WHERE g.name = 'Design' AND w.name = 'Brand Kit'"
    },
    {
      "op": "INSERT",
      "table": "objects",
      "values": {
        "id": "{{generate_uuid}}",
        "workspace_id": "{{result[0].id}}",
        "type": "whiteboard",
        "name": "New Board",
        "board_mode": "freeform",
        "archived": 0,
        "created_at": "{{now}}"
      }
    }
  ],
  "confirmation": "Create freeform board in Design > Brand Kit?"
}
```"""
    ))

    # === RENAME ===
    examples.append(make_example(
        "rename this board to Final Version",
        """I'll rename the current board.

```json
{
  "intent": "rename",
  "requires_confirmation": true,
  "actions": [
    {
      "op": "UPDATE",
      "table": "objects",
      "set": {"name": "Final Version", "updated_at": "{{now}}"},
      "where": {"id": ":current_board_id"}
    }
  ],
  "confirmation": "Rename this board to 'Final Version'?"
}
```"""
    ))

    examples.append(make_example(
        "rename workspace Brand Kit to Brand Assets",
        """I'll rename the workspace.

```json
{
  "intent": "rename",
  "requires_confirmation": true,
  "actions": [
    {
      "op": "UPDATE",
      "table": "workspaces",
      "set": {"name": "Brand Assets"},
      "where": {"name": "Brand Kit"}
    }
  ],
  "confirmation": "Rename workspace 'Brand Kit' to 'Brand Assets'?"
}
```"""
    ))

    # === MOVE ===
    examples.append(make_example(
        "move this board to Marketing workspace",
        """I'll move this board to a Marketing workspace.

```json
{
  "intent": "move",
  "requires_confirmation": true,
  "actions": [
    {
      "op": "SELECT",
      "purpose": "Find Marketing workspace",
      "sql": "SELECT w.id FROM workspaces w JOIN groups g ON w.group_id = g.id WHERE g.name = 'Marketing' LIMIT 1"
    },
    {
      "op": "UPDATE",
      "table": "objects",
      "set": {"workspace_id": "{{result[0].id}}", "updated_at": "{{now}}"},
      "where": {"id": ":current_board_id"}
    }
  ],
  "confirmation": "Move this board to Marketing?"
}
```"""
    ))

    examples.append(make_example(
        "move Auth Flow board to the API Docs workspace",
        """I'll move the Auth Flow board.

```json
{
  "intent": "move",
  "requires_confirmation": true,
  "actions": [
    {
      "op": "SELECT",
      "purpose": "Find source board",
      "sql": "SELECT id FROM objects WHERE name = 'Auth Flow' AND type = 'whiteboard'"
    },
    {
      "op": "SELECT",
      "purpose": "Find target workspace",
      "sql": "SELECT id FROM workspaces WHERE name = 'API Docs'"
    },
    {
      "op": "UPDATE",
      "table": "objects",
      "set": {"workspace_id": "{{result[1].id}}", "updated_at": "{{now}}"},
      "where": {"id": "{{result[0].id}}"}
    }
  ],
  "confirmation": "Move 'Auth Flow' to 'API Docs' workspace?"
}
```"""
    ))

    # === LABELS ===
    examples.append(make_example(
        "add label 'reviewed' to this board",
        """I'll add the 'reviewed' label.

```json
{
  "intent": "add_label",
  "requires_confirmation": false,
  "actions": [
    {
      "op": "INSERT",
      "table": "board_labels",
      "values": {
        "id": "{{generate_uuid}}",
        "board_id": ":current_board_id",
        "label": "reviewed",
        "created_at": "{{now}}"
      },
      "on_conflict": "ignore"
    }
  ]
}
```"""
    ))

    examples.append(make_example(
        "remove the draft label from this board",
        """I'll remove the 'draft' label.

```json
{
  "intent": "remove_label",
  "requires_confirmation": false,
  "actions": [
    {
      "op": "DELETE",
      "table": "board_labels",
      "where": {"board_id": ":current_board_id", "label": "draft"}
    }
  ]
}
```"""
    ))

    examples.append(make_example(
        "tag all Design notebooks as 'needs-review'",
        """I'll add the 'needs-review' label to all Design notebooks.

```json
{
  "intent": "bulk_add_label",
  "requires_confirmation": true,
  "actions": [
    {
      "op": "SELECT",
      "purpose": "Find all Design notebooks",
      "sql": "SELECT o.id FROM objects o JOIN workspaces w ON o.workspace_id = w.id JOIN groups g ON w.group_id = g.id WHERE g.name = 'Design' AND o.board_mode = 'notebook'"
    },
    {
      "op": "INSERT_EACH",
      "table": "board_labels",
      "for_each": "{{result[0]}}",
      "values": {
        "id": "{{generate_uuid}}",
        "board_id": "{{item.id}}",
        "label": "needs-review",
        "created_at": "{{now}}"
      }
    }
  ],
  "confirmation": "Add 'needs-review' label to all Design notebooks?"
}
```"""
    ))

    # === STAR/UNSTAR ===
    examples.append(make_example(
        "star this board",
        """I'll star this board.

```json
{
  "intent": "star",
  "requires_confirmation": false,
  "actions": [
    {
      "op": "UPSERT",
      "table": "board_metadata",
      "values": {"board_id": ":current_board_id", "starred": 1},
      "on_conflict": {"update": {"starred": 1}}
    }
  ]
}
```"""
    ))

    examples.append(make_example(
        "unstar this board",
        """I'll unstar this board.

```json
{
  "intent": "unstar",
  "requires_confirmation": false,
  "actions": [
    {
      "op": "UPDATE",
      "table": "board_metadata",
      "set": {"starred": 0},
      "where": {"board_id": ":current_board_id"}
    }
  ]
}
```"""
    ))

    # === MARK AS TEMPLATE ===
    examples.append(make_example(
        "mark this as a template",
        """I'll mark this board as a template.

```json
{
  "intent": "set_template",
  "requires_confirmation": false,
  "actions": [
    {
      "op": "UPSERT",
      "table": "board_metadata",
      "values": {"board_id": ":current_board_id", "template": 1},
      "on_conflict": {"update": {"template": 1}}
    }
  ]
}
```"""
    ))

    # === ARCHIVE ===
    examples.append(make_example(
        "archive this board",
        """I'll archive this board.

```json
{
  "intent": "archive",
  "requires_confirmation": true,
  "actions": [
    {
      "op": "UPDATE",
      "table": "objects",
      "set": {"archived": 1, "updated_at": "{{now}}"},
      "where": {"id": ":current_board_id"}
    }
  ],
  "confirmation": "Archive this board? It will be hidden from normal views."
}
```"""
    ))

    examples.append(make_example(
        "unarchive the Auth Flow board",
        """I'll unarchive the Auth Flow board.

```json
{
  "intent": "unarchive",
  "requires_confirmation": false,
  "actions": [
    {
      "op": "UPDATE",
      "table": "objects",
      "set": {"archived": 0, "updated_at": "{{now}}"},
      "where": {"name": "Auth Flow", "archived": 1}
    }
  ]
}
```"""
    ))

    # === DELETE ===
    examples.append(make_example(
        "delete the empty test workspace",
        """I'll delete the test workspace. This requires confirmation.

```json
{
  "intent": "delete_workspace",
  "requires_confirmation": true,
  "requires_empty_check": true,
  "actions": [
    {
      "op": "SELECT",
      "purpose": "Check if workspace is empty",
      "sql": "SELECT COUNT(*) as count FROM objects WHERE workspace_id = (SELECT id FROM workspaces WHERE name = 'test')"
    },
    {
      "op": "DELETE",
      "table": "workspaces",
      "where": {"name": "test"},
      "condition": "{{result[0].count}} == 0"
    }
  ],
  "confirmation": "Delete workspace 'test'? This cannot be undone."
}
```"""
    ))

    # === RATING ===
    examples.append(make_example(
        "rate this board 5 stars",
        """I'll rate this board.

```json
{
  "intent": "rate",
  "requires_confirmation": false,
  "actions": [
    {
      "op": "UPSERT",
      "table": "board_metadata",
      "values": {"board_id": ":current_board_id", "rating": 5},
      "on_conflict": {"update": {"rating": 5}}
    }
  ]
}
```"""
    ))

    # === More CREATE variations ===
    for group in GROUPS[:3]:
        examples.append(make_example(
            f"create a new workspace in {group}",
            f"""I'll create a new workspace in {group}.

```json
{{
  "intent": "create_workspace",
  "requires_confirmation": true,
  "actions": [
    {{
      "op": "SELECT",
      "purpose": "Find {group} group ID",
      "sql": "SELECT id FROM groups WHERE name = '{group}'"
    }},
    {{
      "op": "INSERT",
      "table": "workspaces",
      "values": {{
        "id": "{{{{generate_uuid}}}}",
        "group_id": "{{{{result[0].id}}}}",
        "name": "New Workspace",
        "created_at": "{{{{now}}}}"
      }}
    }}
  ],
  "confirmation": "Create new workspace in {group}? You can rename it after."
}}
```"""
        ))

    examples.append(make_example(
        "add a board called Design System",
        """I'll create a new board called Design System.

```json
{
  "intent": "create_board",
  "requires_confirmation": true,
  "actions": [
    {
      "op": "INSERT",
      "table": "objects",
      "values": {
        "id": "{{generate_uuid}}",
        "workspace_id": ":current_workspace_id",
        "type": "whiteboard",
        "name": "Design System",
        "board_mode": "notebook",
        "archived": 0,
        "created_at": "{{now}}"
      }
    }
  ],
  "confirmation": "Create board 'Design System' in current workspace?"
}
```"""
    ))

    examples.append(make_example(
        "new freeform board for brainstorming",
        """I'll create a freeform board for brainstorming.

```json
{
  "intent": "create_board",
  "requires_confirmation": true,
  "actions": [
    {
      "op": "INSERT",
      "table": "objects",
      "values": {
        "id": "{{generate_uuid}}",
        "workspace_id": ":current_workspace_id",
        "type": "whiteboard",
        "name": "Brainstorming",
        "board_mode": "freeform",
        "archived": 0,
        "created_at": "{{now}}"
      }
    }
  ],
  "confirmation": "Create freeform board 'Brainstorming'?"
}
```"""
    ))

    # === More MOVE variations ===
    examples.append(make_example(
        "move all files to Archive",
        """I'll move all files to the Archive workspace.

```json
{
  "intent": "bulk_move",
  "requires_confirmation": true,
  "actions": [
    {
      "op": "SELECT",
      "purpose": "Find Archive workspace",
      "sql": "SELECT id FROM workspaces WHERE name = 'Archive' LIMIT 1"
    },
    {
      "op": "SELECT",
      "purpose": "Find all files",
      "sql": "SELECT id, name FROM objects WHERE type = 'file'"
    },
    {
      "op": "UPDATE",
      "table": "objects",
      "set": {"workspace_id": "{{result[0].id}}", "updated_at": "{{now}}"},
      "where": {"type": "file"}
    }
  ],
  "confirmation": "Move {{result[1].length}} files to Archive?"
}
```"""
    ))

    # === More LABEL variations ===
    for label in LABELS[:3]:
        examples.append(make_example(
            f"label this board as {label}",
            f"""I'll add the '{label}' label.

```json
{{
  "intent": "add_label",
  "requires_confirmation": false,
  "actions": [
    {{
      "op": "INSERT",
      "table": "board_labels",
      "values": {{
        "id": "{{{{generate_uuid}}}}",
        "board_id": ":current_board_id",
        "label": "{label}",
        "created_at": "{{{{now}}}}"
      }},
      "on_conflict": "ignore"
    }}
  ]
}}
```"""
        ))

    examples.append(make_example(
        "clear all labels from this board",
        """I'll remove all labels from this board.

```json
{
  "intent": "clear_labels",
  "requires_confirmation": true,
  "actions": [
    {
      "op": "DELETE",
      "table": "board_labels",
      "where": {"board_id": ":current_board_id"}
    }
  ],
  "confirmation": "Remove all labels from this board?"
}
```"""
    ))

    # === DESCRIPTION ===
    examples.append(make_example(
        "set workspace description to 'API documentation and examples'",
        """I'll update the workspace description.

```json
{
  "intent": "update_description",
  "requires_confirmation": false,
  "actions": [
    {
      "op": "UPDATE",
      "table": "workspaces",
      "set": {"description": "API documentation and examples"},
      "where": {"id": ":current_workspace_id"}
    }
  ]
}
```"""
    ))

    # === CREATE GROUP ===
    examples.append(make_example(
        "create a new group called Research",
        """I'll create a new group.

```json
{
  "intent": "create_group",
  "requires_confirmation": true,
  "actions": [
    {
      "op": "INSERT",
      "table": "groups",
      "values": {
        "id": "{{generate_uuid}}",
        "name": "Research",
        "created_at": "{{now}}"
      }
    }
  ],
  "confirmation": "Create group 'Research'?"
}
```"""
    ))

    examples.append(make_example(
        "add group Operations with 🔧 icon",
        """I'll create the Operations group with an icon.

```json
{
  "intent": "create_group",
  "requires_confirmation": true,
  "actions": [
    {
      "op": "INSERT",
      "table": "groups",
      "values": {
        "id": "{{generate_uuid}}",
        "name": "Operations",
        "icon": "🔧",
        "created_at": "{{now}}"
      }
    }
  ],
  "confirmation": "Create group 'Operations' with 🔧 icon?"
}
```"""
    ))

    # === RENAME variations ===
    examples.append(make_example(
        "rename Engineering group to Development",
        """I'll rename the group.

```json
{
  "intent": "rename",
  "requires_confirmation": true,
  "actions": [
    {
      "op": "UPDATE",
      "table": "groups",
      "set": {"name": "Development"},
      "where": {"name": "Engineering"}
    }
  ],
  "confirmation": "Rename 'Engineering' group to 'Development'?"
}
```"""
    ))

    return examples


def generate_workflow_examples() -> list:
    """Generate multi-step workflow examples."""
    examples = []

    # === COPY BOARD ===
    examples.append(make_example(
        "copy this board to Marketing workspace",
        """I'll copy this board to Marketing.

```json
{
  "intent": "copy_board",
  "requires_confirmation": true,
  "actions": [
    {
      "op": "SELECT",
      "purpose": "Get current board data",
      "sql": "SELECT * FROM objects WHERE id = :current_board_id"
    },
    {
      "op": "SELECT",
      "purpose": "Find Marketing workspace",
      "sql": "SELECT w.id FROM workspaces w JOIN groups g ON w.group_id = g.id WHERE g.name = 'Marketing' LIMIT 1"
    },
    {
      "op": "INSERT",
      "table": "objects",
      "values": {
        "id": "{{generate_uuid}}",
        "workspace_id": "{{result[1].id}}",
        "type": "{{result[0].type}}",
        "name": "{{result[0].name}} (Copy)",
        "board_mode": "{{result[0].board_mode}}",
        "archived": 0,
        "created_at": "{{now}}"
      }
    },
    {
      "op": "SELECT",
      "purpose": "Get cells to copy",
      "sql": "SELECT * FROM notebook_cells WHERE board_id = :current_board_id"
    },
    {
      "op": "INSERT_EACH",
      "table": "notebook_cells",
      "for_each": "{{result[3]}}",
      "values": {
        "id": "{{generate_uuid}}",
        "board_id": "{{result[2].id}}",
        "cell_type": "{{item.cell_type}}",
        "content": "{{item.content}}",
        "cell_order": "{{item.cell_order}}",
        "created_at": "{{now}}"
      }
    }
  ],
  "confirmation": "Copy this board (including all cells) to Marketing?"
}
```"""
    ))

    # === DUPLICATE TEMPLATE ===
    examples.append(make_example(
        "create a new board from the onboarding template",
        """I'll create a new board from the onboarding template.

```json
{
  "intent": "copy_template",
  "requires_confirmation": true,
  "actions": [
    {
      "op": "SELECT",
      "purpose": "Find onboarding template",
      "sql": "SELECT o.* FROM objects o JOIN board_metadata bm ON bm.board_id = o.id WHERE o.name LIKE '%onboarding%' AND bm.template = 1 LIMIT 1"
    },
    {
      "op": "INSERT",
      "table": "objects",
      "values": {
        "id": "{{generate_uuid}}",
        "workspace_id": "{{result[0].workspace_id}}",
        "type": "{{result[0].type}}",
        "name": "New Onboarding",
        "board_mode": "{{result[0].board_mode}}",
        "created_at": "{{now}}"
      }
    },
    {
      "op": "SELECT",
      "purpose": "Get template cells",
      "sql": "SELECT * FROM notebook_cells WHERE board_id = '{{result[0].id}}' ORDER BY cell_order"
    },
    {
      "op": "INSERT_EACH",
      "table": "notebook_cells",
      "for_each": "{{result[2]}}",
      "values": {
        "id": "{{generate_uuid}}",
        "board_id": "{{result[1].id}}",
        "cell_type": "{{item.cell_type}}",
        "content": "{{item.content}}",
        "cell_order": "{{item.cell_order}}",
        "created_at": "{{now}}"
      }
    }
  ],
  "confirmation": "Create new board from onboarding template?"
}
```"""
    ))

    # === CREATE WORKSPACE WITH BOARDS ===
    examples.append(make_example(
        "create a Q2 Planning workspace in Product with a roadmap board",
        """I'll create the workspace and initial board.

```json
{
  "intent": "create_workspace_with_content",
  "requires_confirmation": true,
  "actions": [
    {
      "op": "SELECT",
      "purpose": "Find Product group",
      "sql": "SELECT id FROM groups WHERE name = 'Product'"
    },
    {
      "op": "INSERT",
      "table": "workspaces",
      "values": {
        "id": "{{generate_uuid}}",
        "group_id": "{{result[0].id}}",
        "name": "Q2 Planning",
        "created_at": "{{now}}"
      },
      "save_as": "new_workspace"
    },
    {
      "op": "INSERT",
      "table": "objects",
      "values": {
        "id": "{{generate_uuid}}",
        "workspace_id": "{{new_workspace.id}}",
        "type": "whiteboard",
        "name": "Roadmap",
        "board_mode": "notebook",
        "created_at": "{{now}}"
      }
    }
  ],
  "confirmation": "Create workspace 'Q2 Planning' in Product with a 'Roadmap' board?"
}
```"""
    ))

    # === ARCHIVE OLD BOARDS ===
    examples.append(make_example(
        "archive all boards older than 90 days",
        """I'll archive boards not updated in 90 days.

```json
{
  "intent": "bulk_archive",
  "requires_confirmation": true,
  "actions": [
    {
      "op": "SELECT",
      "purpose": "Find old boards",
      "sql": "SELECT id, name FROM objects WHERE type = 'whiteboard' AND archived = 0 AND updated_at < strftime('%s', 'now', '-90 days')"
    },
    {
      "op": "UPDATE",
      "table": "objects",
      "set": {"archived": 1, "updated_at": "{{now}}"},
      "where": {"id": {"$in": "{{result[0].map(r => r.id)}}"}}
    }
  ],
  "confirmation": "Archive {{result[0].length}} boards not updated in 90 days?",
  "preview": "{{result[0].map(r => r.name).join(', ')}}"
}
```"""
    ))

    # === BULK LABEL ===
    examples.append(make_example(
        "add label 'legacy' to all boards not accessed this year",
        """I'll add the 'legacy' label to old boards.

```json
{
  "intent": "bulk_label",
  "requires_confirmation": true,
  "actions": [
    {
      "op": "SELECT",
      "purpose": "Find boards not accessed this year",
      "sql": "SELECT id, name FROM objects WHERE type = 'whiteboard' AND updated_at < strftime('%s', 'now', 'start of year')"
    },
    {
      "op": "INSERT_EACH",
      "table": "board_labels",
      "for_each": "{{result[0]}}",
      "values": {
        "id": "{{generate_uuid}}",
        "board_id": "{{item.id}}",
        "label": "legacy",
        "created_at": "{{now}}"
      },
      "on_conflict": "ignore"
    }
  ],
  "confirmation": "Add 'legacy' label to {{result[0].length}} boards?"
}
```"""
    ))

    # === MOVE ALL NOTEBOOKS ===
    examples.append(make_example(
        "move all Design notebooks to Archive workspace",
        """I'll move all Design notebooks to Archive.

```json
{
  "intent": "bulk_move",
  "requires_confirmation": true,
  "actions": [
    {
      "op": "SELECT",
      "purpose": "Find Archive workspace",
      "sql": "SELECT id FROM workspaces WHERE name = 'Archive' LIMIT 1"
    },
    {
      "op": "SELECT",
      "purpose": "Find Design notebooks",
      "sql": "SELECT o.id, o.name FROM objects o JOIN workspaces w ON o.workspace_id = w.id JOIN groups g ON w.group_id = g.id WHERE g.name = 'Design' AND o.board_mode = 'notebook'"
    },
    {
      "op": "UPDATE",
      "table": "objects",
      "set": {"workspace_id": "{{result[0].id}}", "updated_at": "{{now}}"},
      "where": {"id": {"$in": "{{result[1].map(r => r.id)}}"}}
    }
  ],
  "confirmation": "Move {{result[1].length}} notebooks from Design to Archive?"
}
```"""
    ))

    # === COMPLEX: CREATE, COPY, RENAME ===
    examples.append(make_example(
        "create workspace Q1 Planning, copy the kickoff template there, and rename it to Q1 Kickoff",
        """I'll create the workspace, copy the template, and rename it.

```json
{
  "intent": "workflow_create_copy_rename",
  "requires_confirmation": true,
  "actions": [
    {
      "op": "SELECT",
      "purpose": "Find group for new workspace (default to Product)",
      "sql": "SELECT id FROM groups WHERE name = 'Product' LIMIT 1"
    },
    {
      "op": "INSERT",
      "table": "workspaces",
      "values": {
        "id": "{{generate_uuid}}",
        "group_id": "{{result[0].id}}",
        "name": "Q1 Planning",
        "created_at": "{{now}}"
      },
      "save_as": "new_workspace"
    },
    {
      "op": "SELECT",
      "purpose": "Find kickoff template",
      "sql": "SELECT o.* FROM objects o JOIN board_metadata bm ON bm.board_id = o.id WHERE o.name LIKE '%kickoff%' AND bm.template = 1 LIMIT 1"
    },
    {
      "op": "INSERT",
      "table": "objects",
      "values": {
        "id": "{{generate_uuid}}",
        "workspace_id": "{{new_workspace.id}}",
        "type": "{{result[2].type}}",
        "name": "Q1 Kickoff",
        "board_mode": "{{result[2].board_mode}}",
        "created_at": "{{now}}"
      },
      "save_as": "new_board"
    },
    {
      "op": "SELECT",
      "purpose": "Get template cells",
      "sql": "SELECT * FROM notebook_cells WHERE board_id = '{{result[2].id}}' ORDER BY cell_order"
    },
    {
      "op": "INSERT_EACH",
      "table": "notebook_cells",
      "for_each": "{{result[4]}}",
      "values": {
        "id": "{{generate_uuid}}",
        "board_id": "{{new_board.id}}",
        "cell_type": "{{item.cell_type}}",
        "content": "{{item.content}}",
        "cell_order": "{{item.cell_order}}",
        "created_at": "{{now}}"
      }
    }
  ],
  "confirmation": "Create 'Q1 Planning' workspace, copy kickoff template as 'Q1 Kickoff'?"
}
```"""
    ))

    # === STAR ALL MERMAID BOARDS ===
    examples.append(make_example(
        "star all boards with mermaid diagrams",
        """I'll star all boards that have mermaid cells.

```json
{
  "intent": "bulk_star",
  "requires_confirmation": true,
  "actions": [
    {
      "op": "SELECT",
      "purpose": "Find boards with mermaid",
      "sql": "SELECT DISTINCT o.id, o.name FROM objects o JOIN notebook_cells nc ON nc.board_id = o.id WHERE nc.cell_type = 'mermaid'"
    },
    {
      "op": "UPSERT_EACH",
      "table": "board_metadata",
      "for_each": "{{result[0]}}",
      "values": {"board_id": "{{item.id}}", "starred": 1},
      "on_conflict": {"update": {"starred": 1}}
    }
  ],
  "confirmation": "Star {{result[0].length}} boards with mermaid diagrams?"
}
```"""
    ))

    # === CLEANUP WORKFLOW ===
    examples.append(make_example(
        "clean up: archive boards with no cells and remove empty workspaces",
        """I'll clean up empty boards and workspaces.

```json
{
  "intent": "cleanup",
  "requires_confirmation": true,
  "actions": [
    {
      "op": "SELECT",
      "purpose": "Find boards with no cells",
      "sql": "SELECT o.id, o.name FROM objects o WHERE o.board_mode = 'notebook' AND o.id NOT IN (SELECT DISTINCT board_id FROM notebook_cells)"
    },
    {
      "op": "UPDATE",
      "table": "objects",
      "set": {"archived": 1},
      "where": {"id": {"$in": "{{result[0].map(r => r.id)}}"}}
    },
    {
      "op": "SELECT",
      "purpose": "Find empty workspaces",
      "sql": "SELECT w.id, w.name FROM workspaces w WHERE w.id NOT IN (SELECT DISTINCT workspace_id FROM objects WHERE archived = 0)"
    },
    {
      "op": "DELETE",
      "table": "workspaces",
      "where": {"id": {"$in": "{{result[2].map(r => r.id)}}"}}
    }
  ],
  "confirmation": "Archive {{result[0].length}} empty boards and delete {{result[2].length}} empty workspaces?"
}
```"""
    ))

    # === More COPY variations ===
    examples.append(make_example(
        "duplicate this board",
        """I'll duplicate this board in the same workspace.

```json
{
  "intent": "duplicate",
  "requires_confirmation": true,
  "actions": [
    {
      "op": "SELECT",
      "purpose": "Get current board",
      "sql": "SELECT * FROM objects WHERE id = :current_board_id"
    },
    {
      "op": "INSERT",
      "table": "objects",
      "values": {
        "id": "{{generate_uuid}}",
        "workspace_id": "{{result[0].workspace_id}}",
        "type": "{{result[0].type}}",
        "name": "{{result[0].name}} (Copy)",
        "board_mode": "{{result[0].board_mode}}",
        "created_at": "{{now}}"
      },
      "save_as": "new_board"
    },
    {
      "op": "SELECT",
      "purpose": "Get cells",
      "sql": "SELECT * FROM notebook_cells WHERE board_id = :current_board_id ORDER BY cell_order"
    },
    {
      "op": "INSERT_EACH",
      "table": "notebook_cells",
      "for_each": "{{result[2]}}",
      "values": {
        "id": "{{generate_uuid}}",
        "board_id": "{{new_board.id}}",
        "cell_type": "{{item.cell_type}}",
        "content": "{{item.content}}",
        "cell_order": "{{item.cell_order}}",
        "created_at": "{{now}}"
      }
    }
  ],
  "confirmation": "Duplicate this board?"
}
```"""
    ))

    examples.append(make_example(
        "copy all templates to a new Templates workspace",
        """I'll copy all template boards to a new workspace.

```json
{
  "intent": "consolidate_templates",
  "requires_confirmation": true,
  "actions": [
    {
      "op": "SELECT",
      "purpose": "Find or create Templates workspace",
      "sql": "SELECT id FROM workspaces WHERE name = 'Templates' LIMIT 1"
    },
    {
      "op": "SELECT",
      "purpose": "Find all templates",
      "sql": "SELECT o.* FROM objects o JOIN board_metadata bm ON bm.board_id = o.id WHERE bm.template = 1"
    },
    {
      "op": "INSERT_EACH",
      "table": "objects",
      "for_each": "{{result[1]}}",
      "values": {
        "id": "{{generate_uuid}}",
        "workspace_id": "{{result[0].id}}",
        "type": "{{item.type}}",
        "name": "{{item.name}}",
        "board_mode": "{{item.board_mode}}",
        "created_at": "{{now}}"
      }
    }
  ],
  "confirmation": "Copy {{result[1].length}} templates to Templates workspace?"
}
```"""
    ))

    # === ORGANIZE workflows ===
    examples.append(make_example(
        "organize: move all draft boards to a Drafts workspace",
        """I'll move all draft-labeled boards to a Drafts workspace.

```json
{
  "intent": "organize_drafts",
  "requires_confirmation": true,
  "actions": [
    {
      "op": "SELECT",
      "purpose": "Find or get Drafts workspace",
      "sql": "SELECT id FROM workspaces WHERE name = 'Drafts' LIMIT 1"
    },
    {
      "op": "SELECT",
      "purpose": "Find all draft boards",
      "sql": "SELECT o.id, o.name FROM objects o JOIN board_labels bl ON bl.board_id = o.id WHERE bl.label = 'draft'"
    },
    {
      "op": "UPDATE",
      "table": "objects",
      "set": {"workspace_id": "{{result[0].id}}", "updated_at": "{{now}}"},
      "where": {"id": {"$in": "{{result[1].map(r => r.id)}}"}}
    }
  ],
  "confirmation": "Move {{result[1].length}} draft boards to Drafts workspace?"
}
```"""
    ))

    examples.append(make_example(
        "merge workspace 'Old Designs' into 'Brand Kit'",
        """I'll merge workspaces by moving all content.

```json
{
  "intent": "merge_workspaces",
  "requires_confirmation": true,
  "actions": [
    {
      "op": "SELECT",
      "purpose": "Find source workspace",
      "sql": "SELECT id FROM workspaces WHERE name = 'Old Designs'"
    },
    {
      "op": "SELECT",
      "purpose": "Find target workspace",
      "sql": "SELECT id FROM workspaces WHERE name = 'Brand Kit'"
    },
    {
      "op": "SELECT",
      "purpose": "Get boards to move",
      "sql": "SELECT id, name FROM objects WHERE workspace_id = '{{result[0].id}}'"
    },
    {
      "op": "UPDATE",
      "table": "objects",
      "set": {"workspace_id": "{{result[1].id}}", "updated_at": "{{now}}"},
      "where": {"workspace_id": "{{result[0].id}}"}
    },
    {
      "op": "DELETE",
      "table": "workspaces",
      "where": {"id": "{{result[0].id}}"}
    }
  ],
  "confirmation": "Move {{result[2].length}} items from 'Old Designs' to 'Brand Kit' and delete 'Old Designs'?"
}
```"""
    ))

    # === BATCH operations ===
    examples.append(make_example(
        "unstar all boards",
        """I'll unstar all boards.

```json
{
  "intent": "bulk_unstar",
  "requires_confirmation": true,
  "actions": [
    {
      "op": "SELECT",
      "purpose": "Count starred boards",
      "sql": "SELECT COUNT(*) as count FROM board_metadata WHERE starred = 1"
    },
    {
      "op": "UPDATE",
      "table": "board_metadata",
      "set": {"starred": 0},
      "where": {"starred": 1}
    }
  ],
  "confirmation": "Unstar {{result[0].count}} boards?"
}
```"""
    ))

    examples.append(make_example(
        "mark all reviewed boards as approved",
        """I'll change labels from 'reviewed' to 'approved'.

```json
{
  "intent": "relabel",
  "requires_confirmation": true,
  "actions": [
    {
      "op": "SELECT",
      "purpose": "Find reviewed boards",
      "sql": "SELECT board_id FROM board_labels WHERE label = 'reviewed'"
    },
    {
      "op": "UPDATE",
      "table": "board_labels",
      "set": {"label": "approved"},
      "where": {"label": "reviewed"}
    }
  ],
  "confirmation": "Change {{result[0].length}} boards from 'reviewed' to 'approved'?"
}
```"""
    ))

    # === BACKUP/EXPORT workflows ===
    examples.append(make_example(
        "prepare for archive: label all old boards and move to Archive",
        """I'll label and archive old boards.

```json
{
  "intent": "archive_old",
  "requires_confirmation": true,
  "actions": [
    {
      "op": "SELECT",
      "purpose": "Find old boards",
      "sql": "SELECT id, name FROM objects WHERE type = 'whiteboard' AND updated_at < strftime('%s', 'now', '-180 days') AND archived = 0"
    },
    {
      "op": "INSERT_EACH",
      "table": "board_labels",
      "for_each": "{{result[0]}}",
      "values": {
        "id": "{{generate_uuid}}",
        "board_id": "{{item.id}}",
        "label": "archived-2024",
        "created_at": "{{now}}"
      },
      "on_conflict": "ignore"
    },
    {
      "op": "UPDATE",
      "table": "objects",
      "set": {"archived": 1, "updated_at": "{{now}}"},
      "where": {"id": {"$in": "{{result[0].map(r => r.id)}}"}}
    }
  ],
  "confirmation": "Label and archive {{result[0].length}} boards older than 6 months?"
}
```"""
    ))

    examples.append(make_example(
        "set up new project: create workspace, add planning and notes boards",
        """I'll set up a new project workspace with starter boards.

```json
{
  "intent": "setup_project",
  "requires_confirmation": true,
  "actions": [
    {
      "op": "SELECT",
      "purpose": "Find Product group",
      "sql": "SELECT id FROM groups WHERE name = 'Product' LIMIT 1"
    },
    {
      "op": "INSERT",
      "table": "workspaces",
      "values": {
        "id": "{{generate_uuid}}",
        "group_id": "{{result[0].id}}",
        "name": "New Project",
        "created_at": "{{now}}"
      },
      "save_as": "workspace"
    },
    {
      "op": "INSERT",
      "table": "objects",
      "values": {
        "id": "{{generate_uuid}}",
        "workspace_id": "{{workspace.id}}",
        "type": "whiteboard",
        "name": "Planning",
        "board_mode": "notebook",
        "created_at": "{{now}}"
      }
    },
    {
      "op": "INSERT",
      "table": "objects",
      "values": {
        "id": "{{generate_uuid}}",
        "workspace_id": "{{workspace.id}}",
        "type": "whiteboard",
        "name": "Notes",
        "board_mode": "notebook",
        "created_at": "{{now}}"
      }
    },
    {
      "op": "INSERT",
      "table": "objects",
      "values": {
        "id": "{{generate_uuid}}",
        "workspace_id": "{{workspace.id}}",
        "type": "whiteboard",
        "name": "Brainstorm",
        "board_mode": "freeform",
        "created_at": "{{now}}"
      }
    }
  ],
  "confirmation": "Create 'New Project' workspace with Planning, Notes, and Brainstorm boards?"
}
```"""
    ))

    return examples


def main():
    output_dir = Path("/mnt/user-data/outputs/data/cyan-sql-v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate all examples
    query_examples = generate_query_examples()
    mutation_examples = generate_mutation_examples()
    workflow_examples = generate_workflow_examples()

    all_examples = query_examples + mutation_examples + workflow_examples
    random.shuffle(all_examples)

    # Split 90/10 for train/valid
    split_idx = int(len(all_examples) * 0.9)
    train_examples = all_examples[:split_idx]
    valid_examples = all_examples[split_idx:]

    # Write train.jsonl
    with open(output_dir / "train.jsonl", "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")

    # Write valid.jsonl
    with open(output_dir / "valid.jsonl", "w") as f:
        for ex in valid_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Generated training data:")
    print(f"  Queries:   {len(query_examples)}")
    print(f"  Mutations: {len(mutation_examples)}")
    print(f"  Workflows: {len(workflow_examples)}")
    print(f"  Total:     {len(all_examples)}")
    print(f"  Train:     {len(train_examples)}")
    print(f"  Valid:     {len(valid_examples)}")
    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()