#!/usr/bin/env python3
"""
Generate SQL training data for cyan-lens-sql LoRA adapter.

This creates training data in the SAME format as your existing train.jsonl,
so it can be used with your existing train.py script.

The adapter will be trained on TOP of your merged cyan-lens model,
adding SQL generation capability without losing existing skills.

Usage:
    python generate_sql_data.py

Output:
    data/sql/train.jsonl   (90% - training)
    data/sql/valid.jsonl   (10% - validation)
"""

import json
import random
from pathlib import Path
from typing import List, Tuple

# System prompt matching your existing format
SYSTEM_PROMPT = """You are CyanLens, an AI assistant for Cyan workspace. You help users search and navigate their workspace by generating SQLite queries.

Available tables:
- groups(id, name, icon, color, created_at)
- workspaces(id, group_id, name, created_at)
- objects(id, workspace_id, type, name, board_mode, created_at) - boards, chats, files
- notebook_cells(id, board_id, cell_type, content, cell_order, created_at)
- whiteboard_elements(id, board_id, element_type, x, y, width, height)

Key relationships:
- To find boards in a group: JOIN objects → workspaces → groups
- objects.type can be 'whiteboard', 'chat', or 'file'
- notebook_cells.cell_type can be 'markdown', 'mermaid', or 'code'
- objects.board_mode can be 'freeform' or 'notebook'

Generate valid SQLite SELECT queries. Always use proper JOINs for group filtering."""

# Training pairs: (natural query, SQL response)
SQL_PAIRS: List[Tuple[str, str]] = [
    # ===== GROUP QUERIES =====
    ("list all groups",
     "```sql\nSELECT id, name, icon, color FROM groups ORDER BY name;\n```"),

    ("show me all groups",
     "```sql\nSELECT id, name, icon, color FROM groups ORDER BY name;\n```"),

    ("what groups exist",
     "```sql\nSELECT name FROM groups ORDER BY name;\n```"),

    ("find the Design group",
     "```sql\nSELECT id, name, icon, color FROM groups WHERE name = 'Design';\n```"),

    ("find groups with Engineering in the name",
     "```sql\nSELECT id, name FROM groups WHERE name LIKE '%Engineering%';\n```"),

    ("how many groups are there",
     "```sql\nSELECT COUNT(*) as group_count FROM groups;\n```"),

    # ===== WORKSPACE QUERIES =====
    ("list all workspaces",
     "```sql\nSELECT w.id, w.name, g.name as group_name FROM workspaces w JOIN groups g ON w.group_id = g.id ORDER BY g.name, w.name;\n```"),

    ("show workspaces in the Design group",
     "```sql\nSELECT w.id, w.name FROM workspaces w JOIN groups g ON w.group_id = g.id WHERE g.name = 'Design';\n```"),

    ("find workspaces in Engineering",
     "```sql\nSELECT w.id, w.name FROM workspaces w JOIN groups g ON w.group_id = g.id WHERE g.name = 'Engineering';\n```"),

    ("what workspaces are in Marketing",
     "```sql\nSELECT w.id, w.name FROM workspaces w JOIN groups g ON w.group_id = g.id WHERE g.name = 'Marketing';\n```"),

    ("count workspaces per group",
     "```sql\nSELECT g.name, COUNT(w.id) as workspace_count FROM groups g LEFT JOIN workspaces w ON w.group_id = g.id GROUP BY g.id, g.name ORDER BY workspace_count DESC;\n```"),

    # ===== BOARD QUERIES - CRITICAL JOINs =====
    ("find all boards",
     "```sql\nSELECT o.id, o.name, o.type, o.board_mode FROM objects o WHERE o.type = 'whiteboard' ORDER BY o.created_at DESC;\n```"),

    ("list Design boards",
     "```sql\nSELECT o.id, o.name, o.board_mode FROM objects o JOIN workspaces w ON o.workspace_id = w.id JOIN groups g ON w.group_id = g.id WHERE g.name = 'Design' AND o.type = 'whiteboard';\n```"),

    ("show me boards in Design",
     "```sql\nSELECT o.id, o.name, o.board_mode FROM objects o JOIN workspaces w ON o.workspace_id = w.id JOIN groups g ON w.group_id = g.id WHERE g.name = 'Design' AND o.type = 'whiteboard';\n```"),

    ("find Engineering boards",
     "```sql\nSELECT o.id, o.name, o.board_mode FROM objects o JOIN workspaces w ON o.workspace_id = w.id JOIN groups g ON w.group_id = g.id WHERE g.name = 'Engineering' AND o.type = 'whiteboard';\n```"),

    ("boards in the Marketing group",
     "```sql\nSELECT o.id, o.name, o.board_mode FROM objects o JOIN workspaces w ON o.workspace_id = w.id JOIN groups g ON w.group_id = g.id WHERE g.name = 'Marketing' AND o.type = 'whiteboard';\n```"),

    ("show Product boards",
     "```sql\nSELECT o.id, o.name, o.board_mode FROM objects o JOIN workspaces w ON o.workspace_id = w.id JOIN groups g ON w.group_id = g.id WHERE g.name = 'Product' AND o.type = 'whiteboard';\n```"),

    ("find Sales boards",
     "```sql\nSELECT o.id, o.name, o.board_mode FROM objects o JOIN workspaces w ON o.workspace_id = w.id JOIN groups g ON w.group_id = g.id WHERE g.name = 'Sales' AND o.type = 'whiteboard';\n```"),

    ("Design group boards",
     "```sql\nSELECT o.id, o.name FROM objects o JOIN workspaces w ON o.workspace_id = w.id JOIN groups g ON w.group_id = g.id WHERE g.name = 'Design' AND o.type = 'whiteboard';\n```"),

    ("all the Engineering boards",
     "```sql\nSELECT o.id, o.name FROM objects o JOIN workspaces w ON o.workspace_id = w.id JOIN groups g ON w.group_id = g.id WHERE g.name = 'Engineering' AND o.type = 'whiteboard';\n```"),

    # ===== BOARD MODE QUERIES =====
    ("find notebook boards",
     "```sql\nSELECT o.id, o.name FROM objects o WHERE o.type = 'whiteboard' AND o.board_mode = 'notebook';\n```"),

    ("show all notebooks",
     "```sql\nSELECT o.id, o.name FROM objects o WHERE o.board_mode = 'notebook';\n```"),

    ("list freeform boards",
     "```sql\nSELECT o.id, o.name FROM objects o WHERE o.type = 'whiteboard' AND o.board_mode = 'freeform';\n```"),

    ("find notebook boards in Engineering",
     "```sql\nSELECT o.id, o.name FROM objects o JOIN workspaces w ON o.workspace_id = w.id JOIN groups g ON w.group_id = g.id WHERE g.name = 'Engineering' AND o.board_mode = 'notebook';\n```"),

    ("notebooks in Design",
     "```sql\nSELECT o.id, o.name FROM objects o JOIN workspaces w ON o.workspace_id = w.id JOIN groups g ON w.group_id = g.id WHERE g.name = 'Design' AND o.board_mode = 'notebook';\n```"),

    # ===== MERMAID/DIAGRAM QUERIES =====
    ("find boards with mermaid diagrams",
     "```sql\nSELECT DISTINCT o.id, o.name FROM objects o JOIN notebook_cells nc ON nc.board_id = o.id WHERE nc.cell_type = 'mermaid';\n```"),

    ("show me boards that have mermaid",
     "```sql\nSELECT DISTINCT o.id, o.name FROM objects o JOIN notebook_cells nc ON nc.board_id = o.id WHERE nc.cell_type = 'mermaid';\n```"),

    ("which boards contain mermaid cells",
     "```sql\nSELECT DISTINCT o.id, o.name FROM objects o JOIN notebook_cells nc ON nc.board_id = o.id WHERE nc.cell_type = 'mermaid';\n```"),

    ("boards with diagrams",
     "```sql\nSELECT DISTINCT o.id, o.name FROM objects o JOIN notebook_cells nc ON nc.board_id = o.id WHERE nc.cell_type = 'mermaid';\n```"),

    ("find mermaid boards in Design",
     "```sql\nSELECT DISTINCT o.id, o.name FROM objects o JOIN notebook_cells nc ON nc.board_id = o.id JOIN workspaces w ON o.workspace_id = w.id JOIN groups g ON w.group_id = g.id WHERE nc.cell_type = 'mermaid' AND g.name = 'Design';\n```"),

    # ===== CODE CELL QUERIES =====
    ("find boards with code",
     "```sql\nSELECT DISTINCT o.id, o.name FROM objects o JOIN notebook_cells nc ON nc.board_id = o.id WHERE nc.cell_type = 'code';\n```"),

    ("show boards that have code cells",
     "```sql\nSELECT DISTINCT o.id, o.name FROM objects o JOIN notebook_cells nc ON nc.board_id = o.id WHERE nc.cell_type = 'code';\n```"),

    ("which notebooks have python code",
     "```sql\nSELECT DISTINCT o.id, o.name FROM objects o JOIN notebook_cells nc ON nc.board_id = o.id WHERE nc.cell_type = 'code' AND nc.content LIKE '%python%';\n```"),

    ("boards with pandas",
     "```sql\nSELECT DISTINCT o.id, o.name FROM objects o JOIN notebook_cells nc ON nc.board_id = o.id WHERE nc.cell_type = 'code' AND nc.content LIKE '%pandas%';\n```"),

    ("find boards with matplotlib",
     "```sql\nSELECT DISTINCT o.id, o.name FROM objects o JOIN notebook_cells nc ON nc.board_id = o.id WHERE nc.cell_type = 'code' AND nc.content LIKE '%matplotlib%';\n```"),

    # ===== TEMPORAL QUERIES =====
    ("show recent boards",
     "```sql\nSELECT o.id, o.name, o.created_at FROM objects o WHERE o.type = 'whiteboard' ORDER BY o.created_at DESC LIMIT 10;\n```"),

    ("find recently created boards",
     "```sql\nSELECT o.id, o.name, o.created_at FROM objects o WHERE o.type = 'whiteboard' ORDER BY o.created_at DESC LIMIT 10;\n```"),

    ("latest boards in Design",
     "```sql\nSELECT o.id, o.name, o.created_at FROM objects o JOIN workspaces w ON o.workspace_id = w.id JOIN groups g ON w.group_id = g.id WHERE g.name = 'Design' AND o.type = 'whiteboard' ORDER BY o.created_at DESC LIMIT 5;\n```"),

    ("oldest boards",
     "```sql\nSELECT o.id, o.name, o.created_at FROM objects o WHERE o.type = 'whiteboard' ORDER BY o.created_at ASC LIMIT 10;\n```"),

    ("boards from last week",
     "```sql\nSELECT o.id, o.name, o.created_at FROM objects o WHERE o.type = 'whiteboard' AND o.created_at > strftime('%s', 'now', '-7 days') ORDER BY o.created_at DESC;\n```"),

    # ===== COUNT/AGGREGATION QUERIES =====
    ("how many boards are there",
     "```sql\nSELECT COUNT(*) as board_count FROM objects WHERE type = 'whiteboard';\n```"),

    ("count boards in each group",
     "```sql\nSELECT g.name, COUNT(o.id) as board_count FROM groups g LEFT JOIN workspaces w ON w.group_id = g.id LEFT JOIN objects o ON o.workspace_id = w.id AND o.type = 'whiteboard' GROUP BY g.id, g.name ORDER BY board_count DESC;\n```"),

    ("how many notebooks exist",
     "```sql\nSELECT COUNT(*) as notebook_count FROM objects WHERE board_mode = 'notebook';\n```"),

    ("count mermaid diagrams",
     "```sql\nSELECT COUNT(*) as mermaid_count FROM notebook_cells WHERE cell_type = 'mermaid';\n```"),

    ("boards per workspace",
     "```sql\nSELECT w.name, COUNT(o.id) as board_count FROM workspaces w LEFT JOIN objects o ON o.workspace_id = w.id AND o.type = 'whiteboard' GROUP BY w.id, w.name ORDER BY board_count DESC;\n```"),

    ("total cells by type",
     "```sql\nSELECT cell_type, COUNT(*) as count FROM notebook_cells GROUP BY cell_type ORDER BY count DESC;\n```"),

    # ===== NAME SEARCH QUERIES =====
    ("find boards named Logo",
     "```sql\nSELECT o.id, o.name FROM objects o WHERE o.type = 'whiteboard' AND o.name LIKE '%Logo%';\n```"),

    ("search for Homepage",
     "```sql\nSELECT o.id, o.name FROM objects o WHERE o.name LIKE '%Homepage%';\n```"),

    ("find boards containing wireframe",
     "```sql\nSELECT o.id, o.name FROM objects o WHERE o.type = 'whiteboard' AND o.name LIKE '%wireframe%';\n```"),

    ("boards with analysis in the name",
     "```sql\nSELECT o.id, o.name FROM objects o WHERE o.type = 'whiteboard' AND LOWER(o.name) LIKE '%analysis%';\n```"),

    ("search boards for API",
     "```sql\nSELECT o.id, o.name FROM objects o WHERE o.type = 'whiteboard' AND o.name LIKE '%API%';\n```"),

    # ===== WORKSPACE-SPECIFIC QUERIES =====
    ("boards in Brand Kit workspace",
     "```sql\nSELECT o.id, o.name FROM objects o JOIN workspaces w ON o.workspace_id = w.id WHERE w.name = 'Brand Kit' AND o.type = 'whiteboard';\n```"),

    ("show Data Engineering boards",
     "```sql\nSELECT o.id, o.name FROM objects o JOIN workspaces w ON o.workspace_id = w.id WHERE w.name = 'Data Engineering' AND o.type = 'whiteboard';\n```"),

    ("find boards in Marketing Site",
     "```sql\nSELECT o.id, o.name FROM objects o JOIN workspaces w ON o.workspace_id = w.id WHERE w.name = 'Marketing Site' AND o.type = 'whiteboard';\n```"),

    # ===== CONTENT SEARCH =====
    ("boards mentioning sales",
     "```sql\nSELECT DISTINCT o.id, o.name FROM objects o JOIN notebook_cells nc ON nc.board_id = o.id WHERE LOWER(nc.content) LIKE '%sales%';\n```"),

    ("find boards about revenue",
     "```sql\nSELECT DISTINCT o.id, o.name FROM objects o JOIN notebook_cells nc ON nc.board_id = o.id WHERE LOWER(nc.content) LIKE '%revenue%';\n```"),

    ("search content for authentication",
     "```sql\nSELECT DISTINCT o.id, o.name FROM objects o JOIN notebook_cells nc ON nc.board_id = o.id WHERE LOWER(nc.content) LIKE '%authentication%';\n```"),

    # ===== COMPLEX MULTI-CONDITION =====
    ("find Design notebooks with mermaid",
     "```sql\nSELECT DISTINCT o.id, o.name FROM objects o JOIN notebook_cells nc ON nc.board_id = o.id JOIN workspaces w ON o.workspace_id = w.id JOIN groups g ON w.group_id = g.id WHERE g.name = 'Design' AND o.board_mode = 'notebook' AND nc.cell_type = 'mermaid';\n```"),

    ("Engineering boards with code cells",
     "```sql\nSELECT DISTINCT o.id, o.name FROM objects o JOIN notebook_cells nc ON nc.board_id = o.id JOIN workspaces w ON o.workspace_id = w.id JOIN groups g ON w.group_id = g.id WHERE g.name = 'Engineering' AND nc.cell_type = 'code';\n```"),

    ("recent notebooks with code",
     "```sql\nSELECT DISTINCT o.id, o.name, o.created_at FROM objects o JOIN notebook_cells nc ON nc.board_id = o.id WHERE o.board_mode = 'notebook' AND nc.cell_type = 'code' ORDER BY o.created_at DESC LIMIT 5;\n```"),

    ("Marketing notebooks with diagrams",
     "```sql\nSELECT DISTINCT o.id, o.name FROM objects o JOIN notebook_cells nc ON nc.board_id = o.id JOIN workspaces w ON o.workspace_id = w.id JOIN groups g ON w.group_id = g.id WHERE g.name = 'Marketing' AND o.board_mode = 'notebook' AND nc.cell_type = 'mermaid';\n```"),

    # ===== WHITEBOARD ELEMENTS =====
    ("boards with many elements",
     "```sql\nSELECT o.id, o.name, COUNT(we.id) as element_count FROM objects o JOIN whiteboard_elements we ON we.board_id = o.id GROUP BY o.id, o.name ORDER BY element_count DESC LIMIT 10;\n```"),

    ("find busy whiteboards",
     "```sql\nSELECT o.id, o.name, COUNT(we.id) as element_count FROM objects o JOIN whiteboard_elements we ON we.board_id = o.id GROUP BY o.id, o.name HAVING COUNT(we.id) > 5 ORDER BY element_count DESC;\n```"),

    # ===== CHAT QUERIES =====
    ("show all chats",
     "```sql\nSELECT o.id, o.name, o.created_at FROM objects o WHERE o.type = 'chat' ORDER BY o.created_at DESC;\n```"),

    ("chats in Sales workspace",
     "```sql\nSELECT o.id, o.name FROM objects o JOIN workspaces w ON o.workspace_id = w.id WHERE o.type = 'chat' AND w.name LIKE '%Sales%';\n```"),

    ("recent chat messages",
     "```sql\nSELECT o.id, o.name, o.created_at FROM objects o WHERE o.type = 'chat' ORDER BY o.created_at DESC LIMIT 20;\n```"),

    # ===== FILE QUERIES =====
    ("list all files",
     "```sql\nSELECT o.id, o.name, o.size, o.hash FROM objects o WHERE o.type = 'file' ORDER BY o.created_at DESC;\n```"),

    ("large files",
     "```sql\nSELECT o.id, o.name, o.size FROM objects o WHERE o.type = 'file' AND o.size > 1000000 ORDER BY o.size DESC;\n```"),

    ("files in Engineering",
     "```sql\nSELECT o.id, o.name, o.size FROM objects o JOIN workspaces w ON o.workspace_id = w.id JOIN groups g ON w.group_id = g.id WHERE o.type = 'file' AND g.name = 'Engineering';\n```"),

    # ===== INFORMAL VARIATIONS =====
    ("gimme Design boards",
     "```sql\nSELECT o.id, o.name FROM objects o JOIN workspaces w ON o.workspace_id = w.id JOIN groups g ON w.group_id = g.id WHERE g.name = 'Design' AND o.type = 'whiteboard';\n```"),

    ("I want to see Marketing boards",
     "```sql\nSELECT o.id, o.name FROM objects o JOIN workspaces w ON o.workspace_id = w.id JOIN groups g ON w.group_id = g.id WHERE g.name = 'Marketing' AND o.type = 'whiteboard';\n```"),

    ("can you show me the boards in Product",
     "```sql\nSELECT o.id, o.name FROM objects o JOIN workspaces w ON o.workspace_id = w.id JOIN groups g ON w.group_id = g.id WHERE g.name = 'Product' AND o.type = 'whiteboard';\n```"),

    ("what boards do we have in Sales?",
     "```sql\nSELECT o.id, o.name FROM objects o JOIN workspaces w ON o.workspace_id = w.id JOIN groups g ON w.group_id = g.id WHERE g.name = 'Sales' AND o.type = 'whiteboard';\n```"),

    ("show me everything in Design",
     "```sql\nSELECT o.id, o.name, o.type FROM objects o JOIN workspaces w ON o.workspace_id = w.id JOIN groups g ON w.group_id = g.id WHERE g.name = 'Design' ORDER BY o.type, o.name;\n```"),
]


def format_example(query: str, sql: str) -> dict:
    """Format as messages for training (matching existing format)."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
            {"role": "assistant", "content": sql}
        ]
    }


def generate_variations(query: str, sql: str) -> List[Tuple[str, str]]:
    """Generate query variations for augmentation."""
    variations = [(query, sql)]

    # Question mark variation
    if not query.endswith("?"):
        variations.append((query + "?", sql))

    # Lowercase variation (if starts with capital)
    if query[0].isupper():
        variations.append((query[0].lower() + query[1:], sql))

    return variations


def main():
    output_dir = Path("data/sql")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_examples = []

    for query, sql in SQL_PAIRS:
        # Add original
        all_examples.append(format_example(query, sql))

        # Add variations with 50% probability
        for var_query, var_sql in generate_variations(query, sql):
            if var_query != query and random.random() < 0.5:
                all_examples.append(format_example(var_query, var_sql))

    # Shuffle
    random.shuffle(all_examples)

    # Split 90/10
    split_idx = int(len(all_examples) * 0.9)
    train_examples = all_examples[:split_idx]
    valid_examples = all_examples[split_idx:]

    # Write training data
    train_path = output_dir / "train.jsonl"
    with open(train_path, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")

    # Write validation data
    valid_path = output_dir / "valid.jsonl"
    with open(valid_path, "w") as f:
        for ex in valid_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"✅ Generated SQL training data:")
    print(f"   Train: {len(train_examples)} examples → {train_path}")
    print(f"   Valid: {len(valid_examples)} examples → {valid_path}")
    print()
    print(f"📋 Next steps:")
    print(f"   1. Train LoRA on your merged model:")
    print(f"      python -m mlx_lm lora \\")
    print(f"        --model blockxaero/cyan-lens \\")
    print(f"        --train --data data/sql \\")
    print(f"        --batch-size 4 --iters 300 \\")
    print(f"        --adapter-path adapters/cyan-lens-sql")
    print()
    print(f"   2. Or train on base Phi-3 for clean separation:")
    print(f"      python -m mlx_lm lora \\")
    print(f"        --model microsoft/Phi-3-mini-4k-instruct \\")
    print(f"        --train --data data/sql \\")
    print(f"        --batch-size 4 --iters 300 \\")
    print(f"        --adapter-path adapters/cyan-lens-sql-base")


if __name__ == "__main__":
    main()