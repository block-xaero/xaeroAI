#!/usr/bin/env python3
"""
Generate training data for design-analyst model.

Usage:
    python scripts/generate_data.py --type mermaid --count 50
    python scripts/generate_data.py --type patterns --count 100
    python scripts/generate_data.py --validate  # Validate existing data

Requires:
    pip install anthropic
    npm install -g @mermaid-js/mermaid-cli  (for validation)
"""

import argparse
import json
import subprocess
import tempfile
import re
import os
from pathlib import Path

try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

SYSTEM_PROMPT = """You are a software design analyst for Cyan, a design-first collaboration tool. You have deep expertise in:
- Design patterns (GoF, enterprise, microservices, Rust idioms)
- Software architecture (clean architecture, DDD, CQRS, event sourcing)
- UML and diagram generation using Mermaid syntax
- Project health analysis from integration data

When asked for diagrams, output valid Mermaid syntax in a code block. Be specific, actionable, and concise."""

MERMAID_PROMPTS = [
    ("sequenceDiagram", "Create a Mermaid sequenceDiagram for {topic}"),
    ("classDiagram", "Create a Mermaid classDiagram for {topic}"),
    ("flowchart", "Create a Mermaid flowchart for {topic}"),
    ("erDiagram", "Create a Mermaid erDiagram for {topic}"),
    ("stateDiagram", "Create a Mermaid stateDiagram-v2 for {topic}"),
]

MERMAID_TOPICS = {
    "sequenceDiagram": [
        "user authentication with JWT",
        "OAuth2 authorization code flow",
        "API request with retry logic",
        "message queue publish/subscribe",
        "database transaction with rollback",
    ],
    "classDiagram": [
        "Repository pattern",
        "Factory pattern",
        "Observer pattern",
        "Strategy pattern",
        "Adapter pattern",
    ],
    "flowchart": [
        "error handling with retry",
        "CI/CD pipeline",
        "user registration",
        "order processing",
        "authentication flow",
    ],
    "erDiagram": [
        "e-commerce orders and customers",
        "blog posts and comments",
        "project management tasks",
        "inventory management",
        "social media users and posts",
    ],
    "stateDiagram": [
        "order lifecycle",
        "connection state machine",
        "user session states",
        "task workflow",
        "payment processing",
    ],
}


def extract_mermaid(text: str) -> str | None:
    """Extract mermaid code from markdown block."""
    match = re.search(r'```mermaid\n(.*?)```', text, re.DOTALL)
    return match.group(1).strip() if match else None


def validate_mermaid(code: str) -> bool:
    """Validate mermaid syntax using mmdc."""
    with tempfile.NamedTemporaryFile(suffix='.mmd', delete=False, mode='w') as f:
        f.write(code)
        f.flush()

        with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as out:
            result = subprocess.run(
                ['mmdc', '-i', f.name, '-o', out.name, '-q'],
                capture_output=True, timeout=30
            )
            Path(f.name).unlink(missing_ok=True)
            Path(out.name).unlink(missing_ok=True)
            return result.returncode == 0


def generate_mermaid_examples(output_file: str, count: int = 50):
    """Generate Mermaid diagram training examples."""
    if not HAS_ANTHROPIC:
        print("❌ anthropic not installed. Run: pip install anthropic")
        return

    client = Anthropic()
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    examples = []

    for diagram_type, template in MERMAID_PROMPTS:
        topics = MERMAID_TOPICS.get(diagram_type, [])

        for topic in topics[:count // len(MERMAID_PROMPTS)]:
            prompt = template.format(topic=topic)
            print(f"Generating: {prompt[:50]}...")

            try:
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1000,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}]
                )

                content = response.content[0].text
                mermaid_code = extract_mermaid(content)

                if mermaid_code and validate_mermaid(mermaid_code):
                    example = {
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": content}
                        ]
                    }
                    examples.append(example)
                    print(f"  ✅ Valid {diagram_type}")
                else:
                    print(f"  ⚠️ Invalid syntax, skipping")

            except Exception as e:
                print(f"  ❌ Error: {e}")

    # Save
    with open(output_path, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')

    print(f"\n✅ Saved {len(examples)} examples to {output_path}")


def validate_training_data(data_file: str):
    """Validate Mermaid syntax in training data."""
    data_path = Path(data_file)
    if not data_path.exists():
        print(f"❌ File not found: {data_file}")
        return

    valid = 0
    invalid = 0

    with open(data_path) as f:
        for i, line in enumerate(f, 1):
            example = json.loads(line)

            # Find assistant response
            for msg in example.get("messages", []):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    mermaid_code = extract_mermaid(content)

                    if mermaid_code:
                        if validate_mermaid(mermaid_code):
                            valid += 1
                        else:
                            invalid += 1
                            print(f"Line {i}: Invalid Mermaid")

    print(f"\nResults: {valid} valid, {invalid} invalid")


def main():
    parser = argparse.ArgumentParser(description="Generate training data")
    parser.add_argument("--type", choices=["mermaid", "patterns"], default="mermaid")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--output", default="data/patterns/generated.jsonl")
    parser.add_argument("--validate", action="store_true", help="Validate existing data")
    parser.add_argument("--data", default="data/mlx_final/train.jsonl", help="Data to validate")
    args = parser.parse_args()

    if args.validate:
        validate_training_data(args.data)
    elif args.type == "mermaid":
        generate_mermaid_examples(args.output, args.count)
    else:
        print("Pattern generation not implemented yet")


if __name__ == "__main__":
    main()