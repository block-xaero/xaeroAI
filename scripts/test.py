#!/usr/bin/env python3
"""
Test design-analyst model.

Usage:
    python scripts/test.py --use-mlx
    python scripts/test.py --model models/design-analyst-v4-q4.gguf
    python scripts/test.py --use-ollama
    python scripts/test.py --test 1  # Run specific test
"""

import argparse
import subprocess
import sys

TEST_CASES = [
    {
        "name": "Ticket Health Analysis",
        "prompt": "<context>Ticket: AUTH-101, Days open: 45, Slack mentions: 67, PRs: 0</context> Analyze this ticket health",
        "check": lambda x: len(x) > 100,
    },
    {
        "name": "Sequence Diagram",
        "prompt": "Create a Mermaid sequenceDiagram for user login with JWT",
        "check": lambda x: "sequenceDiagram" in x,
    },
    {
        "name": "Flowchart",
        "prompt": "Create a Mermaid flowchart for error handling with retry",
        "check": lambda x: "flowchart" in x.lower() or "graph" in x.lower(),
    },
    {
        "name": "Class Diagram",
        "prompt": "Create a Mermaid classDiagram for Repository pattern",
        "check": lambda x: "classDiagram" in x,
    },
    {
        "name": "Design Pattern",
        "prompt": "Explain the Observer pattern briefly",
        "check": lambda x: len(x) > 50,
    },
]


def run_mlx(prompt: str, adapter: str, max_tokens: int = 400) -> str:
    """Run with MLX adapter."""
    result = subprocess.run([
        "python", "-m", "mlx_lm", "generate",
        "--model", "microsoft/Phi-3-mini-4k-instruct",
        "--adapter-path", adapter,
        "--max-tokens", str(max_tokens),
        "--prompt", prompt,
    ], capture_output=True, text=True, timeout=120)
    return result.stdout


def run_gguf(prompt: str, model: str, max_tokens: int = 400) -> str:
    """Run with GGUF model via llama-cli."""
    formatted = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
    result = subprocess.run([
        "llama-cli", "-m", model,
        "-p", formatted,
        "-n", str(max_tokens),
        "--temp", "0.7",
        "-ngl", "99",
    ], capture_output=True, text=True, timeout=120)
    return result.stdout


def run_ollama(prompt: str, model: str) -> str:
    """Run with Ollama."""
    result = subprocess.run([
        "ollama", "run", model, prompt
    ], capture_output=True, text=True, timeout=120)
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description="Test design-analyst model")
    parser.add_argument("--use-mlx", action="store_true", help="Use MLX adapter")
    parser.add_argument("--use-ollama", action="store_true", help="Use Ollama")
    parser.add_argument("--model", default="models/design-analyst-v4-q4.gguf")
    parser.add_argument("--adapter", default="adapters/design-analyst-v4")
    parser.add_argument("--ollama-model", default="design-analyst")
    parser.add_argument("--test", type=int, help="Run specific test (1-5)")
    parser.add_argument("--max-tokens", type=int, default=400)
    args = parser.parse_args()

    print("=" * 60)
    print("Design Analyst Test Suite")
    print("=" * 60)

    if args.use_mlx:
        print(f"Backend: MLX ({args.adapter})")
        run_fn = lambda p: run_mlx(p, args.adapter, args.max_tokens)
    elif args.use_ollama:
        print(f"Backend: Ollama ({args.ollama_model})")
        run_fn = lambda p: run_ollama(p, args.ollama_model)
    else:
        print(f"Backend: GGUF ({args.model})")
        run_fn = lambda p: run_gguf(p, args.model, args.max_tokens)

    tests = TEST_CASES
    if args.test:
        tests = [TEST_CASES[args.test - 1]]

    results = []
    for i, test in enumerate(tests, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {test['name']}")
        print(f"{'='*60}")
        print(f"Prompt: {test['prompt'][:60]}...")
        print("-" * 60)

        try:
            output = run_fn(test['prompt'])
            print(output[:500] if len(output) > 500 else output)
            passed = test['check'](output)
        except Exception as e:
            print(f"Error: {e}")
            passed = False

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"\nStatus: {status}")
        results.append((test['name'], passed))

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for name, passed in results:
        print(f"  {'✅' if passed else '❌'} {name}")

    passed_count = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed_count}/{len(results)}")

    return 0 if passed_count == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())