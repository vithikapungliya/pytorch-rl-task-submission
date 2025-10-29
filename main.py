import asyncio
import json
import os
import subprocess
from collections.abc import Callable
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, TypedDict

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolUnionParam

MAX_TOKENS = 2000


class PythonExpressionToolResult(TypedDict):
    result: Any
    error: str | None


class SubmitAnswerToolResult(TypedDict):
    answer: Any
    submitted: bool


def read_file(filepath: str) -> dict[str, Any]:
    """
    Tool to read content from a file.
    """
    try:
        with open(filepath, "r") as f:
            content = f.read()
        return {"content": content}
    except Exception as e:
        return {"error": str(e)}


def write_file(filepath: str, content: str) -> dict[str, Any]:
    """
    Tool to write content to a file.
    """
    try:
        with open(filepath, "w") as f:
            f.write(content)
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def run_agent_loop(
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    max_steps: int = 20,
    model: str = "claude-haiku-4-5",
    verbose: bool = True,
) -> Any | None:
    """
    Runs an agent loop with the given prompt and tools.

    Args:
        prompt: The initial prompt for the agent
        tools: List of tool definitions for Anthropic API
        tool_handlers: Dictionary mapping tool names to their handler functions
        max_steps: Maximum number of steps before stopping (default 5)
        model: The Anthropic model to use
        verbose: Whether to print detailed output (default True)

    Returns:
        The submitted answer if submit_answer was called, otherwise None
    """
    client = AsyncAnthropic()
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]

    for step in range(max_steps):
        if verbose:
            print(f"\n=== Step {step + 1}/{max_steps} ===")

        response = await client.messages.create(
            model=model, max_tokens=MAX_TOKENS, tools=tools, messages=messages
        )

        assert response.stop_reason in ["max_tokens", "tool_use", "end_turn"], (
            f"unsupported stop_reason {response.stop_reason}"
        )
        if response.stop_reason == "max_tokens":
            print(
                f"Model reached max_tokens limit {MAX_TOKENS}. Increase "
                "MAX_TOKENS, simplify your task, or update the code to provide "
                "a message back to the model when it exceeds MAX_TOKENS."
            )

        # Track if we need to continue
        has_tool_use = False
        tool_results = []
        submitted_answer = None

        # Process the response
        for content in response.content:
            if content.type == "text":
                if verbose:
                    print(f"Assistant: {content.text}")
            elif content.type == "tool_use":
                has_tool_use = True
                tool_name = content.name

                if tool_name in tool_handlers:
                    if verbose:
                        print(f"Using tool: {tool_name}")

                    try:
                        # Extract arguments based on tool
                        handler = tool_handlers[tool_name]
                        tool_input = content.input

                        # Call the appropriate tool handler
                        if tool_name == "write_file":
                            assert (
                                isinstance(tool_input, dict)
                                and "filepath" in tool_input
                                and "content" in tool_input
                            )
                            if verbose:
                                print("\nInput:")
                                print(f"Filepath: {tool_input['filepath']}")
                                print("Content:")
                                print("```")
                                for line in tool_input["content"].split("\n"):
                                    print(f"{line}")
                                print("```")
                            result = handler(tool_input["filepath"], tool_input["content"])
                            if verbose:
                                print("\nOutput:")
                                print("```")
                                print(result)
                                print("```")
                        elif tool_name == "read_file":
                            assert (
                                isinstance(tool_input, dict) and "filepath" in tool_input
                            )
                            if verbose:
                                print("\nInput:")
                                print(f"Filepath: {tool_input['filepath']}")
                            result = handler(tool_input["filepath"])
                            if verbose:
                                print("\nOutput:")
                                print("```")
                                # Preview first 1000 chars
                                content_preview = result.get("content", "")[:1000]
                                print(content_preview)
                                if len(result.get("content", "")) > 1000:
                                    print("... (content truncated)")
                                print("```")
                        else:
                            # Generic handler call
                            result = (
                                handler(**tool_input)
                                if isinstance(tool_input, dict)
                                else handler(tool_input)
                            )

                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": json.dumps(result),
                            }
                        )
                    except Exception as e:
                        if verbose:
                            print(f"Error handling tool {tool_name}: {e}")
                        # Optionally, you could send an error message back to the model
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": json.dumps({"error": str(e)}),
                                "is_error": True,
                            }
                        )

        # If we have tool uses, add them to the conversation
        if has_tool_use:
            messages.append({"role": "assistant", "content": response.content})

            messages.append({"role": "user", "content": tool_results})

            # If an answer was submitted, return it
            if submitted_answer is not None:
                if verbose:
                    print(f"\nAgent submitted answer: {submitted_answer}")
                return submitted_answer
        else:
            # No tool use, conversation might be complete
            if verbose:
                print("\nNo tool use in response, ending loop.")
            break

    if verbose:
        print(f"\nReached maximum steps ({max_steps}) without submitting answer.")
    return None


async def run_single_test(
    run_id: int,
    num_runs: int,
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    grader: Callable[[], tuple[bool, str]],
    verbose: bool = False,
) -> tuple[int, bool, Any]:
    if verbose:
        print(f"\n\n{'=' * 20} RUN {run_id}/{num_runs} {'=' * 20}")

    # Clean up from previous runs
    if os.path.exists("buggy_script.py"):
        os.remove("buggy_script.py")
    if os.path.exists("model.pth"):
        os.remove("model.pth")

    # Create the buggy script for the agent to fix
    with open("buggy_script.py", "w") as f:
        f.write(BUGGY_SCRIPT_CONTENT)

    await run_agent_loop(
        prompt=prompt,
        tools=tools,
        tool_handlers=tool_handlers,
        max_steps=5,
        verbose=verbose,
    )

    success, message = grader()

    if success:
        print(f"✓ Run {run_id}: SUCCESS - {message}")
    else:
        print(f"✗ Run {run_id}: FAILURE - {message}")

    return run_id, success, message


BUGGY_SCRIPT_CONTENT = """
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Define the CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128) # Fixed the shape bug
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        # The bug is here: applying log_softmax when CrossEntropyLoss is also used.
        output = torch.log_softmax(x, dim=1)
        return output

def main():
    # Training settings
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download and load the training data
    dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Simple training loop for one batch
    model.train()
    data, target = next(iter(train_loader))
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    print("Training step completed successfully.")
    
    # Save the model
    torch.save(model.state_dict(), "model.pth")
    print("Model saved to model.pth")

if __name__ == '__main__':
    main()
"""

ERROR_TRACEBACK = """
Traceback (most recent call last):
  File "buggy_script.py", line 59, in <module>
    main()
  File "buggy_script.py", line 47, in main
    output = model(data)
  File "/path/to/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/path/to/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "buggy_script.py", line 22, in forward
    x = self.fc1(x)
  File "/path/to/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/path/to/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/path/to/site-packages/torch/nn/modules/linear.py", line 134, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x9216 and 10x128)
"""

def grade_script() -> tuple[bool, str]:
    """
    Grader function for the PyTorch debugging task.
    It checks if the model has fixed the logical bug in the script.
    """
    try:
        with open("buggy_script.py", "r") as f:
            content = f.read()
    except FileNotFoundError:
        return False, "The buggy_script.py file was not found."

    # Two valid solutions:
    # 1. Remove log_softmax from the model and keep CrossEntropyLoss
    # 2. Keep log_softmax but change loss to NLLLoss

    solution_1_valid = "log_softmax" not in content and "CrossEntropyLoss" in content
    solution_2_valid = "log_softmax" in content and "NLLLoss" in content

    if solution_1_valid:
        return True, "Solution 1: Removed log_softmax and kept CrossEntropyLoss."
    elif solution_2_valid:
        return True, "Solution 2: Kept log_softmax and changed loss to NLLLoss."
    else:
        # Before failing, let's quickly check if the script runs without error
        try:
            subprocess.run(
                ["uv", "run", "python", "buggy_script.py"],
                capture_output=True, text=True, check=True, timeout=60
            )
            return False, "The script runs, but the logical bug was not fixed correctly."
        except Exception as e:
            return False, f"The script still fails to run. Error: {e}"


async def main(concurrent: bool = True):
    tools: list[ToolUnionParam] = [
        {
            "name": "read_file",
            "description": "Reads the entire content of a file.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "The path to the file to read.",
                    }
                },
                "required": ["filepath"],
            },
        },
        {
            "name": "write_file",
            "description": "Writes content to a file, overwriting it.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "The path to the file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file.",
                    },
                },
                "required": ["filepath", "content"],
            },
        },
    ]

    tool_handlers = {
        "read_file": read_file,
        "write_file": write_file,
    }

    # Run the test 10 times and track success rate
    num_runs = 10
    prompt = f"""
I have a PyTorch script in `buggy_script.py`. The script runs without crashing, but it contains a logical bug that prevents the model from training effectively. The loss does not behave as expected.

The problem is a subtle interaction between the model's final activation function and the loss function being used.

Your task is to identify and fix this logical bug. There are two common ways to fix this particular issue.

Please read the `buggy_script.py` file, identify the bug, and use the `write_file` tool to apply a fix.
"""

    execution_mode = "concurrently" if concurrent else "sequentially"
    print(f"Running {num_runs} test iterations {execution_mode}...")
    print("=" * 60)

    # Create all test coroutines
    tasks = [
        run_single_test(
            run_id=i + 1,
            num_runs=num_runs,
            prompt=prompt,
            tools=tools,
            tool_handlers=tool_handlers,
            grader=grade_script,
            verbose=False,
        )
        for i in range(num_runs)
    ]

    # Run concurrently or sequentially based on the flag
    if concurrent:
        # Process results as they complete
        results = []
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
    else:
        # Run sequentially by awaiting each task in order
        results = []
        for task in tasks:
            result = await task
            results.append(result)

    # Count successes
    successes = sum(1 for _, success, _ in results if success)

    # Calculate and display pass rate
    pass_rate = (successes / num_runs) * 100
    print(f"\n{'=' * 60}")
    print("Test Results:")
    print(f"  Passed: {successes}/{num_runs}")
    print(f"  Failed: {num_runs - successes}/{num_runs}")
    print(f"  Pass Rate: {pass_rate:.1f}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    # Set to True for concurrent execution, False for sequential execution
    asyncio.run(main(concurrent=False))
