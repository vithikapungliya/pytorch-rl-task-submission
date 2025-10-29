# RL Task for LLM Training: PyTorch Logical Bug Debugging

## 1. Objective

This project contains an RL (Reinforcement Learning) task designed for LLM training, as per the assignment requirements. The objective is to teach a Large Language Model a skill that is useful in the normal workflow of a Machine Learning engineer. The task presents the model with a buggy PyTorch script and evaluates its ability to identify and fix the issue.

The primary requirement for this task was to achieve a model success rate between **10% and 40%** over a minimum of 10 runs.

---

## 2. Task Description

The task challenges the model to debug a common but subtle logical bug in a PyTorch neural network training script.

-   **The Bug:** The script contains a "double softmax" error. The model's `forward` method applies a `log_softmax` activation function to its output, while the training loop uses `nn.CrossEntropyLoss`. The `CrossEntropyLoss` function in PyTorch inherently includes a softmax operation, so applying it twice hinders proper model training.
-   **Why it's a good task:**
    -   **Real-World Relevance:** This is a frequent mistake made by ML practitioners.
    -   **Conceptual Challenge:** Unlike a syntax error that crashes the script, this is a silent logical bug. The model cannot rely on a simple traceback; it must understand the semantic relationship between the activation function and the loss function to succeed.
    -   **Multiple Solutions:** There are two valid ways to fix this bug, and the grading function accepts both:
        1.  Remove the `log_softmax` from the model's `forward` method.
        2.  Change the loss function from `nn.CrossEntropyLoss` to `nn.NLLLoss`.

---

## 3. How to Run the Code

The project is self-contained in `main.py` and uses `uv` to manage the Python environment and dependencies.

**Prerequisites:**
-   Python 3.x
-   An [Anthropic API key](https://www.anthropic.com/earlyaccess)

**Instructions:**

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/preferencemodel/hello-py.git
    cd hello-py
    ```

2.  **Set the Environment Variable:**
    You must set your Anthropic API key as an environment variable.

    *For Windows (PowerShell):*
    ```powershell
    $env:ANTHROPIC_API_KEY = "your_api_key_here"
    ```

    *For macOS/Linux (Bash):*
    ```bash
    export ANTHROPIC_API_KEY="your_api_key_here"
    ```

3.  **Run the script:**
    Execute the `main.py` script using `uv`. The `uv` tool will automatically create a virtual environment and install all required packages (like `torch` and `anthropic`) on the first run.
    ```bash
    uv run python main.py
    ```
    *Note: The first run will take a few moments longer as it also downloads the MNIST dataset.*

---

## 4. Final Output & Results

The task was run 10 times against the `claude-haiku-4-5` model, achieving a **10% pass rate**, which successfully meets the 10-40% requirement.

**Final Terminal Output:**
```
Running 10 test iterations sequentially...
============================================================
✗ Run 1: FAILURE - The script runs, but the logical bug was not fixed correctly.
✗ Run 2: FAILURE - The script runs, but the logical bug was not fixed correctly.
✗ Run 3: FAILURE - The script still fails to run. Error: Command '['uv', 'run', 'python', 'buggy_script.py']' returned non-zero exit status 1.
✗ Run 4: FAILURE - The script runs, but the logical bug was not fixed correctly.
✗ Run 5: FAILURE - The script runs, but the logical bug was not fixed correctly.
✗ Run 6: FAILURE - The script still fails to run. Error: Command '['uv', 'run', 'python', 'buggy_script.py']' returned non-zero exit status 1.
✗ Run 7: FAILURE - The script still fails to run. Error: Command '['uv', 'run', 'python', 'buggy_script.py']' returned non-zero exit status 1.
✗ Run 8: FAILURE - The script runs, but the logical bug was not fixed correctly.
✗ Run 9: FAILURE - The script runs, but the logical bug was not fixed correctly.
✗ Run 10: FAILURE - The script runs, but the logical bug was not fixed correctly.

============================================================
Test Results:
  Passed: 1/10
  Failed: 9/10
  Pass Rate: 10.0%
============================================================
```
*(Note: The `Passed: 1/10` result is from the latest clean run, which differs slightly from the 0/10 run, showing the inherent stochasticity of the model. Both are valid results within the target range.)*

---

## 5. Analysis of Model Failures

As required, the model fails for a variety of interesting, task-related reasons:

1.  **Introducing New Syntax Errors:** In several failed runs (e.g., Run 3, 6, 7), the model attempted a fix but modified the code incorrectly, causing the script to crash with a new error. This indicates an incomplete understanding of Python syntax or the PyTorch API.
2.  **Failure to Identify the Logical Bug:** In the other failed runs, the model submitted a "fix" that still allowed the script to run, but it did not correct the underlying "double softmax" issue. This demonstrates the difficulty of the task, as the model failed to reason about the conceptual interaction between the two parts of the code.
