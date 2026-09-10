"""
Verification prompts for each condition (A, B, C, D).

Condition A: Screenshot only (no text reference)
Condition B: Screenshot + text reference
Condition C: Screenshot + reference screenshot (no text)
Condition D: Screenshot + reference screenshot + text reference
"""

PROMPTS = {
    "A": {
        "system": (
            "You are an expert evaluator for web automation agents. Your job is to "
            "determine whether a web task was completed successfully by examining "
            "the final state of the web page.\n\n"
            "You must respond in EXACTLY this JSON format:\n"
            "{\n"
            '  "verdict": "SUCCESS" or "FAILURE",\n'
            '  "confidence": <integer from 1 to 10>,\n'
            '  "reasoning": "<2-3 sentences explaining your judgment>"\n'
            "}\n\n"
            "Be precise. A task is only successful if ALL aspects of the task "
            "description are satisfied in the final page state. Partial completions "
            "count as FAILURE."
        ),
        "user": (
            "## Task\n{task_text}\n\n"
            "## Instructions\n"
            "Examine the screenshot of the web page's final state. "
            "Determine whether the task described above was completed successfully.\n\n"
            "Respond in the required JSON format."
        ),
    },
    "B": {
        "system": (
            "You are an expert evaluator for web automation agents. Your job is to "
            "determine whether a web task was completed successfully by examining "
            "the final state of the web page.\n\n"
            "You must respond in EXACTLY this JSON format:\n"
            "{\n"
            '  "verdict": "SUCCESS" or "FAILURE",\n'
            '  "confidence": <integer from 1 to 10>,\n'
            '  "reasoning": "<2-3 sentences explaining your judgment>"\n'
            "}\n\n"
            "Be precise. A task is only successful if ALL aspects of the task "
            "description are satisfied in the final page state. Partial completions "
            "count as FAILURE."
        ),
        "user": (
            "## Task\n{task_text}\n\n"
            "## Expected Outcome (text description)\n{text_reference}\n\n"
            "## Instructions\n"
            "Examine the screenshot of the web page's final state. "
            "Compare it against the expected outcome description above. "
            "Determine whether the task was completed successfully.\n\n"
            "Respond in the required JSON format."
        ),
    },
    "C": {
        "system": (
            "You are an expert evaluator for web automation agents. Your job is to "
            "determine whether a web task was completed successfully by examining "
            "the final state of the web page.\n\n"
            "You must respond in EXACTLY this JSON format:\n"
            "{\n"
            '  "verdict": "SUCCESS" or "FAILURE",\n'
            '  "confidence": <integer from 1 to 10>,\n'
            '  "reasoning": "<2-3 sentences explaining your judgment>"\n'
            "}\n\n"
            "Be precise. A task is only successful if ALL aspects of the task "
            "description are satisfied in the final page state. Partial completions "
            "count as FAILURE."
        ),
        "user": (
            "## Task\n{task_text}\n\n"
            "## Instructions\n"
            "Two screenshots are provided:\n"
            "1. The first screenshot shows the reference state: the final page state "
            "from a human demonstration that completed this task successfully.\n"
            "2. The second screenshot shows the final state after the agent's actions.\n\n"
            "Compare the two screenshots. Determine whether the task was completed "
            "successfully based on how the agent's final state compares with the reference.\n\n"
            "Respond in the required JSON format."
        ),
    },
    "D": {
        "system": (
            "You are an expert evaluator for web automation agents. Your job is to "
            "determine whether a web task was completed successfully by examining "
            "the final state of the web page.\n\n"
            "You must respond in EXACTLY this JSON format:\n"
            "{\n"
            '  "verdict": "SUCCESS" or "FAILURE",\n'
            '  "confidence": <integer from 1 to 10>,\n'
            '  "reasoning": "<2-3 sentences explaining your judgment>"\n'
            "}\n\n"
            "Be precise. A task is only successful if ALL aspects of the task "
            "description are satisfied in the final page state. Partial completions "
            "count as FAILURE."
        ),
        "user": (
            "## Task\n{task_text}\n\n"
            "## Expected Outcome (text description)\n{text_reference}\n\n"
            "## Instructions\n"
            "Two screenshots are provided:\n"
            "1. The first screenshot shows the reference state: the final page state "
            "from a human demonstration that completed this task successfully.\n"
            "2. The second screenshot shows the final state after the agent's actions.\n\n"
            "Compare the two screenshots against the expected outcome description above. "
            "Determine whether the task was completed successfully.\n\n"
            "Respond in the required JSON format."
        ),
    },
}
