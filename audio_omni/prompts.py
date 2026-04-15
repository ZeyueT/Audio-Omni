"""
Prompt templates for Audio-Omni tasks.

Each template follows the Qwen chat format:
    <|im_start|>system\n{system}<|im_end|>
    <|im_start|>user\n{user}<|im_end|>
    <|im_start|>assistant
"""

# ---------------------------------------------------------------------------
# System prompts (one per task family)
# ---------------------------------------------------------------------------

SYSTEM_T2A = (
    "You are a professional sound designer. "
    "Your task is to synthesize a high-fidelity audio clip based on the following text description."
)

SYSTEM_T2M = (
    "You are a versatile music producer and composer. "
    "Your goal is to create a high-quality, original piece of music based on the user's request. "
    "Focus on melody, harmony, rhythm, and instrumentation to capture the desired mood."
)

SYSTEM_TTS = "Generate speech from the input text."

SYSTEM_V2A = (
    "You are a professional foley artist. "
    "Understand this video and create the sounds for this video."
)

SYSTEM_V2M = (
    "You are a versatile music producer. "
    "Create a high-quality background music track that fits the style, energy, and mood of this video."
)

SYSTEM_EDITING = (
    "You are a professional sound engineer. "
    "Your task is to precisely edit the provided audio based on the user's instructions."
)

SYSTEM_UNDERSTANDING = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)

# ---------------------------------------------------------------------------
# Editing user-prompt templates
# ---------------------------------------------------------------------------

EDITING_ADD      = "Add the sound of '{desc}' to the input audio."
EDITING_REMOVE   = "Remove the sound of '{desc}' from the input audio."
EDITING_EXTRACT  = "Extract the sound of '{desc}' from the input audio."
EDITING_STYLE    = "Change the sound of '{source}' to '{target}'."

# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------

def _wrap(system: str, user: str) -> str:
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def _wrap_video(system: str, user: str) -> str:
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}\n"
        "<|vision_bos|><|VIDEO|><|vision_eos|>"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def build_prompt(task: str, text: str = "", **kwargs) -> str:
    """Build a Qwen-format prompt for the given task.

    Args:
        task: One of "T2A", "T2M", "TTS", "V2A", "V2M",
              "Add", "Remove", "Extract", "Style Transfer".
        text: The user text (description / transcript / optional hint).
        **kwargs: Extra fields for editing templates:
            - desc: sound object for Add / Remove / Extract
            - source, target: for Style Transfer

    Returns:
        A fully formatted prompt string.
    """
    task = task.strip()

    if task == "T2A":
        return _wrap(SYSTEM_T2A, f"{text}.")
    elif task == "T2M":
        return _wrap(SYSTEM_T2M, f"{text}.")
    elif task == "TTS":
        return _wrap(SYSTEM_TTS, "")
    elif task == "V2A":
        return _wrap_video(SYSTEM_V2A, text)
    elif task == "V2M":
        return _wrap_video(SYSTEM_V2M, text)
    elif task in ("Add", "Remove", "Extract", "Style Transfer"):
        desc = kwargs.get("desc", text)
        if task == "Add":
            user_text = EDITING_ADD.format(desc=desc)
        elif task == "Remove":
            user_text = EDITING_REMOVE.format(desc=desc)
        elif task == "Extract":
            user_text = EDITING_EXTRACT.format(desc=desc)
        else:
            user_text = EDITING_STYLE.format(
                source=kwargs.get("source", ""),
                target=kwargs.get("target", ""),
            )
        return _wrap(SYSTEM_EDITING, user_text)
    else:
        raise ValueError(f"Unknown task: {task!r}")
