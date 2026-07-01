# Build-in Computer Use Agent (CUA) for browser automation using Playwright
# Given a URL, and the task, the CUA will try to complete the task
# Supports customized system prompt, task prompt, and conclusion prompt

# Experimental supported now

import asyncio
import base64
import datetime as dt
import importlib
import json
import logging
import os
import random
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from openai import RateLimitError
from tqdm import tqdm

from ..core.const import ParseError
from ..utils import is_openai_rate_limit_error

logger = logging.getLogger(__name__)

last_successful_screenshot = None


def _load_async_azure_openai():
    try:
        module = importlib.import_module("openai")
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI SDK is required to run the Computer Use Agent."
        ) from exc
    if not hasattr(module, "AsyncAzureOpenAI"):
        raise RuntimeError(
            "AsyncAzureOpenAI client is unavailable. Install the official openai package "
            "with Azure extras to use the Computer Use Agent."
        )
    return module.AsyncAzureOpenAI


_PLAYWRIGHT_LOADER = None
_PLAYWRIGHT_TIMEOUT = None


def _ensure_playwright():
    global _PLAYWRIGHT_LOADER, _PLAYWRIGHT_TIMEOUT
    if _PLAYWRIGHT_LOADER is None:
        try:
            module = importlib.import_module("playwright.async_api")
            async_playwright = getattr(module, "async_playwright")
            timeout_error = getattr(module, "TimeoutError")
        except Exception as exc:
            raise RuntimeError(
                "Playwright is required for the Computer Use Agent. "
                "Install it with `pip install playwright` to enable browser automation."
            ) from exc
        _PLAYWRIGHT_LOADER = async_playwright
        _PLAYWRIGHT_TIMEOUT = timeout_error
    return _PLAYWRIGHT_LOADER, _PLAYWRIGHT_TIMEOUT


class ControlSignals(Enum):
    TERMINATE = "terminate"  # Terminate the session


# Key mapping for special keys in Playwright
KEY_MAPPING = {
    "/": "Slash",
    "\\": "Backslash",
    "alt": "Alt",
    "arrowdown": "ArrowDown",
    "arrowleft": "ArrowLeft",
    "arrowright": "ArrowRight",
    "arrowup": "ArrowUp",
    "backspace": "Backspace",
    "ctrl": "Control",
    "delete": "Delete",
    "enter": "Enter",
    "esc": "Escape",
    "shift": "Shift",
    "space": " ",
    "tab": "Tab",
    "win": "Meta",
    "cmd": "Meta",
    "super": "Meta",
    "option": "Alt",
}


@dataclass
class ComputerUseHandler:
    DISPLAY_WIDTH: int = 1280
    DISPLAY_HEIGHT: int = 800

    async def take_screenshot(self, page):
        """Take a screenshot and return base64 encoding with caching for failures."""
        global last_successful_screenshot

        try:
            screenshot_bytes = await page.screenshot(full_page=False)
            last_successful_screenshot = base64.b64encode(screenshot_bytes).decode(
                "utf-8"
            )
            return last_successful_screenshot
        except Exception as e:
            logger.warning("Screenshot failed: %s — using cached screenshot", e)
            if last_successful_screenshot:
                return last_successful_screenshot

    def validate_coordinates(self, x, y):
        """Ensure coordinates are within display bounds."""
        return max(0, min(x, self.DISPLAY_WIDTH)), max(0, min(y, self.DISPLAY_HEIGHT))

    def handle_control_signal(self, action) -> ControlSignals:
        """Check if the action is a special action."""
        # handle termination signal: Ctrl+W, Alt+F4, Cmd+W
        action_type = action.type
        if action_type == "keypress":
            keys = getattr(action, "keys", [])
            signal = "+".join([key.lower() for key in keys])
            if signal in ["cmd+w", "ctrl+w", "alt+f4"]:
                return ControlSignals.TERMINATE
        return None

    async def handle_action(self, page, action):
        # raise NotImplementedError("handle_action is not implemented in the base class, please implement it in the derived class.")

        """Handle different action types from the model."""
        action_type = action.type

        if action_type == "drag":
            logger.debug("Drag action is not supported, skipping.")
            return

        elif action_type == "click":
            button = getattr(action, "button", "left")
            # Validate coordinates
            x, y = self.validate_coordinates(action.x, action.y)

            # print(f"\tAction: click at ({x}, {y}) with button '{button}'")

            if button == "back":
                await page.go_back()
            elif button == "forward":
                await page.go_forward()
            elif button == "wheel":
                await page.mouse.wheel(x, y)
            else:
                button_type = {
                    "left": "left",
                    "right": "right",
                    "middle": "middle",
                }.get(button, "left")
                await page.mouse.click(x, y, button=button_type)
                try:
                    await page.wait_for_load_state("domcontentloaded", timeout=3000)
                except _PLAYWRIGHT_TIMEOUT:
                    pass

        elif action_type == "double_click":
            # Validate coordinates
            x, y = self.validate_coordinates(action.x, action.y)

            # print(f"\tAction: double click at ({x}, {y})")
            await page.mouse.dblclick(x, y)

        elif action_type == "scroll":
            scroll_x = getattr(action, "scroll_x", 0)
            scroll_y = getattr(action, "scroll_y", 0)
            # Validate coordinates
            x, y = self.validate_coordinates(action.x, action.y)

            # print(f"\tAction: scroll at ({x}, {y}) with offsets ({scroll_x}, {scroll_y})")
            await page.mouse.move(x, y)
            await page.evaluate(
                f"window.scrollBy({{left: {scroll_x}, top: {scroll_y}, behavior: 'smooth'}});"
            )

        elif action_type == "keypress":
            keys = getattr(action, "keys", [])
            # print(f"\tAction: keypress {keys}")
            mapped_keys = [KEY_MAPPING.get(key.lower(), key) for key in keys]

            if len(mapped_keys) > 1:
                # For key combinations (like Ctrl+C)
                for key in mapped_keys:
                    await page.keyboard.down(key)
                await asyncio.sleep(0.1)
                for key in reversed(mapped_keys):
                    await page.keyboard.up(key)
            else:
                for key in mapped_keys:
                    await page.keyboard.press(key)

        elif action_type == "type":
            text = getattr(action, "text", "")
            # print(f"\tAction: type text: {text}")
            await page.keyboard.type(text, delay=20)

        elif action_type == "wait":
            ms = getattr(action, "ms", 1000)
            # print(f"\tAction: wait {ms}ms")
            await asyncio.sleep(ms / 1000)

        elif action_type == "screenshot":
            pass
            # print("\tAction: screenshot")

        else:
            pass
            # print(f"\tUnrecognized action: {action_type}")


_DEFAULT_CUA_CONFIGS = {
    "display_height": 800,
    "display_width": 1280,
    "max_iterations": 10,
}

_CONTROL_INSTRUCTIONS = """
## Instructions

 - If you wish to terminate the session (e.g., the task accomplished, error occurred), you can press `Ctrl+W`, `Alt+F4`, or `Cmd+W` to close the browser tab.  
    - Be patient, when you encounter a problem, try to solve it first, only terminate the session if you cannot solve it.
    - This is the ONLY WAY to terminate the session by yourself.
 - You will be asked to provide a report at the end of the session to conclude the session, comment on your experience, provide feedback, or report issues.
 - NO ACTION WILL NOT BE RECOGNIZED AS A TERMINATION SIGNAL. If you provide no action, the system will process it as a wait action (for 1000ms) by default.
"""


class AgentException(Exception):
    """Custom exception for agent errors."""

    def __init__(self, message):
        super().__init__(message)
        self.message = message


@dataclass
class CUASession:
    url: str
    system: str
    user_input: str
    trace_dir: str
    conclude: str = None
    conclude_parser: callable = None  # Optional parser for the conclusion
    responses: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    ckpt_dir: str = None
    report: str = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.ckpt_dir is not None:
            if not os.path.exists(self.ckpt_dir):
                os.makedirs(self.ckpt_dir, exist_ok=True)
        self.id = uuid.uuid4().hex[:6]  # Unique session ID

    def get_report(self):
        if self.report is None:
            return None
        if isinstance(self.report, str):
            return self.report.strip()
        if isinstance(self.report, dict):
            return self.report["raw"].strip()

    @property
    def ckpt_file(self):
        return (
            os.path.join(self.ckpt_dir, "cua_session.json") if self.ckpt_dir else None
        )

    def log_response(self, call_args, response, previous_response_id) -> int:
        """Log the call arguments."""
        response_id = getattr(response, "id", "unknown")
        _data = {
            "args": call_args,
            "timestamp": dt.datetime.now().isoformat(),
            "response": response.model_dump_json(),
            "response_id": response_id,
            "previous_response_id": previous_response_id,
        }
        self.responses.append(_data)
        self.save()  # Save the session after logging the action

    def log_action(self, computer_call, response_id) -> int:
        """Log the action taken by the model."""
        _data = {
            "response_id": response_id,
            "action": computer_call.model_dump_json(),
            "timestamp": dt.datetime.now().isoformat(),
        }
        self.actions.append(_data)
        self.save()  # Save the session after logging the action

    def to_dict(self):
        data = asdict(self)
        data["conclude_parser"] = None  # Do not serialize the parser
        data["id"] = self.id  # Include the unique session ID
        return data

    def save(self, path=None):
        """Save the session to a file."""
        if path is None:
            path = self.ckpt_file  # .json file in the ckpt_dir
        if path is None:
            # print("No path provided to save the session. Please provide a valid path.")
            return
        with open(path, "w") as f:
            _dict = self.to_dict()
            _dict.pop("conclude_parser", None)  # Remove parser from dict
            json.dump(_dict, f, indent=4)

    @classmethod
    def from_dict(cls, data):
        """Create a session from a dictionary."""
        return cls(
            url=data.get("url"),
            system=data.get("system"),
            trace_dir=data.get("trace_dir", ""),
            user_input=data.get("user_input"),
            conclude=data.get("conclude"),
            responses=data.get("responses", []),
            actions=data.get("actions", []),
            ckpt_dir=data.get("ckpt_dir"),
            report=data.get("report", None),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def new(
        cls,
        url,
        user_input,
        trace_dir,
        system=None,
        conclude=None,
        ckpt_dir=None,
        conclude_parser=None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Create a new CUA session."""
        return cls(
            url=url,
            trace_dir=trace_dir,
            system=system,
            user_input=user_input,
            conclude=conclude,
            ckpt_dir=ckpt_dir,
            conclude_parser=conclude_parser,
            metadata=metadata or {},
        )

    @property
    def prompt(self):
        return f"""The task performed by the CUA is as follows:
{self.user_input}

The result of the task is as follows:

{self.report}
"""


class OpenAICUA:
    def __init__(self, cua_configs, client=None):
        self.cua_configs = cua_configs
        for key in _DEFAULT_CUA_CONFIGS:
            if key not in self.cua_configs:
                logger.warning(
                    "'%s' not in CUA configs, using default %s",
                    key,
                    _DEFAULT_CUA_CONFIGS[key],
                )
                self.cua_configs[key] = _DEFAULT_CUA_CONFIGS[key]
        self.handler = ComputerUseHandler(
            DISPLAY_HEIGHT=self.cua_configs["display_height"],
            DISPLAY_WIDTH=self.cua_configs["display_width"],
        )
        self.DISPLAY_WIDTH = self.handler.DISPLAY_WIDTH
        self.DISPLAY_HEIGHT = self.handler.DISPLAY_HEIGHT
        self.model = "computer-use-preview"
        self._client = client

    def _get_client(self):
        if self._client is not None:
            return self._client

        api_key = os.getenv("CUA_API_KEY")
        endpoint = os.getenv("AZURE_CUA_ENDPOINT")
        if not api_key or not endpoint:
            raise RuntimeError(
                "OpenAICUA requires either an explicit client instance or the "
                "CUA_API_KEY and AZURE_CUA_ENDPOINT environment variables."
            )
        api_version = os.getenv("CUA_API_VERSION", "2025-04-01-preview")
        AsyncAzureOpenAI = _load_async_azure_openai()
        self._client = AsyncAzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
        )
        return self._client

    async def create_response(
        self, sess, input, previous_response_id=None, max_recall=3, **kwargs
    ):
        """Create a response object for the model."""
        _call_args = {
            "model": self.model,
            "input": input,
            "tools": [
                {
                    "type": "computer_use_preview",
                    "display_width": self.DISPLAY_WIDTH,
                    "display_height": self.DISPLAY_HEIGHT,
                    "environment": "browser",
                }
            ],
            "truncation": "auto",
            "previous_response_id": previous_response_id,
        }
        for key, value in kwargs.items():
            _call_args[key] = value
        llm_recall = max(1, max_recall)
        response = None
        while llm_recall > 0:
            try:
                response = await self._get_client().responses.create(**_call_args)
                break
            except RateLimitError:
                wait_time = random.random() * 15 + 1
                logger.warning("Rate limit error, retrying in %.2fs", wait_time)
                await asyncio.sleep(wait_time)
            except Exception as e:
                if is_openai_rate_limit_error(e):  # for safer
                    wait_time = random.random() * 15 + 1
                    logger.warning("Rate limit error, retrying in %.2fs", wait_time)
                    await asyncio.sleep(5)
                else:
                    llm_recall -= 1  # Decrement for other errors
                    logger.error("Unexpected API error: %s", e, exc_info=True)
                    if llm_recall > 0:
                        logger.info(
                            "Retrying in 2 seconds. Retries left: %d", llm_recall
                        )
                        await asyncio.sleep(2)
                    else:
                        logger.error("Max retries reached, aborting.")
        if response is None:
            raise AgentException(
                "Unable to obtain a response from the Computer Use API after retries."
            )
        sess.log_response(_call_args, response, previous_response_id)
        return response

    async def process_model_response(
        self, sess, response, page, safety_checks=False, max_recall=3
    ):
        """Process the model's response and execute actions."""
        max_iterations = self.cua_configs["max_iterations"]
        report = None
        _termination_call_id = None
        _termination_reason = None
        conclude = sess.conclude

        for iteration in tqdm(
            range(max_iterations), desc=f"CUA session {sess.id}", unit="iteration"
        ):
            try:
                if not hasattr(response, "output") or not response.output:
                    # print("No output from model.")
                    raise AgentException(
                        "There is no output from the model. Will wait by default. Please provide a valid response."
                    )

                # Safely access response id
                response_id = getattr(response, "id", "unknown")
                # print(f"\nIteration {iteration + 1} - Response ID: {response_id}\n")

                # Print text responses and reasoning
                for item in response.output:
                    # Handle text output
                    # if hasattr(item, 'type') and item.type == "text":
                    #     print(f"\nModel message: {item.text}\n")

                    # Handle reasoning output
                    if hasattr(item, "type") and item.type == "reasoning":
                        # Extract meaningful content from the reasoning
                        meaningful_content = []

                        if hasattr(item, "summary") and item.summary:
                            for summary in item.summary:
                                # Handle different potential formats of summary content
                                if isinstance(summary, str) and summary.strip():
                                    meaningful_content.append(summary)
                                elif hasattr(summary, "text") and summary.text.strip():
                                    meaningful_content.append(summary.text)

                        # Only print reasoning section if there's actual content
                        # if meaningful_content:
                        #     print("=== Model Reasoning ===")
                        #     for idx, content in enumerate(meaningful_content, 1):
                        #         print(f"{content}")
                        #     print("=====================\n")

                # Extract computer calls
                computer_calls = [
                    item
                    for item in response.output
                    if hasattr(item, "type") and item.type == "computer_call"
                ]

                if len(computer_calls) == 0:
                    # print("No computer calls found in the response.")
                    if iteration == max_iterations - 1:  # last iteration
                        # print("Reached maximum number of iterations. Stopping.")
                        _termination_reason = "Max iterations reached, the session is terminated by the system."
                        break
                    raise AgentException(
                        "There are no computer calls in the response. Will wait by default. If you wish to terminate the session, please press Ctrl+W, Alt+F4, or Cmd+W to close the tab."
                    )

                computer_call = computer_calls[0]
                missing_attributes = [
                    attr
                    for attr in ["call_id", "action"]
                    if not hasattr(computer_call, attr)
                ]
                if len(missing_attributes) > 0:
                    # print(f"Computer call is missing required attributes: {', '.join(missing_attributes)}.")
                    raise AgentException(
                        f"Computer call is missing required attributes: {', '.join(missing_attributes)}. Will wait by default. Please provide a valid response."
                    )

                call_id = computer_call.call_id
                action = computer_call.action
                sess.log_action(computer_call, response_id)
                control_signal = self.handler.handle_control_signal(action)
                if control_signal == ControlSignals.TERMINATE:
                    # print("Control signal received: Terminating session.")
                    screenshot_base64 = await self.handler.take_screenshot(page)
                    _termination_call_id = call_id
                    _termination_reason = "Termination signal received from user."
                    break

                if iteration == max_iterations - 1:  # last iteration
                    # print("Reached maximum number of iterations. Stopping.")
                    screenshot_base64 = await self.handler.take_screenshot(page)
                    _termination_call_id = call_id
                    _termination_reason = "Max iterations reached, the session is terminated by the system."
                    break

                # Handle safety checks
                acknowledged_checks = []
                # if safety_checks:
                if (
                    hasattr(computer_call, "pending_safety_checks")
                    and computer_call.pending_safety_checks
                ):
                    pending_checks = computer_call.pending_safety_checks
                    # print("\nSafety checks required:")
                    # for check in pending_checks:
                    #     print(f"- {check.code}: {check.message}")

                    # if input("\nDo you want to proceed? (y/n): ").lower() != 'y':
                    #     print("Operation cancelled by user.")
                    #     break

                    acknowledged_checks = pending_checks

                # Execute the action
                try:
                    await page.bring_to_front()
                    await self.handler.handle_action(page, action)

                    # Check if a new page was created after the action
                    if action.type in ["click"]:
                        await asyncio.sleep(1.5)
                        all_pages = page.context.pages
                        if len(all_pages) > 1:
                            newest_page = all_pages[-1]
                            if newest_page != page and newest_page.url not in [
                                "about:blank",
                                "",
                            ]:
                                logger.debug(
                                    "Switching to new tab: %s", newest_page.url
                                )
                                page = newest_page
                    elif action.type != "wait":
                        await asyncio.sleep(0.5)

                except Exception as e:
                    logger.error(
                        "Error handling action '%s': %s", action.type, e, exc_info=True
                    )

                # Take a screenshot after the action
                screenshot_base64 = await self.handler.take_screenshot(page)

                # print("\tNew screenshot taken")

                # Prepare input for the next request
                input_content = [
                    {
                        "type": "computer_call_output",
                        "call_id": call_id,
                        "output": {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{screenshot_base64}",
                        },
                    }
                ]

                # Add acknowledged safety checks if any
                if acknowledged_checks:
                    acknowledged_checks_dicts = []
                    for check in acknowledged_checks:
                        acknowledged_checks_dicts.append(
                            {
                                "id": check.id,
                                "code": check.code,
                                "message": check.message,
                            }
                        )
                    input_content[0]["acknowledged_safety_checks"] = (
                        acknowledged_checks_dicts
                    )

                # Add current URL for context
                try:
                    current_url = page.url
                    if current_url and current_url != "about:blank":
                        input_content[0]["current_url"] = current_url
                        logger.debug("Current URL: %s", current_url)
                except Exception as e:
                    logger.warning("Error getting page URL: %s", e)

                # Send the screenshot back for the next step
                try:
                    response = await self.create_response(
                        sess=sess,
                        previous_response_id=response_id,
                        input=input_content,
                    )

                    # print("\tModel processing screenshot")
                except Exception as e:
                    logger.error("API call failed: %s", e, exc_info=True)
                    break

            except AgentException as e:
                logger.warning("Agent exception during CUA iteration: %s", e.message)
                response = await self.create_response(
                    sess=sess,
                    input=e.message,
                    previous_response_id=getattr(response, "id", "unknown"),
                )
                await asyncio.sleep(1)  # Wait before retrying
                continue

        if conclude:
            logger.info("CUA session %s: concluding with final instructions", sess.id)
            response_id = getattr(response, "id", "unknown")
            assert response_id != "unknown", (
                "Response ID is unknown, cannot conclude session."
            )
            inputs = []
            if _termination_call_id:
                inputs.append(
                    {
                        "type": "computer_call_output",
                        "call_id": call_id,
                        "output": {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{screenshot_base64}",
                        },
                    }
                )
                inputs.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "The last screenshot before termination is attached. The browser tab was closed.",
                            }
                        ],
                    }
                )
            if _termination_reason:
                inputs.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": f"{_termination_reason} The session terminated successfully. ",
                            }
                        ],
                    }
                )
            inputs.append(
                {"role": "user", "content": [{"type": "input_text", "text": conclude}]}
            )
            for i in range(max_recall):
                response = await self.create_response(
                    sess=sess,
                    input=inputs,
                    # reasoning={"generate_summary": "concise"},
                    previous_response_id=response_id,
                )
                response_id = getattr(response, "id", "unknown")
                report = response.output_text
                try:
                    if sess.conclude_parser:
                        report = sess.conclude_parser(report)
                    break
                except ParseError as e:
                    call_id = None
                    computer_calls = [
                        item
                        for item in response.output
                        if hasattr(item, "type") and item.type == "computer_call"
                    ]
                    if len(computer_calls) != 0:
                        computer_call = computer_calls[0]
                        missing_attributes = [
                            attr
                            for attr in ["call_id", "action"]
                            if not hasattr(computer_call, attr)
                        ]
                        if len(missing_attributes) == 0:
                            call_id = computer_call.call_id

                    inputs = []
                    if call_id:
                        inputs.append(
                            {
                                "type": "computer_call_output",
                                "call_id": call_id,
                                "output": {
                                    "type": "input_image",
                                    "image_url": f"data:image/png;base64,{screenshot_base64}",
                                },
                            }
                        )
                        inputs.append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_text",
                                        "text": "The session was terminated already. The action was ignored. Please do not make any more actions.",
                                    }
                                ],
                            }
                        )
                    inputs.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": f"There is any error from your response: {e}. Please follow the instructions closely.",
                                }
                            ],
                        }
                    )
                    logger.warning("Conclusion parse error: %s", e)
                    continue
            logger.info("CUA session %s concluded successfully", sess.id)

        return report

    async def call(
        self,
        url,
        user_input,
        system,
        conclude=None,
        conclude_parser=None,
        safety_checks=False,
        wait_until="domcontentloaded",
        headless=False,
        ckpt_dir=None,
        metadata=None,
        trace_dir=None,
    ) -> CUASession:

        async_playwright, _ = _ensure_playwright()
        DISPLAY_WIDTH = self.handler.DISPLAY_WIDTH
        DISPLAY_HEIGHT = self.handler.DISPLAY_HEIGHT
        report = None

        sess = CUASession.new(
            url=url,
            user_input=user_input,
            system=system,
            conclude=conclude,
            ckpt_dir=ckpt_dir,
            conclude_parser=conclude_parser,
            metadata=metadata,
            trace_dir=trace_dir,
        )

        error = None
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                headless=headless,
                args=[
                    f"--window-size={DISPLAY_WIDTH},{DISPLAY_HEIGHT}",
                    "--disable-extensions",
                ],
            )

            context = await browser.new_context(
                viewport={"width": DISPLAY_WIDTH, "height": DISPLAY_HEIGHT},
                accept_downloads=True,
            )

            page = await context.new_page()

            # Navigate to starting page
            await page.goto(url, wait_until=wait_until)
            logger.info("Browser initialized to %s", url)

            if system is None:
                system = """You are an AI agent with the ability to control a browser. 
You can control the keyboard and mouse. You take a screenshot after each action to check if your action was successful. 
Once you have completed the requested task you should stop running and pass back control to your human operator."""

            # Append control instructions
            system += _CONTROL_INSTRUCTIONS

            # Main interaction loop
            try:
                # Take initial screenshot
                screenshot_base64 = await self.handler.take_screenshot(page)

                # Initial request to the model
                response = await self.create_response(
                    sess=sess,
                    instructions=system,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": user_input},
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/png;base64,{screenshot_base64}",
                                },
                            ],
                        }
                    ],
                    reasoning={"generate_summary": "concise"},
                )
                logger.debug("Sent initial screenshot and instructions to model")

                # Process model actions
                report = await self.process_model_response(
                    sess, response, page, safety_checks=safety_checks
                )

            except Exception as e:
                import traceback as _tb

                logger.error("CUA session error: %s", e, exc_info=True)
                error = f"Error occured in CUA: {e}\n{'-' * 100}\n{_tb.format_exc()}"
            finally:
                # Close browser
                await context.close()
                await browser.close()
                logger.debug("Browser closed")
        if report is None:
            report = {
                "raw": error
                if error
                else "No report generated. The session was terminated without a report.",
                "json": {"decision": "fail", "explanation": "N/A", "comments": "N/A"},
            }
            report["analysis"] = report["raw"]

        sess.report = report
        sess.save()  # Save the session state to file
        return sess
