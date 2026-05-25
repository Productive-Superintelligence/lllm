import functools as ft
import hashlib
import json
import os
import re
import shutil
from itertools import islice
from pathlib import Path
from typing import Dict, Optional

import requests
from filelock import FileLock

from ..core.const import ParseError

pjoin = os.path.join
psplit = os.path.split
pexists = os.path.exists
mkdirs = ft.partial(os.makedirs, exist_ok=True)
rmtree = ft.partial(shutil.rmtree, ignore_errors=True)

TMP_DIR = os.getenv("TMP_DIR") or pjoin(os.path.expanduser("~"), ".lllm")

CACHE_DIR = pjoin(TMP_DIR, ".cache")
mkdirs(CACHE_DIR)


def load_json(file, default={}):
    if not pexists(file):
        if default is None:
            raise FileNotFoundError(f"File {file} not found")
        return default
    with open(file, encoding="utf-8") as f:
        return json.load(f)


def save_json(file, data, indent=4):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def html_collapse(summary: str, content: str):
    return f"""
    <details>
    <summary>{summary}</summary>
    {content}
    </details>
    """


def find_level1_blocks_sorted(text):  # find all ```xxx``` blocks
    # Regular expressions for opening and closing patterns
    opening_pattern = r"```[^\s]+"  # Matches any pattern like ```xxx followed by non-whitespace characters
    closing_pattern = r"```(?=\s|$)"  # Matches standalone closing patterns, followed by space, newline, or end of string

    # Finding all opening and closing positions
    open_positions = [
        (m.start(), m.group()) for m in re.finditer(opening_pattern, text)
    ]
    close_positions = [m.start() for m in re.finditer(closing_pattern, text)]

    matches = []
    open_stack = []
    nesting_level = 0

    i, j = 0, 0
    last_match_end = -1

    while i < len(open_positions) or j < len(close_positions):
        if i < len(open_positions) and (
            j >= len(close_positions) or open_positions[i][0] < close_positions[j]
        ):
            # Handle an opening pattern
            open_stack.append(open_positions[i])
            nesting_level += 1
            i += 1
        else:
            # Handle a closing pattern
            if open_stack:
                start_pos, start_tag = open_stack.pop()
                nesting_level -= 1
                # If we're back to level 0, it's a level 1 match
                if nesting_level == 0:
                    match_start = start_pos
                    match_end = close_positions[j] + len("```")
                    if match_start > last_match_end:
                        matches.append((match_start, match_end))
                        last_match_end = match_end
            j += 1

    # Extract the substrings corresponding to the level 1 matches
    matches.sort(key=lambda x: x[0])
    result = [text[start:end] for start, end in matches]
    return result


def find_md_blocks(text: str, tag: str):  # find all ```block_tag``` blocks
    blocks = find_level1_blocks_sorted(text)
    matches = [
        block[len(f"```{tag}") : -3].strip()
        for block in blocks
        if block.startswith(f"```{tag}")
    ]
    return matches


def find_xml_blocks(text: str, tag: str):  # find all <tag> </tag> blocks
    opening_pattern = rf"<{tag}>(.*?)</{tag}>"
    matches = re.findall(opening_pattern, text, re.DOTALL)
    return matches


def find_all_xml_tags_sorted(text: str):
    """Finds all tag blocks and returns them sorted by position."""
    pattern = r"<([a-zA-Z0-9_]+)>(.*?)</\1>"
    matches = []
    for match in re.finditer(pattern, text, re.DOTALL):
        tag_name = match.group(1)
        content = match.group(2).strip()
        start_pos = match.start()
        matches.append({"tag": tag_name, "pos": start_pos, "content": content})

    matches.sort(key=lambda x: x["pos"])
    return matches


def directory_tree(
    dir_path: Path,
    level: int = -1,
    limit_to_directories: bool = False,
    length_limit: int = 1000,
):
    """Given a directory Path object print a visual tree structure"""
    space = "    "
    branch = "│   "
    tee = "├── "
    last = "└── "
    dir_path = Path(dir_path)  # accept string coerceable to Path
    files = 0
    directories = 0
    tree_dir = []

    def inner(dir_path: Path, prefix: str = "", level=-1):
        nonlocal files, directories
        if not level:
            return  # 0, stop iterating
        if limit_to_directories:
            contents = [d for d in dir_path.iterdir() if d.is_dir()]
        else:
            contents = list(dir_path.iterdir())
        pointers = [tee] * (len(contents) - 1) + [last]
        for pointer, path in zip(pointers, contents):
            if path.is_dir():
                yield prefix + pointer + path.name
                directories += 1
                extension = branch if pointer == tee else space
                yield from inner(path, prefix=prefix + extension, level=level - 1)
            elif not limit_to_directories:
                yield prefix + pointer + path.name
                files += 1

    tree_dir.append(dir_path.name)
    iterator = inner(dir_path, level=level)
    for line in islice(iterator, length_limit):
        tree_dir.append(line)
    if next(iterator, None):
        tree_dir.append(f"... length_limit, {length_limit}, reached, counted:")
    tree_dir.append(
        f"\n{directories} directories" + (f", {files} files" if files else "")
    )
    return "\n".join(tree_dir)


def create_cache_key(func_key: str, params: dict):
    key_seed = f"{func_key}-{params}"
    return hashlib.sha256(key_seed.encode()).hexdigest()[:32]


def save_cache_by_key(cache_name: str, cache_key: str, data: dict):
    _cache_dir = pjoin(CACHE_DIR, cache_name)
    mkdirs(_cache_dir)
    cache_file = pjoin(_cache_dir, f"{cache_key}.json")
    save_json(cache_file, data)


def load_cache_by_key(cache_name: str, cache_key: str):
    _cache_dir = pjoin(CACHE_DIR, cache_name)
    mkdirs(_cache_dir)
    cache_file = pjoin(_cache_dir, f"{cache_key}.json")
    if pexists(cache_file):
        try:
            return load_json(cache_file)
        except Exception:
            return None
    return None


def cache_response(cache_name: str, func_key: str, params: dict, response: dict):
    cache_key = create_cache_key(func_key, params)
    save_cache_by_key(cache_name, cache_key, response)


def load_api_cache(cache_name: str, func_key: str, params: dict):
    cache_key = create_cache_key(func_key, params)
    return load_cache_by_key(cache_name, cache_key)


# assume it return a dict, for api calls most of the time
def cache_call(cache_name: str):
    def decorator(func):
        @ft.wraps(func)
        def wrapper(
            func_key: str,
            params: dict,
            headers: Optional[dict] = None,
            use_cache: bool = True,
            json_response: bool = True,
        ):
            cached_response = load_api_cache(cache_name, func_key, params)
            if cached_response is not None and use_cache:
                return cached_response
            response = func(func_key, params, headers, use_cache, json_response)
            # always save the response, but read from cache if cache is True
            cache_response(cache_name, func_key, params, response)
            return response

        return wrapper

    return decorator


def raise_error(response: dict):
    error_keys = ["error", "Error Message", "Error"]
    if any(key in response for key in error_keys):
        raise ValueError(response)


@cache_call("API_CALL")
def call_api(
    url: str,
    params: dict,
    headers: Optional[dict] = None,
    use_cache: bool = True,
    json_response: bool = True,
):
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    if response.status_code == 200:
        if json_response:
            response_json = response.json()
            raise_error(response_json)
            return response_json
        else:
            return response
    else:
        raise ValueError(response.text)


@cache_call("API_CALL_POST")
def call_api_post(
    url: str,
    json: dict,
    headers: Optional[dict] = None,
    use_cache: bool = True,
    json_response: bool = True,
):
    response = requests.post(url, json=json, headers=headers)
    response.raise_for_status()
    if response.status_code == 200:
        if json_response:
            response_json = response.json()
            raise_error(response_json)
            return response_json
        else:
            return response
    else:
        raise ValueError(response.text)


def make_file_lock(lock_name: str, timeout: int = 20):
    lock_dir = pjoin(TMP_DIR, "locks")
    mkdirs(lock_dir)
    lock_file_path = pjoin(lock_dir, f"{lock_name}.lock")
    lock = FileLock(lock_file_path, timeout=timeout)  # Timeout after 20 seconds
    return lock


def check_item(item: dict, required_keys: Dict[str, type]) -> dict:
    err = ""
    _keys = set(item.keys())
    if not isinstance(item, dict):
        err += f"Item {item} is not a dict of persona group.\n"
    missing_keys = set(required_keys) - _keys
    if missing_keys:
        err += f"Item {item} is missing required keys: {missing_keys}.\n"
    for key, expected_type in required_keys.items():
        if not isinstance(item[key], expected_type):
            err += f"Item {item} has '{key}' key that is not of type {expected_type.__name__}.\n"
    if err:
        raise ParseError(err)
    return {k: item[k] for k in required_keys}  # return only the required keys


def is_openai_rate_limit_error(e):
    if "Please wait and try again later." in str(e):
        return True
    if "Rate limit is exceeded." in str(e):
        return True
    return False
