"""
Tests for lllm/utils/__init__.py
Covers: load_json, save_json, find_md_blocks, find_xml_blocks,
        find_all_xml_tags_sorted, html_collapse, directory_tree,
        check_item, is_openai_rate_limit_error, cache utilities.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ===========================================================================
# JSON utilities
# ===========================================================================


class TestLoadJson(unittest.TestCase):

    def test_existing_file(self):
        import lllm.utils as U
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"key": "value"}, f)
            fname = f.name
        try:
            result = U.load_json(fname)
            self.assertEqual(result, {"key": "value"})
        finally:
            os.unlink(fname)

    def test_missing_file_returns_default(self):
        import lllm.utils as U
        result = U.load_json("/nonexistent/path/file.json", default={"default": True})
        self.assertEqual(result, {"default": True})

    def test_missing_file_empty_default(self):
        import lllm.utils as U
        result = U.load_json("/nonexistent/path.json")
        self.assertEqual(result, {})

    def test_missing_file_none_default_raises(self):
        import lllm.utils as U
        with self.assertRaises(FileNotFoundError):
            U.load_json("/nonexistent/path.json", default=None)

    def test_utf8_content(self):
        import lllm.utils as U
        data = {"emoji": "🎉", "unicode": "こんにちは"}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
            fname = f.name
        try:
            result = U.load_json(fname)
            self.assertEqual(result["emoji"], "🎉")
        finally:
            os.unlink(fname)

    def test_list_json(self):
        import lllm.utils as U
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([1, 2, 3], f)
            fname = f.name
        try:
            result = U.load_json(fname)
            self.assertEqual(result, [1, 2, 3])
        finally:
            os.unlink(fname)


class TestSaveJson(unittest.TestCase):

    def test_saves_and_loads(self):
        import lllm.utils as U
        data = {"a": 1, "b": [1, 2, 3]}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            fname = f.name
        try:
            U.save_json(fname, data)
            result = U.load_json(fname)
            self.assertEqual(result, data)
        finally:
            os.unlink(fname)

    def test_indent_formatting(self):
        import lllm.utils as U
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            fname = f.name
        try:
            U.save_json(fname, {"key": "val"}, indent=2)
            with open(fname) as g:
                content = g.read()
            self.assertIn("\n", content)  # indented = has newlines
        finally:
            os.unlink(fname)

    def test_overwrites_existing(self):
        import lllm.utils as U
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"old": True}, f)
            fname = f.name
        try:
            U.save_json(fname, {"new": True})
            result = U.load_json(fname)
            self.assertEqual(result, {"new": True})
        finally:
            os.unlink(fname)


# ===========================================================================
# XML block parsing
# ===========================================================================


class TestFindXmlBlocks(unittest.TestCase):

    def test_single_block(self):
        import lllm.utils as U
        result = U.find_xml_blocks("<answer>42</answer>", "answer")
        self.assertEqual(result, ["42"])

    def test_multiple_blocks(self):
        import lllm.utils as U
        result = U.find_xml_blocks("<item>a</item><item>b</item><item>c</item>", "item")
        self.assertEqual(result, ["a", "b", "c"])

    def test_no_match_returns_empty(self):
        import lllm.utils as U
        result = U.find_xml_blocks("no tags here", "answer")
        self.assertEqual(result, [])

    def test_multiline_content(self):
        import lllm.utils as U
        text = "<code>\nline1\nline2\n</code>"
        result = U.find_xml_blocks(text, "code")
        self.assertIn("line1", result[0])
        self.assertIn("line2", result[0])

    def test_nested_different_tags(self):
        import lllm.utils as U
        text = "<outer><inner>content</inner></outer>"
        result = U.find_xml_blocks(text, "outer")
        self.assertIn("<inner>content</inner>", result[0])

    def test_wrong_tag_no_match(self):
        import lllm.utils as U
        result = U.find_xml_blocks("<answer>42</answer>", "response")
        self.assertEqual(result, [])

    def test_empty_tags(self):
        import lllm.utils as U
        result = U.find_xml_blocks("<empty></empty>", "empty")
        self.assertEqual(result, [""])

    def test_special_chars_in_content(self):
        import lllm.utils as U
        text = "<desc>Hello & World, x<y, z>w</desc>"
        result = U.find_xml_blocks(text, "desc")
        self.assertEqual(len(result), 1)


# ===========================================================================
# Markdown block parsing
# ===========================================================================


class TestFindMdBlocks(unittest.TestCase):

    def test_single_python_block(self):
        import lllm.utils as U
        text = "```python\nprint('hello')\n```"
        result = U.find_md_blocks(text, "python")
        self.assertEqual(result, ["print('hello')"])

    def test_multiple_blocks_same_tag(self):
        import lllm.utils as U
        text = "```json\n{\"a\": 1}\n```\n\nsome text\n\n```json\n{\"b\": 2}\n```"
        result = U.find_md_blocks(text, "json")
        self.assertEqual(len(result), 2)

    def test_no_match(self):
        import lllm.utils as U
        result = U.find_md_blocks("no code here", "python")
        self.assertEqual(result, [])

    def test_different_tag_no_match(self):
        import lllm.utils as U
        text = "```javascript\nconsole.log('hi')\n```"
        result = U.find_md_blocks(text, "python")
        self.assertEqual(result, [])

    def test_multiline_code(self):
        import lllm.utils as U
        code = "def foo():\n    return 42\n\ndef bar():\n    return foo()"
        text = f"```python\n{code}\n```"
        result = U.find_md_blocks(text, "python")
        self.assertIn("def foo():", result[0])
        self.assertIn("def bar():", result[0])

    def test_content_stripped(self):
        import lllm.utils as U
        text = "```python\n\n  code  \n\n```"
        result = U.find_md_blocks(text, "python")
        # strip() is applied
        self.assertEqual(result[0], "code")


# ===========================================================================
# find_all_xml_tags_sorted
# ===========================================================================


class TestFindAllXmlTagsSorted(unittest.TestCase):

    def test_multiple_different_tags(self):
        import lllm.utils as U
        text = "<b>second</b><a>first</a>"
        result = U.find_all_xml_tags_sorted(text)
        # sorted by position: <b> appears before <a> in the string
        self.assertEqual(result[0]["tag"], "b")
        self.assertEqual(result[1]["tag"], "a")

    def test_returns_dict_structure(self):
        import lllm.utils as U
        text = "<answer>42</answer>"
        result = U.find_all_xml_tags_sorted(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["tag"], "answer")
        self.assertEqual(result[0]["content"], "42")
        self.assertIn("pos", result[0])

    def test_empty_string(self):
        import lllm.utils as U
        result = U.find_all_xml_tags_sorted("")
        self.assertEqual(result, [])

    def test_no_tags(self):
        import lllm.utils as U
        result = U.find_all_xml_tags_sorted("plain text no tags")
        self.assertEqual(result, [])

    def test_content_stripped(self):
        import lllm.utils as U
        text = "<tag>  whitespace content  </tag>"
        result = U.find_all_xml_tags_sorted(text)
        self.assertEqual(result[0]["content"], "whitespace content")


# ===========================================================================
# html_collapse
# ===========================================================================


class TestHtmlCollapse(unittest.TestCase):

    def test_returns_string(self):
        import lllm.utils as U
        result = U.html_collapse("Summary", "Content")
        self.assertIsInstance(result, str)

    def test_contains_details_tags(self):
        import lllm.utils as U
        result = U.html_collapse("My Summary", "My Content")
        self.assertIn("<details>", result)
        self.assertIn("</details>", result)
        self.assertIn("<summary>My Summary</summary>", result)
        self.assertIn("My Content", result)


# ===========================================================================
# directory_tree
# ===========================================================================


class TestDirectoryTree(unittest.TestCase):

    def test_returns_string(self):
        import lllm.utils as U
        with tempfile.TemporaryDirectory() as tmp:
            result = U.directory_tree(Path(tmp))
            self.assertIsInstance(result, str)

    def test_shows_files(self):
        import lllm.utils as U
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "file_a.txt").write_text("content")
            Path(tmp, "file_b.py").write_text("code")
            result = U.directory_tree(Path(tmp))
            self.assertIn("file_a.txt", result)
            self.assertIn("file_b.py", result)

    def test_shows_subdirectory(self):
        import lllm.utils as U
        with tempfile.TemporaryDirectory() as tmp:
            sub = Path(tmp, "subdir")
            sub.mkdir()
            (sub / "nested.txt").write_text("hello")
            result = U.directory_tree(Path(tmp))
            self.assertIn("subdir", result)

    def test_directory_only_mode(self):
        import lllm.utils as U
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "file.txt").write_text("x")
            Path(tmp, "subdir").mkdir()
            result = U.directory_tree(Path(tmp), limit_to_directories=True)
            self.assertIn("subdir", result)
            # file should not appear
            self.assertNotIn("file.txt", result)

    def test_level_limit(self):
        import lllm.utils as U
        with tempfile.TemporaryDirectory() as tmp:
            deep = Path(tmp, "a", "b", "c")
            deep.mkdir(parents=True)
            (deep / "deep.txt").write_text("x")
            result = U.directory_tree(Path(tmp), level=1)
            self.assertNotIn("deep.txt", result)


# ===========================================================================
# check_item
# ===========================================================================


class TestCheckItem(unittest.TestCase):

    def test_valid_item(self):
        import lllm.utils as U
        item = {"name": "Alice", "age": 30, "active": True}
        result = U.check_item(item, {"name": str, "age": int})
        self.assertEqual(result, {"name": "Alice", "age": 30})

    def test_only_required_keys_returned(self):
        """check_item should return ONLY the keys in required_keys."""
        import lllm.utils as U
        item = {"name": "Alice", "age": 30, "extra": "ignored"}
        result = U.check_item(item, {"name": str})
        self.assertNotIn("extra", result)

    def test_missing_key_raises(self):
        """check_item raises when required keys are missing.
        (Implementation raises KeyError when the loop accesses the missing key.)
        """
        import lllm.utils as U
        from lllm.core.const import ParseError
        item = {"name": "Alice"}
        with self.assertRaises((ParseError, KeyError)):
            U.check_item(item, {"name": str, "age": int})

    def test_wrong_type_raises_parse_error(self):
        import lllm.utils as U
        from lllm.core.const import ParseError
        item = {"name": 42}  # should be str
        with self.assertRaises(ParseError):
            U.check_item(item, {"name": str})

    def test_empty_required_keys(self):
        import lllm.utils as U
        item = {"anything": "value"}
        result = U.check_item(item, {})
        self.assertEqual(result, {})


# ===========================================================================
# is_openai_rate_limit_error
# ===========================================================================


class TestIsOpenAIRateLimitError(unittest.TestCase):

    def test_please_wait_message(self):
        import lllm.utils as U
        e = Exception("Please wait and try again later.")
        self.assertTrue(U.is_openai_rate_limit_error(e))

    def test_rate_limit_exceeded_message(self):
        import lllm.utils as U
        e = Exception("Rate limit is exceeded. Try again in 30 seconds.")
        self.assertTrue(U.is_openai_rate_limit_error(e))

    def test_unrelated_error(self):
        import lllm.utils as U
        e = Exception("Connection refused")
        self.assertFalse(U.is_openai_rate_limit_error(e))

    def test_partial_match_please_wait(self):
        import lllm.utils as U
        e = Exception("503 Service Unavailable: Please wait and try again later.")
        self.assertTrue(U.is_openai_rate_limit_error(e))


# ===========================================================================
# Cache utilities
# ===========================================================================


class TestCacheUtilities(unittest.TestCase):

    def test_create_cache_key_deterministic(self):
        import lllm.utils as U
        k1 = U.create_cache_key("func_a", {"x": 1, "y": 2})
        k2 = U.create_cache_key("func_a", {"x": 1, "y": 2})
        self.assertEqual(k1, k2)

    def test_create_cache_key_different_params(self):
        import lllm.utils as U
        k1 = U.create_cache_key("func_a", {"x": 1})
        k2 = U.create_cache_key("func_a", {"x": 2})
        self.assertNotEqual(k1, k2)

    def test_create_cache_key_different_func_keys(self):
        import lllm.utils as U
        k1 = U.create_cache_key("func_a", {"x": 1})
        k2 = U.create_cache_key("func_b", {"x": 1})
        self.assertNotEqual(k1, k2)

    def test_save_and_load_cache(self):
        import lllm.utils as U
        data = {"result": [1, 2, 3], "status": "ok"}
        U.save_cache_by_key("test_cache_suite", "my_key", data)
        loaded = U.load_cache_by_key("test_cache_suite", "my_key")
        self.assertEqual(loaded, data)

    def test_load_nonexistent_returns_none(self):
        import lllm.utils as U
        result = U.load_cache_by_key("test_cache_suite", "definitely_nonexistent_key_xyz")
        self.assertIsNone(result)

    def test_cache_response_and_load(self):
        import lllm.utils as U
        data = {"response": "test value", "code": 200}
        U.cache_response("test_suite_2", "my_func", {"param": "value"}, data)
        loaded = U.load_api_cache("test_suite_2", "my_func", {"param": "value"})
        self.assertEqual(loaded, data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
