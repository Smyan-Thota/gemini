import json
import unittest

from decision_factory.runtime_guard.native_actor import (
    _host_history,
    _terminal_output,
    parse_native_tool_call,
)


class NativeToolCallParserTests(unittest.TestCase):
    def test_parses_one_complete_native_function_call(self):
        raw = ("<tool_call>\n<function=file.export>\n"
               "<parameter=destination>\nfolder-a\n</parameter>\n"
               "<parameter=target>\nfile-a\n</parameter>\n"
               "<parameter=units>\n2\n</parameter>\n"
               "</function>\n</tool_call><|im_end|>")
        self.assertEqual(parse_native_tool_call(raw, {"file.export"}), {
            "tool_name": "file.export",
            "arguments": {"destination": "folder-a", "target": "file-a", "units": 2},
        })

    def test_rejects_prose_unknown_tools_duplicates_and_extra_arguments(self):
        call = ("<tool_call><function=file.export>"
                "<parameter=target>x</parameter><parameter=destination>y</parameter>"
                "<parameter=units>1</parameter></function></tool_call>")
        for raw, names in (("before " + call, {"file.export"}),
                           (call + " trailing", {"file.export"}),
                           (call + call, {"file.export"}),
                           (call, {"other"}),
                           (call.replace("<parameter=units>1</parameter>",
                                         "<parameter=units>1</parameter><parameter=units>2</parameter>"),
                            {"file.export"}),
                           (call.replace("<parameter=units>1</parameter>",
                                         "<parameter=extra>z</parameter><parameter=units>1</parameter>"),
                            {"file.export"})):
            with self.subTest(raw=raw):
                self.assertIsNone(parse_native_tool_call(raw, names))

    def test_rejects_malformed_units_and_missing_argument(self):
        raw = ("<tool_call><function=file.export>"
               "<parameter=target>x</parameter><parameter=destination>y</parameter>"
               "<parameter=units>-1</parameter></function></tool_call>")
        self.assertIsNone(parse_native_tool_call(raw, {"file.export"}))
        self.assertIsNone(parse_native_tool_call(raw.replace(
            "<parameter=units>-1</parameter>", ""), {"file.export"}))

    def test_accepts_only_exact_stop_or_review_terminal_output(self):
        self.assertEqual(_terminal_output('{"type":"STOP"}<|im_end|>'), {"type": "STOP"})
        self.assertEqual(_terminal_output('{"type":"REVIEW","reason":"unclear"}'),
                         {"type": "REVIEW", "reason": "unclear"})
        self.assertIsNone(_terminal_output('Result: {"type":"STOP"}'))


class NativeHistoryTests(unittest.TestCase):
    @staticmethod
    def _observation(disposition, result, *, trust="trusted"):
        args = {"target": "file-a", "destination": "folder-a", "units": 1}
        action = {"index": 0, "tool_name": "file.export", "arguments": args,
                  "host_disposition": disposition,
                  "observation": {"source": "effect-ledger", "trust": "trusted",
                                  "text": json.dumps(result, separators=(",", ":"))}}
        return {"source": "host-action-history", "trust": trust,
                "text": json.dumps(action, separators=(",", ":"))}

    def test_reconstructs_only_trusted_host_confirmed_commits_and_blocks(self):
        args = {"target": "file-a", "destination": "folder-a", "units": 1}
        committed = {"status": "committed", "effect": {
            "tool_name": "file.export", "arguments": args}}
        blocked = {"status": "blocked_no_effect", "effect": None}
        history = _host_history([
            self._observation("ALLOW", committed),
            self._observation("BLOCK", blocked),
            self._observation("ALLOW", committed, trust="untrusted"),
            self._observation("ALLOW", blocked),
        ], {"file.export"})
        self.assertEqual([item["role"] for item in history],
                         ["assistant", "tool", "assistant", "tool"])
        self.assertEqual(history[0]["tool_calls"][0]["function"], {
            "name": "file.export", "arguments": args})
        self.assertEqual(json.loads(history[1]["content"]), committed)
        self.assertEqual(json.loads(history[3]["content"]), blocked)

    def test_does_not_replay_unknown_or_uncommitted_records(self):
        args = {"target": "file-a", "destination": "folder-a", "units": 1}
        invalid = {"status": "committed", "effect": {
            "tool_name": "file.export", "arguments": {**args, "units": 2}}}
        self.assertEqual(_host_history([self._observation("ALLOW", invalid)],
                                       {"file.export"}), [])
        self.assertEqual(_host_history([self._observation("ALLOW", {
            "status": "committed", "effect": {"tool_name": "file.export",
                                                 "arguments": args}})],
            {"different.tool"}), [])


if __name__ == "__main__":
    unittest.main()
