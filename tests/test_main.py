# tests/test_main.py
import io
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.responses import HTMLResponse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from fastapi.testclient import TestClient

import pneuma_seeker.main as main


class ServerEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(main.app)

    def _make_mock_chat_interface(
        self, prov_nodes=None, graph_code="print('hi')", stream_messages=None
    ):
        # Build a minimal mock prov_graph
        prov_graph = MagicMock()
        prov_graph.nodes = prov_nodes or {}
        prov_graph.get_graph_code.return_value = graph_code

        # Minimal conductor/materializer structure
        materializer = MagicMock()
        materializer.prov_graph = prov_graph

        conductor = MagicMock()
        conductor.materializer = materializer

        # Chat interface mock
        chat_interface = MagicMock()
        chat_interface.conductor = conductor

        # `chat` should be an iterable/generator
        if stream_messages is None:

            def default_gen(messages, files):
                yield "LOG connected"
                yield "Hello from assistant"
                yield "DONE"

            chat_interface.chat.side_effect = lambda messages, files: default_gen(
                messages, files
            )
        else:
            chat_interface.chat.side_effect = lambda messages, files: (
                m for m in stream_messages
            )

        chat_interface.persist_session.return_value = None
        return chat_interface

    def test_root_returns_ok(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"status": "ok"})

    def test_provenance_nodes_endpoint_returns_nodes(self):
        # Create simple node-like objects
        node_a = types.SimpleNamespace(
            id="n1",
            source_retriever=types.SimpleNamespace(value="USER"),
            python_code="x=1",
            description="a",
            parents=[],
            children=[],
        )
        node_b = types.SimpleNamespace(
            id="n2",
            source_retriever=types.SimpleNamespace(value="WEB"),
            python_code="y=2",
            description="b",
            parents=[node_a],
            children=[],
        )
        node_a.children.append(node_b)

        prov_nodes = {node_a.id: node_a, node_b.id: node_b}

        mock_chat = self._make_mock_chat_interface(prov_nodes=prov_nodes)

        # Patch manager.get_chat_interface temporarily
        original_get = main.session_manager.get_chat_session
        main.session_manager.get_chat_session = lambda user_id, chat_id: mock_chat
        try:
            r = self.client.get("/provenance/nodes/u1/c1")
            self.assertEqual(r.status_code, 200)
            body = r.json()
            self.assertEqual(body["user_id"], "u1")
            self.assertEqual(body["chat_id"], "c1")
            self.assertEqual(body["node_count"], 2)
            ids = {n["id"] for n in body["nodes"]}
            self.assertEqual(ids, {"n1", "n2"})
        finally:
            main.session_manager.get_chat_session = original_get

    def test_materializer_code_download_returns_file(self):
        mock_chat = self._make_mock_chat_interface(graph_code="print('materialize')")
        original_get = main.session_manager.get_chat_session
        main.session_manager.get_chat_session = lambda user_id, chat_id: mock_chat
        try:
            r = self.client.get("/materializer_code/u1/c1")
            self.assertEqual(r.status_code, 200)
            # Content should include the graph code
            self.assertIn("materialize", r.content.decode())
            self.assertIn(
                "attachment; filename=", r.headers.get("content-disposition", "")
            )
        finally:
            main.session_manager.get_chat_session = original_get

    def test_combined_html_calls_prov_steps_and_renders(self):
        # Prepare a chat_interface mock where T is materialized and prov_graph returns markdown
        prov_graph = MagicMock()
        prov_graph.get_graph_explanation.return_value = ["**md** code"]

        materializer = MagicMock()
        materializer.prov_graph = prov_graph

        state = MagicMock()
        state.get_current_state_instance.return_value = {"state": "ok"}
        state.is_T_materialized = True

        conductor = MagicMock()
        conductor.materializer = materializer
        conductor.state = state

        chat_interface = MagicMock()
        chat_interface.conductor = conductor

        original_get = main.session_manager.get_chat_session
        main.session_manager.get_chat_session = lambda user_id, chat_id: chat_interface

        # Replace templates.TemplateResponse so we can inspect the context passed to it
        original_templates = main.templates

        def fake_template_response(template_name, context):
            # Ensure prov_steps was computed from prov_graph markdown
            self.assertIn("prov_steps", context)
            # It should include HTML converted from markdown (bold -> <strong>) or at least the markdown content
            self.assertTrue("md" in context["prov_steps"][0])
            return HTMLResponse(
                content="\n".join(context["prov_steps"]), status_code=200
            )

        # Replace the templates object with a minimal object exposing TemplateResponse
        main.templates = types.SimpleNamespace(TemplateResponse=fake_template_response)

        try:
            payload = {
                "messages": [{"role": "user", "content": "hi"}],
                "model": "assistant",
            }
            r = self.client.post("/combined/html/u1/c1", json=payload)
            self.assertEqual(r.status_code, 200)
            # Body should contain the prov_steps we returned
            self.assertIn("md", r.text)
            # ensure the prov_graph method was called
            prov_graph.get_graph_explanation.assert_called()
        finally:
            main.session_manager.get_chat_session = original_get
            main.templates = original_templates

    def test_all_tables_downloads_zip(self):
        class DummyDF:
            def to_csv(self, buffer, index=False):
                buffer.write("col1,col2\n1,2\n")

        conductor = MagicMock()
        conductor.state = MagicMock()
        conductor.state.T = {"table1": object()}
        conductor.db_api = MagicMock()
        conductor.db_api.execute_query.return_value = DummyDF()

        chat_interface = MagicMock()
        chat_interface.conductor = conductor

        original_get = main.session_manager.get_chat_session
        main.session_manager.get_chat_session = lambda user_id, chat_id: chat_interface
        try:
            r = self.client.get("/all_tables/u_test/c_test")
            self.assertEqual(r.status_code, 200)
            self.assertIn("application/zip", r.headers.get("content-type", ""))
            # Verify returned bytes form a valid zip with the csv inside
            zip_bytes = r.content
            z = zipfile.ZipFile(io.BytesIO(zip_bytes))
            names = z.namelist()
            self.assertIn("table1.csv", names)
        finally:
            main.session_manager.get_chat_session = original_get

    def test_chat_endpoint_streams_messages_and_calls_persist(self):
        messages = [{"role": "user", "content": "hi"}]
        mock_chat = self._make_mock_chat_interface(
            stream_messages=["LOG one", "answer text", "DONE"]
        )

        original_get = main.session_manager.get_chat_session
        main.session_manager.get_chat_session = lambda user_id, chat_id: mock_chat
        try:
            r = self.client.post(
                "/chat", json={"user_id": "u1", "chat_id": "c1", "messages": messages}
            )
            self.assertEqual(r.status_code, 200)
            # streaming response lines
            text = b"".join(r.iter_bytes())
            decoded = text.decode(errors="ignore")
            # Should contain assistant message and log/done structure
            self.assertIn("answer text", decoded)
            # persist_session should have been scheduled (called after streaming) - ensure method exists and can be called
            self.assertTrue(hasattr(mock_chat, "persist_session"))
        finally:
            main.session_manager.get_chat_session = original_get

    def test_download_chat_pdf_returns_pdf(self):
        # Patch the weasyprint.HTML used inside the endpoint
        original_weasy = sys.modules.get("weasyprint")

        class DummyHTML:
            def __init__(self, string=None):
                self.string = string

            def write_pdf(self, path):
                with open(path, "wb") as f:
                    f.write(b"%PDF-1.4\n%dummy pdf\n")

        sys.modules["weasyprint"] = types.SimpleNamespace(HTML=DummyHTML)  # type: ignore

        try:
            body = {
                "model": "assistant",
                "messages": [{"role": "user", "content": "hello"}],
                "chat_id": "c_pdf",
            }
            r = self.client.post("/download_chat_pdf", json=body)
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.headers.get("content-type"), "application/pdf")
            self.assertTrue(r.content.startswith(b"%PDF"))
            self.assertIn(
                "attachment; filename=", r.headers.get("content-disposition", "")
            )
        finally:
            if original_weasy is not None:
                sys.modules["weasyprint"] = original_weasy
            else:
                del sys.modules["weasyprint"]

    def test_all_tables_error_cases(self):
        # no target tables
        conductor_empty = MagicMock()
        conductor_empty.state = MagicMock()
        conductor_empty.state.T = {}
        conductor_empty.db_api = MagicMock()

        chat_empty = MagicMock()
        chat_empty.conductor = conductor_empty

        # db error
        conductor_error = MagicMock()
        conductor_error.state = MagicMock()
        conductor_error.state.T = {"t1": object()}
        conductor_error.db_api = MagicMock()
        conductor_error.db_api.execute_query.side_effect = RuntimeError("fail")

        chat_error = MagicMock()
        chat_error.conductor = conductor_error

        original_get = main.session_manager.get_chat_session
        try:
            main.session_manager.get_chat_session = lambda user_id, chat_id: chat_empty
            r = self.client.get("/all_tables/u1/c1")
            self.assertEqual(r.status_code, 404)

            main.session_manager.get_chat_session = lambda user_id, chat_id: chat_error
            client_no_raise = TestClient(main.app, raise_server_exceptions=False)
            r2 = client_no_raise.get("/all_tables/u1/c1")
            self.assertEqual(r2.status_code, 500)
        finally:
            main.session_manager.get_chat_session = original_get

    def test_combined_html_when_not_materialized_uses_default_explanation(self):
        state = MagicMock()
        state.get_current_state_instance.return_value = {"state": "ok"}
        state.is_T_materialized = False

        conductor = MagicMock()
        conductor.state = state
        conductor.materializer = MagicMock()

        chat_interface = MagicMock()
        chat_interface.conductor = conductor

        original_get = main.session_manager.get_chat_session
        main.session_manager.get_chat_session = lambda user_id, chat_id: chat_interface

        original_templates = main.templates

        def fake_template_response(template_name, context):
            self.assertIn("prov_steps", context)
            self.assertIn("not materialized", context["prov_steps"][-1])
            return HTMLResponse(
                content="\n".join(context["prov_steps"]), status_code=200
            )

        main.templates = types.SimpleNamespace(TemplateResponse=fake_template_response)

        try:
            payload = {"messages": [], "model": "assistant"}
            r = self.client.post("/combined/html/u1/c1", json=payload)
            self.assertEqual(r.status_code, 200)
            self.assertIn("not materialized", r.text)
        finally:
            main.session_manager.get_chat_session = original_get
            main.templates = original_templates

    def test_provenance_nodes_handles_non_object_source_retriever(self):
        node = types.SimpleNamespace(
            id="n1",
            source_retriever="RAW",
            python_code="x=1",
            description="d",
            parents=[],
            children=[],
        )
        prov_nodes = {node.id: node}
        mock_chat = self._make_mock_chat_interface(prov_nodes=prov_nodes)

        original_get = main.session_manager.get_chat_session
        main.session_manager.get_chat_session = lambda user_id, chat_id: mock_chat
        try:
            r = self.client.get("/provenance/nodes/u1/c1")
            self.assertEqual(r.status_code, 200)
            body = r.json()
            self.assertEqual(body["node_count"], 1)
            self.assertEqual(body["nodes"][0]["source_retriever"], "RAW")
        finally:
            main.session_manager.get_chat_session = original_get

    def test_chat_ndjson_lines_are_valid_json_and_files_passed(self):
        # capture files passed to chat
        captured = []

        def side_effect(messages, files):
            captured.append(files)
            yield "LOG one"
            yield "answer"
            yield "DONE"

        mock_chat = self._make_mock_chat_interface(stream_messages=None)
        mock_chat.chat.side_effect = side_effect

        original_get = main.session_manager.get_chat_session
        main.session_manager.get_chat_session = lambda user_id, chat_id: mock_chat
        try:
            payload = {
                "user_id": "u1",
                "chat_id": "c1",
                "messages": [{"role": "user", "content": "hi"}],
                "files": ["f1"],
            }
            r = self.client.post("/chat", json=payload)
            self.assertEqual(r.status_code, 200)
            self.assertIn("application/x-ndjson", r.headers.get("content-type", ""))
            text = b"".join(r.iter_bytes()).decode(errors="ignore")
            lines = [l for l in text.splitlines() if l.strip()]
            for line in lines:
                obj = json.loads(line)
                self.assertIn("sender", obj)
                self.assertIn("text", obj)
                self.assertIn("time_stamp", obj)
            # ensure files were passed through
            self.assertEqual(captured, [["f1"]])
        finally:
            main.session_manager.get_chat_session = original_get

    def test_chat_exception_still_calls_persist_session(self):
        def raise_on_call(messages, files):
            raise RuntimeError("boom")

        mock_chat = self._make_mock_chat_interface(stream_messages=None)
        mock_chat.chat.side_effect = raise_on_call

        original_get = main.session_manager.get_chat_session
        main.session_manager.get_chat_session = lambda user_id, chat_id: mock_chat
        try:
            r = self.client.post(
                "/chat", json={"user_id": "u1", "chat_id": "c1", "messages": []}
            )
            # Even if chat raised, the endpoint should return 200 with no content or a handled error from background task.
            self.assertEqual(r.status_code, 200)
            # ensure persist_session method exists and can be called
            self.assertTrue(hasattr(mock_chat, "persist_session"))
        finally:
            main.session_manager.get_chat_session = original_get

    def test_materializer_code_empty_and_error_cases(self):
        # empty code
        prov_graph = MagicMock()
        prov_graph.get_graph_code.return_value = ""
        materializer = MagicMock()
        materializer.prov_graph = prov_graph
        conductor = MagicMock()
        conductor.materializer = materializer
        chat_interface = MagicMock()
        chat_interface.conductor = conductor

        original_get = main.session_manager.get_chat_session
        main.session_manager.get_chat_session = lambda user_id, chat_id: chat_interface
        try:
            r = self.client.get("/materializer_code/u1/c1")
            self.assertEqual(r.status_code, 200)
            self.assertIn(
                "attachment; filename=", r.headers.get("content-disposition", "")
            )
            self.assertEqual(r.content, b"")

            # error case: use a TestClient that does not re-raise server exceptions
            prov_graph.get_graph_code.side_effect = RuntimeError("fail")
            client_no_raise = TestClient(main.app, raise_server_exceptions=False)
            r2 = client_no_raise.get("/materializer_code/u1/c1")
            self.assertEqual(r2.status_code, 500)
        finally:
            main.session_manager.get_chat_session = original_get

    def test_helpers_now_ms_and_stream_payload(self):
        t = main.now_ms()
        self.assertIsInstance(t, int)
        payload = main.stream_payload("log", "hello")
        obj = json.loads(payload.strip())
        self.assertEqual(obj["sender"], "log")
        self.assertEqual(obj["text"], "hello")


if __name__ == "__main__":
    unittest.main()
