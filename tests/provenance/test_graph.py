# tests/pneuma_seeker/provenance/test_graph.py
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

from pneuma_seeker.shared.schemas.core.ir_system import RetrieverType
from pneuma_seeker.provenance.graph import (
    ROOT_NODE_DESCRIPTION,
    NodeKind,
    ProvenanceGraph,
    ProvenanceNode,
)


class ProvenanceGraphTests(unittest.TestCase):
    def setUp(self):
        self.logger = MagicMock()
        self.graph = ProvenanceGraph(logger=self.logger)

    def test_node_creation_and_add_child(self):
        """Tests ProvenanceNode creation and adding child nodes."""
        parent = ProvenanceNode(RetrieverType.USER, "x=1", "")
        child = ProvenanceNode(RetrieverType.USER, "y=2", "")
        parent.add_child(child)
        self.assertIn(child, parent.children)
        self.assertIn(parent, child.parents)

    def test_add_node_and_overwrite(self):
        """Tests adding nodes to the ProvenanceGraph and overwriting existing nodes."""
        node = ProvenanceNode(RetrieverType.USER, "code", "")
        added = self.graph.add_node(node)
        self.assertEqual(added, node)
        self.assertIn(node.id, self.graph.nodes)

        with self.assertRaises(ValueError):
            self.graph.add_node(node)

        node.python_code = "new code"
        overwritten = self.graph.add_node(node, overwrite=True)
        self.assertEqual(overwritten.python_code, "new code")

    def test_connect_nodes(self):
        """Tests connecting two nodes in the ProvenanceGraph."""
        parent = ProvenanceNode(RetrieverType.USER, "p", "")
        child = ProvenanceNode(RetrieverType.USER, "c", "")
        self.graph.add_node(parent)
        self.graph.add_node(child)
        self.graph.connect(parent, child)
        self.assertIn(child, parent.children)
        self.assertIn(parent, child.parents)
        self.logger.info.assert_called()

    def test_reset_materialization_nodes(self):
        """reset_materialization_nodes should remove only MATERIALIZER nodes."""
        node_user = ProvenanceNode(RetrieverType.USER, "u", "")
        node_web = ProvenanceNode(RetrieverType.WEB_SEARCH, "w", "")
        node_mat = ProvenanceNode(RetrieverType.MATERIALIZER, "m", "")

        self.graph.add_node(node_user)
        self.graph.add_node(node_web)
        self.graph.add_node(node_mat)
        self.graph.connect(node_web, node_mat)
        self.graph.connect(node_user, node_mat)

        self.graph.reset_materialization_nodes()

        # USER + WEB_SEARCH nodes should remain.
        self.assertIn(node_user.id, self.graph.nodes)
        self.assertIn(node_web.id, self.graph.nodes)
        # MATERIALIZER node should be removed.
        self.assertNotIn(node_mat.id, self.graph.nodes)
        # Edges to removed nodes should be cleaned.
        kept_user = self.graph.get_node_by_id(node_user.id)
        kept_web = self.graph.get_node_by_id(node_web.id)
        self.assertIsNotNone(kept_user)
        self.assertIsNotNone(kept_web)
        self.assertEqual(getattr(kept_user, "children", []), [])  # type: ignore
        self.assertEqual(getattr(kept_web, "children", []), [])  # type: ignore
        self.logger.info.assert_called()

    def test_get_node_by_id_and_filters(self):
        """Tests retrieving nodes by ID and filters."""
        node = ProvenanceNode(RetrieverType.USER, "x", "")
        self.graph.add_node(node)
        self.assertEqual(self.graph.get_node_by_id(node.id), node)
        self.assertEqual(self.graph.get_node({"python_code": "x"}), node)
        self.assertIsNone(self.graph.get_node({"python_code": "y"}))
        user_nodes = self.graph.get_nodes({"source_retriever": RetrieverType.USER})
        self.assertIn(node, user_nodes)

    def test_trace_upstream_downstream(self):
        """Tests tracing upstream and downstream nodes."""
        n1 = ProvenanceNode(RetrieverType.USER, "a", "")
        n2 = ProvenanceNode(RetrieverType.USER, "b", "")
        n3 = ProvenanceNode(RetrieverType.USER, "c", "")
        n1.add_child(n2)
        n2.add_child(n3)
        self.graph.add_node(n1)
        self.graph.add_node(n2)
        self.graph.add_node(n3)
        upstream = self.graph.trace_upstream(n3)
        downstream = self.graph.trace_downstream(n1)
        self.assertEqual(set(upstream), {n1, n2})
        self.assertEqual(set(downstream), {n2, n3})

    def test_to_text_output(self):
        """Tests the textual representation of the ProvenanceGraph."""
        n1 = ProvenanceNode(RetrieverType.USER, "a", "")
        n2 = ProvenanceNode(RetrieverType.USER, "b", "")
        n1.add_child(n2)
        self.graph.add_node(n1)
        self.graph.add_node(n2)
        text = self.graph.to_text()
        self.assertIn(n1.id, text)
        self.assertIn(n2.id, text)
        self.assertIn("Python Code: a", text)
        self.assertIn("Python Code: b", text)

    def test_get_graph_visualization_returns_html(self):
        """Tests that get_graph_visualization returns valid HTML output."""
        node = ProvenanceNode(RetrieverType.USER, "x=1", "")
        self.graph.add_node(node)
        html_output = self.graph.get_graph_visualization()
        self.assertTrue(html_output.strip().startswith("<!DOCTYPE html>") or "<html" in html_output)

    def test_get_graph_explanation(self):
        """Tests the explanation steps returned by get_graph_explanation."""
        # Create a used data node and a processing (materializer) node
        used_node = ProvenanceNode(RetrieverType.WEB_SEARCH, "df = load_data()", "")
        materializer_node = ProvenanceNode(RetrieverType.MATERIALIZER, "result = process(df)", "")

        # Add nodes and connect them so the used node has downstream usage
        self.graph.add_node(used_node)
        self.graph.add_node(materializer_node)
        self.graph.connect(used_node, materializer_node)

        explanation = self.graph.get_graph_explanation()

        # Now returns a list of Markdown step strings.
        self.assertIsInstance(explanation, list)
        self.assertGreaterEqual(len(explanation), 1)
        step_1 = explanation[0]

        # The materializer node should be listed as a processing step
        self.assertIn("Step 1", step_1)
        self.assertIn(materializer_node.python_code, step_1)

        # The default root node code should not be listed under used data
        self.assertNotIn(self.graph.ROOT_NODE_CODE, step_1)

    def test_get_graph_code_concatenation_simple(self):
        """Tests simple linear graph code concatenation."""
        # Simple linear DAG: n1 -> n2 -> n3
        n1 = ProvenanceNode(RetrieverType.USER, "code_a", "")
        n2 = ProvenanceNode(RetrieverType.USER, "code_b", "")
        n3 = ProvenanceNode(RetrieverType.USER, "code_c", "")
        n1.add_child(n2)
        n2.add_child(n3)
        self.graph.add_node(n1)
        self.graph.add_node(n2)
        self.graph.add_node(n3)

        script = self.graph.get_graph_code()
        # Header, then the runtime preamble, then one section per step in order
        self.assertTrue(script.startswith("# End-to-end materialization script"))
        self.assertLess(script.index(self.graph.ROOT_NODE_CODE), script.index("code_a"))
        self.assertLess(script.index("# Step 1:"), script.index("code_a"))
        self.assertLess(script.index("code_a"), script.index("# Step 2:"))
        self.assertLess(script.index("code_b"), script.index("# Step 3:"))
        self.assertTrue(script.rstrip().endswith("code_c"))

    def test_get_graph_code_concatenation_skips_empty_and_preserves_order(self):
        """Tests that empty python_code nodes are skipped and order is preserved."""
        # n1 -> n2(empty) and n1 -> n3 ; empty python_code should be skipped
        n1 = ProvenanceNode(RetrieverType.USER, "first", "")
        n2 = ProvenanceNode(RetrieverType.USER, "", "")
        n3 = ProvenanceNode(RetrieverType.USER, "third", "")
        n1.add_child(n2)
        n1.add_child(n3)
        self.graph.add_node(n1)
        self.graph.add_node(n2)
        self.graph.add_node(n3)

        concat = self.graph.get_graph_code()
        # n1 should appear before n3 since it's the parent; n2 is skipped
        self.assertIn("first", concat)
        self.assertIn("third", concat)
        self.assertTrue(concat.index("first") < concat.index("third"))

    def test_get_graph_code_concatenation_detects_cycle_and_logs(self):
        """Tests that cycles in the graph are detected and logged."""
        # Create a cycle n1 -> n2 -> n3 -> n1. In this case there will be no
        # node with indegree 0 so Kahn's algorithm will detect a cycle and
        # return an empty concatenation while issuing a warning.
        n1 = ProvenanceNode(RetrieverType.USER, "a", "")
        n2 = ProvenanceNode(RetrieverType.USER, "b", "")
        n3 = ProvenanceNode(RetrieverType.USER, "c", "")
        n1.add_child(n2)
        n2.add_child(n3)
        n3.add_child(n1)
        self.graph.add_node(n1)
        self.graph.add_node(n2)
        self.graph.add_node(n3)

        script = self.graph.get_graph_code()
        # The default root node is still present and will be included even
        # when the rest of the graph forms a cycle. Kahn's algorithm will
        # process the root then detect the cycle among the remaining nodes.
        self.assertTrue(script.rstrip().endswith(self.graph.ROOT_NODE_CODE.rstrip()))
        self.logger.warning.assert_called()

    def test_topological_sort_follows_insertion_order_for_independent_nodes(self):
        """Steps without edges between them keep their execution order, even
        when a later step is shallower in the graph."""
        load = ProvenanceNode(RetrieverType.PNEUMA_RETRIEVER, "load", "")
        deep = ProvenanceNode(RetrieverType.MATERIALIZER, "deep", "")
        shallow = ProvenanceNode(RetrieverType.MATERIALIZER, "shallow", "")
        for node in (load, deep, shallow):
            self.graph.add_node(node)
        self.graph.connect(load, deep)

        codes = [n.python_code for n in self.graph.topological_sort()]
        self.assertEqual(codes[1:], ["load", "deep", "shallow"])

    def test_get_graph_code_keeps_only_steps_needed_by_targets(self):
        """The script only includes ancestors of the last writer of each target,
        and exports the targets."""
        used = ProvenanceNode(
            RetrieverType.PNEUMA_RETRIEVER, "load_used", "", kind=NodeKind.LOAD, output_table_id="ds.used"
        )
        unused = ProvenanceNode(
            RetrieverType.PNEUMA_RETRIEVER, "load_unused", "", kind=NodeKind.LOAD, output_table_id="ds.unused"
        )
        create_t1 = ProvenanceNode(RetrieverType.MATERIALIZER, "create_t1", "Creates t1", output_table_id="t1")
        rewrite_t1 = ProvenanceNode(RetrieverType.MATERIALIZER, "rewrite_t1", "Rewrites t1", output_table_id="t1")
        other = ProvenanceNode(RetrieverType.MATERIALIZER, "other", "", output_table_id="t2")
        for node in (used, unused, create_t1, rewrite_t1, other):
            self.graph.add_node(node)
        self.graph.connect(used, create_t1)
        self.graph.connect(create_t1, rewrite_t1)
        self.graph.connect(unused, other)

        script = self.graph.get_graph_code(["t1"])
        for code in ("load_used", "create_t1", "rewrite_t1", "export_tables(['t1'])"):
            self.assertIn(code, script)
        for code in ("load_unused", "other"):
            self.assertNotIn(code, script)
        self.assertIn("#     - used.csv  (table `used` of dataset `ds`)", script)

        steps = self.graph.get_graph_explanation(["t1"])
        self.assertEqual(len(steps), 3)
        self.assertIn("Rewrites t1", steps[2])

    def test_get_graph_code_without_output_info_keeps_all_nodes(self):
        """Graphs persisted before nodes recorded their output table still
        produce a script with every step."""
        node = ProvenanceNode(RetrieverType.MATERIALIZER, "legacy_step", "")
        self.graph.add_node(node)
        self.assertIn("legacy_step", self.graph.get_graph_code(["t1"]))

    def test_node_kind_is_inferred(self):
        """Nodes without an explicit kind (e.g., loaded from older sessions) get one."""
        self.assertEqual(ProvenanceNode(RetrieverType.MATERIALIZER, "", "").kind, NodeKind.TRANSFORM)
        self.assertEqual(ProvenanceNode(RetrieverType.WEB_SEARCH, "", "").kind, NodeKind.CONTEXT)
        self.assertEqual(ProvenanceNode(RetrieverType.PNEUMA_RETRIEVER, "", "").kind, NodeKind.LOAD)
        legacy_root = ProvenanceNode(RetrieverType.USER, "import pandas as pd", ROOT_NODE_DESCRIPTION)
        self.assertEqual(legacy_root.kind, NodeKind.SETUP)

    def test_legacy_root_code_is_replaced_by_current_preamble(self):
        """Old sessions stored a pandas-only root; the script uses the current preamble."""
        graph = ProvenanceGraph(self.logger, create_default_root=False)
        graph.add_node(
            ProvenanceNode(RetrieverType.USER, "import pandas as pd\n\ntables = {}", ROOT_NODE_DESCRIPTION)
        )
        script = graph.get_graph_code()
        self.assertIn(graph.ROOT_NODE_CODE, script)
        self.assertNotIn("tables = {}", script)

    def test_add_node_invalid_type_raises(self):
        """Adding a non-ProvenanceNode should raise ValueError."""
        with self.assertRaises(ValueError):
            self.graph.add_node("not a node") # type: ignore

    def test_connect_invalid_types_raise(self):
        """Connecting non-ProvenanceNode objects should raise ValueError."""
        n = ProvenanceNode(RetrieverType.USER, "a", "")
        with self.assertRaises(ValueError):
            self.graph.connect("not a node", n) # type: ignore
        with self.assertRaises(ValueError):
            self.graph.connect(n, "not a node") # type: ignore

    def test_topological_sort_branching_order(self):
        """Branching DAG should preserve parent-before-child ordering."""
        parent = ProvenanceNode(RetrieverType.USER, "p", "")
        child1 = ProvenanceNode(RetrieverType.USER, "c1", "")
        child2 = ProvenanceNode(RetrieverType.USER, "c2", "")
        parent.add_child(child1)
        parent.add_child(child2)
        self.graph.add_node(parent)
        self.graph.add_node(child1)
        self.graph.add_node(child2)

        ordered = self.graph.topological_sort()
        ids = [n.id for n in ordered]
        self.assertTrue(ids.index(parent.id) < ids.index(child1.id))
        self.assertTrue(ids.index(parent.id) < ids.index(child2.id))

    def test_trace_include_self(self):
        """trace_upstream/downstream should include the node when requested."""
        a = ProvenanceNode(RetrieverType.USER, "a", "")
        b = ProvenanceNode(RetrieverType.USER, "b", "")
        a.add_child(b)
        self.graph.add_node(a)
        self.graph.add_node(b)

        up_with_self = set(self.graph.trace_upstream(b, include_self=True))
        self.assertIn(a, up_with_self)
        self.assertIn(b, up_with_self)

    def test_get_nodes_multi_filter(self):
        """get_nodes should support filtering by multiple attributes."""
        node = ProvenanceNode(RetrieverType.USER, "unique_code", "desc")
        self.graph.add_node(node)
        result = self.graph.get_nodes({"python_code": "unique_code", "source_retriever": RetrieverType.USER})
        self.assertIn(node, result)

    def test_reset_materialization_nodes_cleans_edges(self):
        """reset_materialization_nodes should strip edges to removed MATERIALIZER nodes."""
        user_node = ProvenanceNode(RetrieverType.USER, "u", "")
        materializer = ProvenanceNode(RetrieverType.MATERIALIZER, "m", "")
        materializer.add_child(user_node)
        self.graph.add_node(user_node)
        self.graph.add_node(materializer)

        # Ensure the parent relationship exists before reset
        self.assertIn(materializer, user_node.parents)

        self.graph.reset_materialization_nodes()

        # materializer should be removed
        self.assertNotIn(materializer.id, self.graph.nodes)

        # user_node should remain, but should no longer have the non-user parent
        kept = self.graph.get_node_by_id(user_node.id)
        self.assertIsNotNone(kept)
        self.assertEqual(kept.parents, []) # type: ignore

if __name__ == "__main__":
    unittest.main()
