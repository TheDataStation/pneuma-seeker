# src/pneuma_seeker/provenance/graph.py
import heapq
import html
import uuid
from logging import Logger

from pyvis.network import Network

from pneuma_seeker.provenance.provenance_helper import (
    SCRIPT_PREAMBLE,
    generate_export_code,
)
from pneuma_seeker.shared.schemas.core.ir_system import RetrieverType

ROOT_NODE_DESCRIPTION = "Imports packages and defines a data structure to hold tables."


class NodeKind:
    """What a provenance node does in the end-to-end script."""

    SETUP = "setup"  # Script runtime (imports, workspace, helpers)
    LOAD = "load"  # Loads a source table from a CSV file
    TRANSFORM = "transform"  # Materialization step producing/modifying a table
    SNAPSHOT = "snapshot"  # Materialization step replayed from a CSV of its result
    CONTEXT = "context"  # Non-tabular context (e.g., web search results)


def infer_node_kind(source_retriever: RetrieverType, description: str) -> str:
    """Kind for nodes created (or persisted) without an explicit one."""
    if description == ROOT_NODE_DESCRIPTION:
        return NodeKind.SETUP
    if source_retriever == RetrieverType.MATERIALIZER:
        return NodeKind.TRANSFORM
    if source_retriever in (RetrieverType.WEB_SEARCH, RetrieverType.WEB_CRAWL):
        return NodeKind.CONTEXT
    return NodeKind.LOAD


class ProvenanceNode:
    """Represents a node in the provenance graph.

    Args:
        source_retriever: Where the data handled by this node comes from.
        python_code: Snippet that performs this step in the end-to-end script.
        description: Markdown description of the step, shown in the UI.
        kind: One of `NodeKind`; inferred from the other args when omitted.
        output_table_id: Table produced (or modified) by this step. Load nodes
            use `dataset.table` for dataset tables.
    """

    def __init__(
        self,
        source_retriever: RetrieverType,
        python_code: str,
        description: str,
        kind: str | None = None,
        output_table_id: str | None = None,
    ):
        self.id = str(uuid.uuid4())
        self.source_retriever = source_retriever
        self.python_code = python_code
        self.description = description
        self.kind = kind or infer_node_kind(source_retriever, description)
        self.output_table_id = output_table_id
        self.parents: list[ProvenanceNode] = []
        self.children: list[ProvenanceNode] = []

    def add_child(self, child: "ProvenanceNode"):
        """Adds a child node to this node."""
        if child not in self.children:
            self.children.append(child)
        if self not in child.parents:
            child.parents.append(self)

    def __repr__(self):
        """String representation of the ProvenanceNode."""
        return f"<ProvenanceNode {self.id} ({self.source_retriever.name})>"


class ProvenanceGraph:
    """Manages a graph of ProvenanceNodes representing data provenance."""

    def __init__(self, logger: Logger, create_default_root: bool = True):
        """Create a provenance graph.

        Args:
            logger: Logger instance for messages.
            create_default_root: When True (default) create a default root node
                with python_code "tables = {}". When False, no default root is
                created. This is useful for deserialization where the serialized
                graph will supply its own nodes.
        """
        self.nodes: dict[str, ProvenanceNode] = {}
        self.logger = logger
        self.ROOT_NODE_CODE = SCRIPT_PREAMBLE

        if create_default_root:
            root_node = ProvenanceNode(
                RetrieverType.USER,
                self.ROOT_NODE_CODE,
                ROOT_NODE_DESCRIPTION,
                kind=NodeKind.SETUP,
            )
            self.nodes[root_node.id] = root_node
            self.logger.info(f"[PROV GRAPH] Root node {root_node.id} initialized.")

    def add_node(self, node: ProvenanceNode, overwrite: bool = False):
        """Adds a node to the graph."""
        if not isinstance(node, ProvenanceNode):
            raise ValueError(f"node must be a ProvenanceNode, got {type(node)}")

        existing = self.nodes.get(node.id)
        if existing and not overwrite:
            raise ValueError(
                f"node {node.id} already exists; set overwrite=True to update"
            )

        if existing and overwrite:
            existing.source_retriever = node.source_retriever
            existing.python_code = node.python_code
            existing.description = node.description
            existing.kind = node.kind
            existing.output_table_id = node.output_table_id
            self.logger.info(f"[PROV GRAPH] Node {node.id} updated (overwrite=True).")
            return existing

        self.nodes[node.id] = node
        self.logger.info(f"[PROV GRAPH] Node {node.id} added.")
        return node

    def connect(self, parent: ProvenanceNode, child: ProvenanceNode):
        """Connects parent and child nodes in the graph."""
        if not isinstance(parent, ProvenanceNode):
            raise ValueError(f"parent must be a ProvenanceNode, got {type(parent)}")
        if not isinstance(child, ProvenanceNode):
            raise ValueError(f"child must be a ProvenanceNode, got {type(child)}")

        if parent.id not in self.nodes:
            self.add_node(parent)
        if child.id not in self.nodes:
            self.add_node(child)

        parent_canon = self.nodes.get(parent.id, parent)
        child_canon = self.nodes.get(child.id, child)

        parent_canon.add_child(child_canon)
        self.logger.info(
            f"[PROV GRAPH] Parent node {parent_canon.id} and child node {child_canon.id} connected successfully."
        )

    def get_nodes(
        self, filters: dict[str, object] | None = None
    ) -> list[ProvenanceNode]:
        """Retrieves all nodes matching given filters (or all if no filters)."""
        if not filters:
            return list(self.nodes.values())
        return [
            node
            for node in self.nodes.values()
            if all(getattr(node, k, None) == v for k, v in filters.items())
        ]

    def get_node(self, filters: dict[str, object]) -> ProvenanceNode | None:
        """Retrieves a single node matching given filters."""
        nodes = self.get_nodes(filters)
        return nodes[0] if len(nodes) > 0 else None

    def get_node_by_id(self, node_id: str) -> ProvenanceNode | None:
        """Retrieves a node by ID (shortcut for get_node({'id': node_id}))."""
        return self.nodes.get(node_id)

    def trace_upstream(self, node: ProvenanceNode, include_self: bool = False):
        """Return all upstream (ancestor) nodes of the given node."""
        visited = set()
        stack = [node]

        while stack:
            n = stack.pop()
            for parent in n.parents:
                if parent not in visited:
                    visited.add(parent)
                    stack.append(parent)

        if include_self:
            visited.add(node)

        return list(visited)

    def trace_downstream(self, node: ProvenanceNode, include_self: bool = False):
        """Return all downstream (descendant) nodes of the given node."""
        visited = set()
        stack = [node]

        while stack:
            n = stack.pop()
            for child in n.children:
                if child not in visited:
                    visited.add(child)
                    stack.append(child)

        if include_self:
            visited.add(node)

        return list(visited)

    def topological_sort(self) -> list[ProvenanceNode]:
        """
        Performs a topological sort (Kahn's algorithm) on the provenance graph.

        Ties are broken by insertion order, so the result follows the order in
        which steps were executed whenever that order is consistent with the
        edges (steps may rewrite a table in place, so execution order matters
        beyond what the edges capture).
        """
        index = {node_id: i for i, node_id in enumerate(self.nodes)}
        indegree = {node_id: 0 for node_id in self.nodes}
        for node in self.nodes.values():
            for child in node.children:
                if child.id in indegree:
                    indegree[child.id] += 1

        heap = [(index[node_id], node_id) for node_id, d in indegree.items() if d == 0]
        heapq.heapify(heap)
        ordered_nodes: list[ProvenanceNode] = []

        while heap:
            _, node_id = heapq.heappop(heap)
            node = self.nodes[node_id]
            ordered_nodes.append(node)
            for child in node.children:
                if child.id not in indegree:
                    continue
                indegree[child.id] -= 1
                if indegree[child.id] == 0:
                    heapq.heappush(heap, (index[child.id], child.id))

        # Detect cycles (non-DAG)
        if len(ordered_nodes) != len(self.nodes):
            self.logger.warning(
                "[PROV GRAPH] Cycle detected during topological ordering!"
            )

        return ordered_nodes

    def get_pipeline_nodes(
        self, target_table_ids: list[str] | None = None
    ) -> list[ProvenanceNode]:
        """Returns the nodes needed to produce the target tables, in execution order.

        For each target, the last node that wrote it is located and all of its
        ancestors are kept (plus the setup node). Snapshot nodes load their
        result from a CSV, so their ancestors are not needed through them.
        Without targets, or when no node records producing them (graphs
        persisted before nodes tracked their output), every node is returned.
        """
        ordered = self.topological_sort()
        if not target_table_ids:
            return ordered

        final_writers: dict[str, ProvenanceNode] = {}
        for node in ordered:
            if node.output_table_id in target_table_ids:
                final_writers[node.output_table_id] = node
        if not final_writers:
            return ordered

        needed: set[ProvenanceNode] = set()
        stack = list(final_writers.values())
        while stack:
            node = stack.pop()
            if node in needed:
                continue
            needed.add(node)
            if node.kind != NodeKind.SNAPSHOT:
                stack.extend(node.parents)
        return [n for n in ordered if n in needed or n.kind == NodeKind.SETUP]

    def get_graph_code(self, target_table_ids: list[str] | None = None) -> str:
        """Returns a runnable end-to-end Python script that rebuilds the target
        tables from CSV files placed next to it.

        The script is: a usage header, the runtime preamble (setup node), one
        commented section per step in execution order, and (when targets are
        given) the export of the target tables to `output/<table>.csv`.
        """
        nodes = self.get_pipeline_nodes(target_table_ids)
        steps = [n for n in nodes if n.kind != NodeKind.SETUP and n.python_code]

        sections = [
            self._render_script_header(steps, target_table_ids or []),
            self.ROOT_NODE_CODE,
        ]
        for step_number, node in enumerate(steps, start=1):
            sections.append(
                f"{self._render_step_banner(step_number, node.description)}\n"
                f"{node.python_code}"
            )
        if target_table_ids:
            sections.append(
                f"{self._render_step_banner(len(steps) + 1, 'Exports the target tables.')}\n"
                f"{generate_export_code(list(target_table_ids))}"
            )
        return "\n\n\n".join(sections) + "\n"

    def get_graph_explanation(
        self, target_table_ids: list[str] | None = None
    ) -> list[str]:
        """Return a list of step explanations for the provenance graph.

        Each list item represents one step (loading a source table or a
        materialization step) in Markdown, in the same order and with the same
        code as the end-to-end script. This format is friendlier for UIs that
        want to page/carousel through steps horizontally instead of rendering a
        single long document.
        """
        steps_nodes = [
            node
            for node in self.get_pipeline_nodes(target_table_ids)
            if node.kind in (NodeKind.LOAD, NodeKind.TRANSFORM, NodeKind.SNAPSHOT)
        ]

        steps: list[str] = []
        for step_number, step_node in enumerate(steps_nodes):
            step_md = (
                f"# Step {step_number + 1}\n\n"
                f"{step_node.description}\n\n"
                f"## Python Code\n\n"
                f"```python\n{step_node.python_code}\n```"
            )
            steps.append(step_md)

        return steps

    def get_snapshot_table_ids(
        self, target_table_ids: list[str] | None = None
    ) -> list[str]:
        """Tables the end-to-end script loads from CSV snapshots (steps whose
        model decisions were too large to embed in the script)."""
        return list(
            dict.fromkeys(
                node.output_table_id
                for node in self.get_pipeline_nodes(target_table_ids)
                if node.kind == NodeKind.SNAPSHOT and node.output_table_id
            )
        )

    @staticmethod
    def _render_step_banner(step_number: int, description: str) -> str:
        rule = "# " + "-" * 77
        lines = [line.rstrip() for line in (description or "").strip().splitlines()]
        first, rest = (lines[0], lines[1:]) if lines else ("", [])
        body = [f"# Step {step_number}: {first}".rstrip()]
        body += [f"#   {line}" if line else "#" for line in rest]
        return "\n".join([rule, *body, rule])

    @staticmethod
    def _render_script_header(
        steps: list[ProvenanceNode], target_table_ids: list[str]
    ) -> str:
        inputs: list[str] = []
        snapshots: list[str] = []
        for node in steps:
            if not node.output_table_id:
                continue
            if node.kind == NodeKind.LOAD:
                dataset, _, table = node.output_table_id.rpartition(".")
                origin = f"table `{table}` of dataset `{dataset}`" if dataset else f"table `{table}`"
                inputs.append(f"#     - {table}.csv  ({origin})")
            elif node.kind == NodeKind.SNAPSHOT:
                line = f"#     - {node.output_table_id}.csv"
                if line not in snapshots:
                    snapshots.append(line)

        lines = [
            "# End-to-end materialization script generated by Pneuma-Seeker.",
            "#",
        ]
        if target_table_ids:
            lines.append(
                f"# It rebuilds the target table(s) {', '.join(target_table_ids)} by replaying"
            )
        else:
            lines.append("# It replays")
        lines[-1] += f" the {len(steps)} recorded step(s) below, in order."
        lines += [
            "#",
            "# Usage:",
            "#   1. Put the input CSV files in the same folder as this script (or in a",
            "#      folder passed as the first argument). File names are matched after",
            "#      the same normalization Pneuma-Seeker applies on ingestion (e.g.,",
            '#      "Sales 2024.csv" also matches table `sales_2024`):',
        ]
        lines += inputs or ["#     (no input files needed)"]
        if snapshots:
            lines += [
                "#      plus these snapshots of steps too large to embed in this script,",
                '#      found in the "Download views" ZIP of the chat in Pneuma-Seeker:',
                *snapshots,
            ]
        lines += [
            "#   2. pip install duckdb pandas",
            "#   3. python <this script> [data_folder]",
        ]
        if target_table_ids:
            lines.append("#   The target tables are written to <data_folder>/output/<table>.csv.")
        lines += [
            "#",
            "# Steps that were performed by a model (LLM-generated columns, semantic",
            "# joins, entity resolution) replay the decisions the model made, so the",
            "# script runs offline and reproduces the same result.",
        ]
        return "\n".join(lines)

    def to_text(self) -> str:
        """Returns a textual representation of the graph."""
        lines = []

        for _, node in sorted(self.nodes.items(), key=lambda x: x[0]):
            retriever_name = (
                getattr(node.source_retriever, "value", None)
                or getattr(node.source_retriever, "name", None)
                or str(node.source_retriever)
            )

            lines.append(f"{node.id}:")
            lines.append(f"  Source Retriever: {retriever_name}")
            lines.append(f"  Python Code: {node.python_code or '<empty>'}")
            lines.append(
                f"  Parents: {', '.join(p.id for p in node.parents) or '<none>'}"
            )
            lines.append(
                f"  Children: {', '.join(c.id for c in node.children) or '<none>'}"
            )
            lines.append("")

        return "\n".join(lines)

    def get_graph_visualization(self) -> str:
        net = Network(notebook=True, directed=True, cdn_resources="in_line")
        seen = set()

        for node in self.nodes.values():
            if node.id not in seen:
                tooltip = (
                    f"Source Retriever: {html.escape(str(node.source_retriever.value))}\n"
                    f"Python Code: {html.escape(node.python_code)}\n"
                    f"# Children: {len(node.children)}\n"
                    f"# Parents: {len(node.parents)}"
                )
                net.add_node(node.id, label=node.id, title=tooltip)
                seen.add(node.id)

            for child in node.children:
                if child.id not in seen:
                    tooltip = (
                        f"Source Retriever: {html.escape(str(child.source_retriever.value))}\n"
                        f"Python Code: {html.escape(child.python_code)}\n"
                        f"# Children: {len(child.children)}\n"
                        f"# Parents: {len(child.parents)}"
                    )
                    net.add_node(child.id, label=child.id, title=tooltip)
                    seen.add(child.id)

                net.add_edge(node.id, child.id)

        return net.generate_html()

    def reset_materialization_nodes(self):
        """Removes previous materialization (processing) nodes.

        This is intended to be called at the beginning of a new materialization run,
        so that the resulting provenance reflects the latest pipeline while still
        keeping upstream/source nodes (retrieved tables, web results, user-provided
        context, etc.).
        """
        keep = {
            node_id: node
            for node_id, node in self.nodes.items()
            if node.source_retriever != RetrieverType.MATERIALIZER
        }

        for node in keep.values():
            node.parents = [p for p in node.parents if p.id in keep]
            node.children = [c for c in node.children if c.id in keep]

        self.nodes = keep
        self.logger.info(
            "[PROV GRAPH] Materialization nodes cleared (non-materializer nodes kept)."
        )
