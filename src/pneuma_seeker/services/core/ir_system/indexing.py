import os
import sys

import pandas as pd
from tqdm import tqdm

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
)

from pneuma_seeker.services.core.api.db import DBAPI
from pneuma_seeker.services.core.api.language_model import LanguageModelAPI
from pneuma_seeker.services.core.ir_system.retriever.impl.pneuma_retriever import (
    PneumaRetriever,
)
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.logger import setup_logger
from pneuma_seeker.shared.schemas.core.ir_system import (
    AbstractDocument,
    RetrieverType,
    Table,
    TableContext,
    Text,
)

# KramaBench
INDEXING_ARCHEOLOGY = False
INDEXING_ASTRONOMY = False
INDEXING_BIOMEDICAL = False
INDEXING_ENVIRONMENT = False
INDEXING_LEGAL = False
INDEXING_WILDFIRE = False

# Internal datasets
INDEXING_BUYSITE = False
INDEXING_FEDERAL_STUDENT_LOAN = False


config = Config("../../../../../.env")
logger = setup_logger(log_path=os.path.join(".", "log"))


LARGE_BUYSITE_DATASET = [
    "JI_ADDRESS",
    "JI_ITEM",
    "JI_PURCHASE_ORDER_AUDIT_TRAIL",
    "JI_PURCHASE_ORDER_CUSTOM_FIELDS_GROUP_RESPONSE_16366401",
    "JI_PURCHASE_ORDER_CUSTOM_FIELDS_SINGLE_VALUE_RESPONSE",
    "JI_PURCHASE_ORDER_CUSTOM_FIELDS",
    "JI_PURCHASE_ORDER_LINE",
    "JI_PURCHASE_ORDER_WORKFLOW",
    "JI_PURCHASE_ORDER",
    "JI_REQUISITION_AUDIT_TRAIL",
    "JI_REQUISITION_CUSTOM_FIELDS_GROUP_RESPONSE_16366401",
    "JI_REQUISITION",
]


def index_dataset(
    dataset_name: str,
    metadata_available=False,
    schema_summaries: pd.DataFrame | None = None,
):
    DATASET_DIR = f"../../../../../data_src/{dataset_name}/dataset"

    schema_summary_docs: list[Text] = []
    if schema_summaries is not None:
        required_cols = {"table_name", "summary"}
        missing = required_cols - set(schema_summaries.columns)
        if missing:
            raise ValueError(
                f"schema_summaries is missing required columns: {sorted(missing)}"
            )

        # Align with PneumaRetriever: one schema-summary document per table,
        # concatenating all per-column narrations using " | ".
        group_cols = ["table_name"]
        if "column_name" in schema_summaries.columns:
            schema_summaries = schema_summaries.sort_values(
                by=["table_name", "column_name"], kind="stable"
            )
        else:
            schema_summaries = schema_summaries.sort_values(
                by=["table_name"], kind="stable"
            )

        grouped = schema_summaries.groupby(group_cols, sort=False)
        for table_name, group in tqdm(grouped, desc="Loading schema summaries..."):
            combined_summary = " | ".join(group["summary"].astype(str).tolist())
            schema_summary_docs.append(
                Text(
                    doc_id=f"{DATASET_DIR}/{table_name}_schema_summary",
                    retriever_type=RetrieverType.PNEUMA_RETRIEVER,
                    content=combined_summary,
                    metadata={
                        "table_name": f"{DATASET_DIR}/{table_name}",
                    },
                )
            )

    documents: list[AbstractDocument] = []
    dataset = os.listdir(DATASET_DIR)
    for table_name in tqdm(dataset, desc="Loading dataset..."):
        try:
            if dataset_name == "buysite" and table_name[:-4] in LARGE_BUYSITE_DATASET:
                table = pd.read_csv(f"{DATASET_DIR}/{table_name}", nrows=1000)
            else:
                table = pd.read_csv(f"{DATASET_DIR}/{table_name}")
        except:
            continue
        documents.append(
            Table(
                doc_id=f"{DATASET_DIR}/{table_name}",
                retriever_type=RetrieverType.PNEUMA_RETRIEVER,
                content=table,
                metadata={
                    "table_name": f"{DATASET_DIR}/{table_name}",
                    "dataset_name": dataset_name,
                },
            )
        )

    if metadata_available:
        dataset_metadata = pd.read_csv(
            f"../../../../../data_src/{dataset_name}/metadata.csv"
        )
        for _, row in tqdm(dataset_metadata.iterrows(), desc="Loading metadata..."):
            table_name = row["table_name"]
            description = row["description"]
            documents.append(
                TableContext(
                    doc_id=f"context_{DATASET_DIR}/{table_name}",
                    retriever_type=RetrieverType.PNEUMA_RETRIEVER,
                    content=description,
                    metadata={
                        "table_name": f"{DATASET_DIR}/{table_name}",
                        "dataset_name": dataset_name,
                        "type": "description",
                    },
                )
            )

    ir_sys = PneumaRetriever(
        "user_id",
        "chat_id",
        config,
        DBAPI(config, logger),
        LanguageModelAPI(config, logger),
    )

    if schema_summaries is not None:
        print(
            f"Indexing {len(documents)} documents with {len(schema_summary_docs)} existing schema summaries..."
        )
        ir_sys.index_with_existing_schema_summaries(
            documents,
            existing_schema_summaries=(
                schema_summary_docs if schema_summaries is not None else None
            ),
        )
    else:
        print(
            f"Indexing {len(documents)} documents without existing schema summaries..."
        )
        ir_sys.index(documents)


if INDEXING_ARCHEOLOGY:
    index_dataset("archeology", True)
if INDEXING_ASTRONOMY:
    index_dataset("astronomy", False)
if INDEXING_BIOMEDICAL:
    index_dataset("biomedical", True)
if INDEXING_ENVIRONMENT:
    index_dataset("environment", True)
if INDEXING_BUYSITE:
    index_dataset("buysite", True)
if INDEXING_FEDERAL_STUDENT_LOAN:
    index_dataset("federal_student_loan", True)
if INDEXING_LEGAL:
    index_dataset("legal", False)
if INDEXING_WILDFIRE:
    index_dataset("wildfire", False)
