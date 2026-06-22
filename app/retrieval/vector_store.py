import os
import pickle

import faiss
import numpy as np


def create_faiss_index(embeddings):
    """
    Create a FAISS index from embeddings.
    """

    embeddings = np.asarray(
        embeddings,
        dtype="float32",
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)

    return index


def save_index(
    index,
    metadata,
    index_path,
    metadata_path,
):
    """
    Save FAISS index and metadata.
    """

    os.makedirs(
        os.path.dirname(index_path),
        exist_ok=True,
    )

    os.makedirs(
        os.path.dirname(metadata_path),
        exist_ok=True,
    )

    faiss.write_index(
        index,
        index_path,
    )

    with open(
        metadata_path,
        "wb",
    ) as file:
        pickle.dump(
            metadata,
            file,
        )


def load_index(
    index_path,
    metadata_path,
):
    """
    Load FAISS index and metadata.
    """

    if not os.path.exists(
        index_path
    ):
        raise FileNotFoundError(
            f"Index not found: {index_path}"
        )

    if not os.path.exists(
        metadata_path
    ):
        raise FileNotFoundError(
            f"Metadata not found: {metadata_path}"
        )

    index = faiss.read_index(
        index_path
    )

    with open(
        metadata_path,
        "rb",
    ) as file:
        metadata = pickle.load(
            file
        )

    return index, metadata