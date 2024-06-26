#!/bin/bash

# Define base directory and script paths
DATA_PATH="original_data"
RESTORE_SCRIPT="utils/restore_annotated.py"

# Create directories if they do not exist
mkdir -p "${DATA_PATH}/raw"
mkdir -p "${DATA_PATH}/context"
mkdir -p "${DATA_PATH}/annotated"
mkdir -p "${DATA_PATH}/generated"

# URLs for downloading datasets
ANNOTATIONS_URL="https://dl.fbaipublicfiles.com/parlai/projects/roscoe/annotations.zip"
DROP_DATASET_URL="https://ai2-public-datasets.s3.amazonaws.com/drop/drop_dataset.zip"
COSMOS_URL="https://github.com/wilburOne/cosmosqa/raw/master/data/valid.csv"
ESNLI_URL="https://github.com/OanaMariaCamburu/e-SNLI/raw/master/dataset/esnli_dev.csv"
GSM8K_URL="https://github.com/openai/grade-school-math/raw/master/grade_school_math/data/example_model_solutions.jsonl"

# Function to safely execute commands and handle errors
safe_run() {
    if ! "$@"; then
        echo "Error executing $1" >&2
        exit 1
    fi
}

# Function to download and process files
download_and_process() {
    local url=$1
    local target=$2
    local check_path=$3  # Path to check before downloading

    # Check if the target file or directory already exists
    if [ -f "${check_path}" ] || [ -d "${check_path}" ]; then
        echo "$(basename ${check_path}) already exists. Skipping download and processing."
        return
    fi

    echo "Downloading $(basename ${target})..."
    safe_run wget -O "${target}" "${url}"

    # Special handling for zip files
    if [[ "${target}" == *.zip ]]; then
        echo "Unzipping $(basename ${target})..."
        local extract_dir="${target%.*}"
        mkdir -p "${extract_dir}"
        safe_run unzip -o "${target}" -d "${extract_dir}"
        rm -f "${target}"
        
        if [ "$(basename ${target})" == "annotations.zip" ]; then
            mv "${extract_dir}/annotation_release/annotated/"* "${DATA_PATH}/annotated/"
            mv "${extract_dir}/annotation_release/generated/"* "${DATA_PATH}/generated/"
            rm -rf "${extract_dir}"
        fi

        if [ "$(basename ${target})" == "drop_dataset.zip" ]; then
            safe_run mv "${extract_dir}/drop_dataset/drop_dataset_dev.json" "${DATA_PATH}/raw/drop.txt"
            rm -rf "${extract_dir}"
        fi
    fi
}

# Download and organize datasets
download_and_process "${ANNOTATIONS_URL}" "${DATA_PATH}/annotations.zip" "${DATA_PATH}/annotated/cosmos.csv"
download_and_process "${DROP_DATASET_URL}" "${DATA_PATH}/drop_dataset.zip" "${DATA_PATH}/raw/drop.txt"
download_and_process "${COSMOS_URL}" "${DATA_PATH}/raw/cosmos.txt" "${DATA_PATH}/raw/cosmos.txt"
download_and_process "${ESNLI_URL}" "${DATA_PATH}/raw/esnli.txt" "${DATA_PATH}/raw/esnli.txt"
download_and_process "${GSM8K_URL}" "${DATA_PATH}/raw/gsm8k.txt" "${DATA_PATH}/raw/gsm8k.txt"

# Restore annotated sets; adjust according to available datasets
safe_run python "$RESTORE_SCRIPT" --datasets drop esnli cosmos gsm8k
