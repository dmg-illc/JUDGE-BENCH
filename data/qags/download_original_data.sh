#!/bin/bash

# Define URLs of the JSONL files to be downloaded
URL1="https://raw.githubusercontent.com/W4ngatang/qags/master/data/mturk_cnndm.jsonl"
URL2="https://raw.githubusercontent.com/W4ngatang/qags/master/data/mturk_xsum.jsonl"

# Define the target directory
DATA_DIR="original_data"

# Create the target directory if it does not exist
if [ ! -d "$DATA_DIR" ]; then
    echo "Creating directory $DATA_DIR..."
    mkdir -p "$DATA_DIR"
    echo "Directory created."
fi

# Function to download a file and check for errors
download_file() {
    local url=$1
    local target_file=$2
    
    echo "Downloading $url..."
    if curl -o "$target_file" -L "$url"; then
        echo "Downloaded $target_file successfully."
    else
        echo "Failed to download $url" >&2
        exit 1
    fi
}

# Download the JSONL files
download_file "$URL1" "$DATA_DIR/mturk_cnndm.jsonl"
download_file "$URL2" "$DATA_DIR/mturk_xsum.jsonl"

echo "All source files downloaded successfully."
