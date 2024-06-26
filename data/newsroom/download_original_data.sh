#!/bin/bash

# Define URLs of the data files to be downloaded
URL="https://raw.githubusercontent.com/lil-lab/newsroom/master/humaneval/newsroom-human-eval.csv"

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
download_file "$URL" "$DATA_DIR/newsroom-human-eval.csv"

echo "All source files downloaded successfully."