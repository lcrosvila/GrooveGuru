#!/bin/bash

# Create the 'dataset' folder if it doesn't exist
mkdir -p dataset

# Define the array of zip file URLs
urls=(
    "https://search.stepmaniaonline.net/static/new/In%20The%20Groove%201.zip"
    "https://search.stepmaniaonline.net/static/new/In%20The%20Groove%202.zip"
    "https://search.stepmaniaonline.net/static/new/In%20The%20Groove%203.zip"
    "https://search.stepmaniaonline.net/static/new/In%20The%20Groove%20Rebirth.zip"
    "https://search.stepmaniaonline.net/static/new/Community%20Keyboard%20Megapack%20-%20Volume%201.zip"
    "https://search.stepmaniaonline.net/static/new/Community%20Keyboard%20Megapack%20-%20Volume%202.zip"
    "https://search.stepmaniaonline.net/static/new/Community%20Keyboard%20Megapack%20-%20Volume%203.zip"
    "https://search.stepmaniaonline.net/static/new/Community%20Keyboard%20Megapack%20-%20Volume%204.zip"
    "https://search.stepmaniaonline.net/static/new/Keyboard%20Mega%20Pack%201.zip"
    "https://search.stepmaniaonline.net/static/new/Keyboard%20Mega%20Pack%201%C2%BD.zip"
    "https://search.stepmaniaonline.net/static/new/Keyboard%20Mega%20Pack%202.zip"
)

# Iterate over the URLs
for url in "${urls[@]}"; do
    # Extract the filename from the URL
    filename=$(basename "$url")

    # Create a temporary directory to unzip the contents
    temp_dir=$(mktemp -d)

    # Download the zip file
    wget "$url" -P "$temp_dir"

    # Unzip the file
    unzip -q "$temp_dir/$filename" -d "$temp_dir"

    # Move the contents of the unzipped folders to the 'dataset' folder
    find "$temp_dir" -mindepth 1 -maxdepth 1 -exec mv -t dataset/ {} +

    # Remove the temporary directory
    rm -r "$temp_dir"
done
