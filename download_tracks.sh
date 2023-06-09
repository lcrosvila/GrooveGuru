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
for url in "${urls[@]}"
do
    # Extract the filename from the URL
    filename=$(basename "$url")
    
    # Download the zip file
    wget -O "$filename" "$url"
    
    # Unzip the file and move its contents to the 'dataset' folder
    unzip -q "$filename" -d dataset
    
    # Remove the downloaded zip file
    rm "$filename"
done