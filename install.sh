#!/bin/bash

# Define the target directory (local user bin is the safest option)
TARGET_DIR="$HOME/.local/bin"

# Ensure the target directory exists
mkdir -p "$TARGET_DIR"

echo "Installing Naoko tools to $TARGET_DIR..."

# Copy the exact files from your tools directory
cp ./tools/email_account_saver "$TARGET_DIR/"
cp ./tools/email_sender "$TARGET_DIR/"
cp ./tools/system_stats "$TARGET_DIR/"

# Make sure Linux knows they are allowed to be executed
chmod +x "$TARGET_DIR/email_account_saver"
chmod +x "$TARGET_DIR/email_sender"
chmod +x "$TARGET_DIR/system_stats"

echo "✅ Installation complete!"