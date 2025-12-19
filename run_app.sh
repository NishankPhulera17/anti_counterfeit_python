#!/bin/bash
# Helper script to run the app with proper library paths for pyzbar on macOS

# Set library path for Homebrew zbar on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"
fi

# Run the app
python3 app.py "$@"

