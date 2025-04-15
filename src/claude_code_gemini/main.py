#!/usr/bin/env python3
"""
This script implements the main CLI entry points for the claude-code-gemini package.
It handles the "gemini" and "gemini-server" commands.
"""

import os
import subprocess
import sys
import socket
import time
import pty

PROXY_PORT = 8082
SCREEN_SESSION_NAME = "gemini_proxy"


def is_server_running(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect(("127.0.0.1", port))
            return True
        except ConnectionRefusedError:
            return False


def start_proxy_server():
    # Start the proxy server in a detached screen session.
    subprocess.run(["screen", "-dmS", SCREEN_SESSION_NAME, "gemini-server"])
    # Wait a bit for the server to start.
    time.sleep(2)


def launch_tui():
    # Launch interactive TUI with a pseudo-terminal (assumes the existence of a command named 'claude')
    os.environ["ANTHROPIC_BASE_URL"] = "http://localhost:8082"
    pty.spawn(["claude"])


def gemini_command():
    """Entry point for the 'gemini' command."""
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY must be set.")
        sys.exit(1)

    if not is_server_running(PROXY_PORT):
        print("Proxy server not running. Starting in detached screen session...")
        start_proxy_server()
    else:
        print("Proxy server already running on port", PROXY_PORT)

    launch_tui()


def server_command():
    """Entry point for the 'gemini-server' command."""
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY must be set.")
        exit(1)

    print("Initializing Anthropic-to-Gemini proxy server...")
    from claude_code_gemini.server import run_server

    run_server()


if __name__ == "__main__":
    # This allows testing the script directly
    gemini_command()
