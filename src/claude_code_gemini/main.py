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
import shutil # Added import

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
    # Ensure TERM is set for the server process as well, just in case
    env = os.environ.copy()
    env['TERM'] = os.environ.get('TERM', 'screen') # Preserve TERM
    subprocess.run(["screen", "-dmS", SCREEN_SESSION_NAME, "gemini-server"], env=env)
    # Wait a bit for the server to start.
    time.sleep(2)


def launch_tui():
    # Launch interactive TUI via screen in the current terminal
    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = "http://localhost:8082"
    # Explicitly pass the parent's TERM variable to screen's environment
    env['TERM'] = os.environ.get('TERM', 'screen') # Use 'screen' as fallback

    if not shutil.which("screen"):
        print("Error: 'screen' command not found. Please install screen (e.g., sudo apt install screen).", file=sys.stderr)
        sys.exit(1)

    try:
        # Use '-q' for quieter screen startup/shutdown
        # Pass the modified environment using the 'env' argument
        result = subprocess.run(["screen", "-q", "claude"], check=True, env=env)
    except FileNotFoundError:
        # This case should technically be caught by shutil.which, but included for robustness
        print("Error: 'screen' command not found. Please install screen.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        # Error likely means claude command itself failed within screen
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        # Catch Ctrl+C if the user interrupts screen itself
        print("\nExiting.")
        sys.exit(0)


def gemini_command():
    """Entry point for the 'gemini' command."""

    if not shutil.which("claude"):
        print("Error: 'claude' command not found in PATH. Please install Claude Code:")
        print("npm install -g @anthropic-ai/claude-code")
        sys.exit(1)

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
