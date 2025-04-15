import pty


def run_tui():
    # Replace 'your_tui_program' with the command you want to run.
    # This command can also include any required options.
    command = ["claude"]

    # This will spawn the process attached to a pseudo-terminal.
    pty.spawn(command)


if __name__ == "__main__":
    run_tui()
