from __future__ import annotations
import sys

from config import AppConfig  # re-export
from processor import VideoProcessor  # re-export
from cli import parse_cli_args, main_cli
from ui_app import App

if __name__ == "__main__":
    # Si vienen flags de CLI, usamos modo CLI; si no, levantamos la UI
    if any(a in sys.argv for a in ["--cli", "--source", "--webcam"]):
        exit(main_cli(parse_cli_args(sys.argv[1:])))
    else:
        app = App()
        app.mainloop()
