"""python -m packsmith [--host H] [--port P]   (or the `packsmith` command)"""
import argparse


def main() -> None:
    from packsmith.system.home import drop_to_real_user
    drop_to_real_user()                       # under sudo: run as the real user, so every file Packsmith writes is theirs
    import uvicorn

    from packsmith.system.config import SETTINGS
    ap = argparse.ArgumentParser(prog="packsmith")
    ap.add_argument("--host", default=SETTINGS.host)
    ap.add_argument("--port", type=int, default=SETTINGS.port)
    a = ap.parse_args()
    uvicorn.run("packsmith.gui.server:app", host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
