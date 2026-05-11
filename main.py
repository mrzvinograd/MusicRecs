import sys
from pathlib import Path


def main():
    app_path = Path(__file__).resolve().with_name("app.py")

    try:
        from streamlit.web import cli as streamlit_cli
    except ImportError as exc:
        raise RuntimeError(
            "Streamlit is not installed. Install dependencies with "
            "`pip install -r requirements.txt` and run `python main.py` again."
        ) from exc

    sys.argv = ["streamlit", "run", str(app_path)]
    raise SystemExit(streamlit_cli.main())


if __name__ == "__main__":
    main()
