import argparse
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", type=Path, required=True)
    parser.add_argument("--extract-dir", type=Path)
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()

    if not args.zip_path.exists():
        raise FileNotFoundError(args.zip_path)

    with zipfile.ZipFile(args.zip_path) as zf:
        names = zf.namelist()
        print(f"files_in_zip={len(names)}")
        for name in names[:200]:
            print(name)
        if args.list_only:
            return
        if args.extract_dir:
            args.extract_dir.mkdir(parents=True, exist_ok=True)
            zf.extractall(args.extract_dir)
            print(f"extracted_to={args.extract_dir}")


if __name__ == "__main__":
    main()
