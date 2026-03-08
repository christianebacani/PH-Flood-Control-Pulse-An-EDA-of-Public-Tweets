import zipfile
import os


def extract_files_from_zipfile(zip_path: str, dest: str = "data") -> None:
    """
    Extract a zip archive to `dest`, skipping any files that already exist.
    This prevents OSError: [Errno 22] Invalid argument on re-runs where the
    CSV files have already been extracted.

    Parameters
    ----------
    zip_path : str  — path to the .zip file
    dest     : str  — destination folder (default: "data")
    """
    if not os.path.exists(zip_path):
        print(f"[extract] Zip not found: {zip_path} — skipping.")
        return

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            target = os.path.join(dest, member.filename)
            if os.path.exists(target):
                print(f"[extract] Already exists, skipping: {member.filename}")
                continue
            zf.extract(member, dest)
            print(f"[extract] Extracted: {member.filename}")