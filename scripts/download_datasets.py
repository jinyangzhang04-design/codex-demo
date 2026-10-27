"""Download the reproducible public sources used by the first experiment."""
from pathlib import Path
from urllib.request import urlopen, Request

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SOURCES = {
    "esol.csv": "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv",
    "clintox.csv.gz": "https://raw.githubusercontent.com/deepchem/deepchem/master/examples/clintox/datasets/clintox.csv.gz",
    "freesolv.txt": "https://raw.githubusercontent.com/MobleyLab/FreeSolv/master/database.txt",
    # The canonical MoleculeNet source is retained here; some restricted
    # networks return 403 for the S3 host, so preparation reports it clearly.
    "BBBP.csv": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
}

def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for filename, url in SOURCES.items():
        destination = RAW / filename
        try:
            request = Request(url, headers={"User-Agent": "AIDD-DrugScreen/0.1"})
            with urlopen(request, timeout=60) as response:
                destination.write_bytes(response.read())
            print(f"downloaded {filename} ({destination.stat().st_size} bytes)")
        except Exception as exc:
            print(f"unavailable {filename}: {exc}")

if __name__ == "__main__":
    main()
