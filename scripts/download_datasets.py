"""Download the reproducible public sources used by the first experiment."""
from pathlib import Path
from urllib.request import urlopen, Request
import hashlib

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SOURCES = {
    "esol.csv": "https://raw.githubusercontent.com/deepchem/deepchem/455d07f3e17e880a4d980b5b0365e4a02b417e14/datasets/delaney-processed.csv",
    "clintox.csv.gz": "https://raw.githubusercontent.com/deepchem/deepchem/455d07f3e17e880a4d980b5b0365e4a02b417e14/examples/clintox/datasets/clintox.csv.gz",
    "freesolv.txt": "https://raw.githubusercontent.com/MobleyLab/FreeSolv/6c7d19b4b565537365ffd22006aa2cd4643200c6/database.txt",
    # The canonical MoleculeNet source is retained here; some restricted
    # networks return 403 for the S3 host, so preparation reports it clearly.
    "BBBP.csv": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
}
EXPECTED_SHA256 = {
    "esol.csv": "8c06a76f0c6487d29ab0f903e6a7a7139f189ab3c1178f159c8be8964602f189",
    "freesolv.txt": "2d13f095713bc39b85f85dd7b4e5483fbb12fc694bf253bb1d92a4c4d484f260",
    "BBBP.csv": "d07a38487aeac5cee5508413e468043ef3097451d2a112701c2d60be9ec6b662",
    "clintox.csv.gz": "2ee7050a830fe9ae7a8218e6577fa6f54284947034ff904b6d8204e993a297cc",
}

def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for filename, url in SOURCES.items():
        destination = RAW / filename
        try:
            request = Request(url, headers={"User-Agent": "AIDD-DrugScreen/0.1"})
            with urlopen(request, timeout=60) as response:
                destination.write_bytes(response.read())
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            if digest != EXPECTED_SHA256[filename]:
                destination.unlink()
                raise RuntimeError(f"checksum mismatch: {digest}")
            print(f"downloaded and verified {filename} ({destination.stat().st_size} bytes)")
        except Exception as exc:
            print(f"unavailable {filename}: {exc}")

if __name__ == "__main__":
    main()
