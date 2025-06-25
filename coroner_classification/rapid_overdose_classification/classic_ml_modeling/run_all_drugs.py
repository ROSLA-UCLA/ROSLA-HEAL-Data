import subprocess
import sys
from rapid_overdose_classification.constants import drug_cols
from tqdm import tqdm

if __name__ == "__main__":

    embedder = sys.argv[1]

    for drug in tqdm(drug_cols):
        print(f"Running training for {embedder} for {drug}")
        command = ["python", embedder, drug]
        print(f"Running: {' '.join(command)}")
        subprocess.run(command)
