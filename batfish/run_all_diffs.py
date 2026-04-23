import os
import subprocess
from itertools import combinations

# Configuration
BASE_DIR = "10_examples"
FOLDERS = [str(i) for i in range(1, 11)]  # Folders 1 through 10
CANDIDATES = ["candidate_1", "candidate_2", "candidate_3"]
SCRIPTS = ["diff_advanced.py", "diff_analysis.py"]
OUTPUT_FILE = "comparison_results.txt"

def run_diffs():
    for folder in FOLDERS:
        folder_path = os.path.join(BASE_DIR, folder)
        output_path = os.path.join(folder_path, OUTPUT_FILE)
        
        # Open the output file in the specific folder
        with open(output_path, "w") as f:
            f.write(f"=== Comparison Report for Folder: {folder} ===\n\n")
            
            # Generate pairs (e.g., ('candidate_1', 'candidate_2'), etc.)
            for c1, c2 in combinations(CANDIDATES, 2):
                f.write(f"{'='*20}\nComparing {c1} vs {c2}\n{'='*20}\n")
                
                for script in SCRIPTS:
                    f.write(f"\n--- Running {script} ---\n")
                    
                    # Construct the shell command
                    cmd = [
                        "python3", script,
                        "--folder", f"{folder_path}/",
                        "--c1", c1,
                        "--c2", c2
                    ]
                    
                    try:
                        # Run the script and capture both stdout and stderr
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        f.write(result.stdout)
                        if result.stderr:
                            f.write("\nERRORS:\n")
                            f.write(result.stderr)
                    except Exception as e:
                        f.write(f"Failed to execute {script}: {str(e)}\n")
                
                f.write("\n\n")
        
        print(f"Completed folder {folder}. Results saved to {output_path}")

if __name__ == "__main__":
    run_diffs()