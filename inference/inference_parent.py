# parent_script.py
"""Master script for batch-processing Hydromoth recordings using pathlib
-----------------------------------------------------------------------
Only the path-handling logic has been modernised; all other behaviour is
unchanged.
"""

from pathlib import Path
import subprocess
from datetime import datetime
import pandas as pd
import numpy as np  # retained for compatibility, remove if unused elsewhere

# -------------------------------------------------------------------------
# USER PARAMETERS — edit only these values
# -------------------------------------------------------------------------

# Copy in the directory containing the Hydromoth recordings
new_audio_dir: Path = Path(r"/path/to/your/audio/files")  # EDIT THIS PATH  # <-------

# Copy in the directory where results should be written
output_folder: Path = Path(r"/path/to/output/directory")  # EDIT THIS PATH  # <-------

# Path to the saved model - use legacy model or your own .keras model
model_dir: Path = Path(
    r"../models/legacy_model"  # For legacy model, or path to your .keras file
)  # <-------

# Decision threshold for the model's positive class
decision_threshold: float = (
    0.8996943831443787  # <------- set your chosen threshold here
)

# Batch size (number of files to run in one go)
batch_size: int = 100

# Target sample-rate passed on to the child script (used for loading/processing)
sample_rate: int = 8000

# -------------------------------------------------------------------------
# MASTER SCRIPT — no edits required below this line
# -------------------------------------------------------------------------

# Ensure the output directory exists
output_folder.mkdir(parents=True, exist_ok=True)

# Create the initial results CSV (now includes Probability and Margin)
results_table = pd.DataFrame(columns=["File", "Timestamp", "Probability", "Margin"])
results_table.to_csv(output_folder / "temporary_results_table.csv", index=False)

# Find all WAV files (case-insensitive)
new_audio_files = [
    f.name for f in new_audio_dir.iterdir() if f.suffix.lower() == ".wav"
]

if not new_audio_files:
    print(
        "\n    No files found — please check that ‘new_audio_dir’ is correct and contains .wav files."
    )
    exit()

# Track batches and files for numbering output
batch_counter = 0
file_counter = 0

total_num_batches = (
    len(new_audio_files) // batch_size
)  # integer division replicates previous logic

# Iterate over audio files in batches
for i in range(0, len(new_audio_files), batch_size):
    batch_files = new_audio_files[i : i + batch_size]

    # Write current batch to a temporary file
    current_batch_path = Path("current_batch.txt")
    current_batch_path.write_text("\n".join(batch_files))

    # Launch the child script
    subprocess.run(
        [
            "python",
            "inference_child.py",
            str(current_batch_path),  # sys1
            str(new_audio_dir),  # sys2
            str(output_folder),  # sys3
            str(model_dir),  # sys4
            str(sample_rate),  # sys5
            str(batch_counter),  # sys6
            str(total_num_batches),  # sys7
            str(file_counter),  # sys8
            str(decision_threshold),  # sys9 (NEW)
        ],
        check=True,
    )

    # Reload the results CSV to get the updated number of rows
    df = pd.read_csv(output_folder / "temporary_results_table.csv")
    num_rows = df.shape[0]

    # Update counters
    file_counter = str(num_rows)
    batch_counter += 1

    # Remove temporary batch file
    current_batch_path.unlink()

# -------------------------------------------------------------------------
# Finalise results table (rename with timestamp)
# -------------------------------------------------------------------------

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
current_path = output_folder / "temporary_results_table.csv"
new_path = output_folder / f"final_results_table_{timestamp}.csv"
current_path.rename(new_path)

# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------

print("   Bomb detection complete on all files!\n")
print(
    f"Found {num_rows} suspected bombs"
)  # Note: skipped files due to corruption are not counted here
print(
    "The suspected bomb files and a results spreadsheet have been written to: "
    f"{output_folder}.\nYou can now inspect these in Audacity."
)
