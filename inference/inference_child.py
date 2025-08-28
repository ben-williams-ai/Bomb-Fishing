# child_script.py
import os
import sys
import datetime
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
from autokeras.keras_layers import CastToFloat32
from tqdm.auto import tqdm


def main():
    print("\n     Starting a new batch!\n")

    # Get the filename and runtime parameters from the command-line arguments
    current_batch = Path(sys.argv[1])
    new_audio_dir = Path(sys.argv[2])
    output_folder = Path(sys.argv[3])
    model_dir = Path(sys.argv[4])
    sample_rate = int(sys.argv[5])
    batch_counter = int(sys.argv[6]) + 1
    total_num_batches = int(sys.argv[7])
    file_counter = int(sys.argv[8])  # will be incremented below

    # Decision threshold: prefer CLI arg; fallback to env var; else 0.5
    if len(sys.argv) > 9:
        decision_threshold = float(sys.argv[9])
    else:
        decision_threshold = float(os.environ.get("DECISION_THRESHOLD", "0.5"))
    print(f"Using decision threshold: {decision_threshold:.4f}")

    # Load the TensorFlow model
    best_model = tf.keras.models.load_model(
        model_dir,
        custom_objects={"CastToFloat32": CastToFloat32},
        compile=False,
    )

    # Recording progress
    skipped_files = []
    suspected_bombs = []  # tuples: (File, Timestamp, Probability, Margin)
    checked_file_counter = 0

    # Processing parameters
    new_sample_rate = 8000  # internal processing sample rate
    window_length_samples = int(2.88 * new_sample_rate)
    stagger = window_length_samples // 2  # 1.44 s

    # Read current batch from file
    with current_batch.open("r") as f:
        current_batch_files = [line.strip() for line in f]

    # The main loop
    for file_name in tqdm(current_batch_files):
        try:
            checked_file_counter += 1
            print(
                f"\nBatch {batch_counter} of {total_num_batches}. Checking file: {file_name}"
            )

            # Load the audio (resample to sample_rate during loading)
            audio_path = new_audio_dir / file_name
            audio, sr = librosa.load(path=str(audio_path), sr=sample_rate)

            # Resample again if needed to our internal rate
            if sr != new_sample_rate:
                audio = librosa.resample(audio, sr, new_sample_rate)

            # Compute the number of windows
            num_windows = len(audio) // window_length_samples

            # For skipping the first stream if needed later in next for-loop
            timestamp2 = None

            # Iterate over windows and compute MFCC for each window
            for i in range(num_windows):
                # Stream 1: starts at 0 s; Stream 2: starts staggered at 1.44 s
                start_index1 = i * window_length_samples
                start_index2 = start_index1 + stagger

                end_index1 = start_index1 + window_length_samples
                end_index2 = start_index2 + window_length_samples

                window1 = audio[start_index1:end_index1]
                window2 = audio[start_index2:end_index2]

                # MFCCs (32 coeffs)
                mfcc_spec1 = librosa.feature.mfcc(
                    y=window1, sr=new_sample_rate, n_mfcc=32
                )
                mfcc_spec2 = librosa.feature.mfcc(
                    y=window2, sr=new_sample_rate, n_mfcc=32
                )

                # Add extra dims for model input shape: (1, n_mfcc, time, channels)
                mfcc_spec1 = np.expand_dims(np.expand_dims(mfcc_spec1, axis=2), axis=0)
                mfcc_spec2 = np.expand_dims(np.expand_dims(mfcc_spec2, axis=2), axis=0)

                # Predict raw probabilities, then threshold
                p1 = float(best_model.predict(mfcc_spec1, verbose=0).ravel()[0])
                result1 = p1 >= decision_threshold

                p2 = None
                result2 = None
                if len(window2) == window_length_samples:
                    p2 = float(best_model.predict(mfcc_spec2, verbose=0).ravel()[0])
                    result2 = p2 >= decision_threshold

                # Time in seconds for "previous" stream2 to avoid double-counting
                check = round((start_index1 / new_sample_rate) - 1.44, 2)

                # If stream1 is positive and not already caught by preceding stream2
                if result1 and timestamp2 != check:
                    timestamp1 = start_index1 / new_sample_rate
                    timestamp_str1 = str(datetime.timedelta(seconds=timestamp1))
                    hmmss = timestamp_str1[0:7]  # HH:MM:SS
                    print(f"Suspected bomb at: {hmmss} (p={p1:.3f})")

                    # Select the 5 s window to save
                    write_start = max(start_index1 - new_sample_rate, 4000)
                    write_end = max(start_index1 + (3 * new_sample_rate), 44000)
                    audio_write = audio[write_start:write_end]

                    # Write new WAV
                    file_counter += 1
                    wav_name = f"{file_counter:07d}_{hmmss.replace(':', '.')}_{file_name[:-4]}.wav"
                    sf.write(output_folder / wav_name, audio_write, new_sample_rate)

                    # Record in results (File, Timestamp, Probability, Margin)
                    suspected_bombs.append(
                        (file_name, timestamp_str1, p1, p1 - decision_threshold)
                    )

                # If stream1 not positive but stream2 is
                elif result2:
                    timestamp2 = start_index2 / new_sample_rate
                    timestamp_str2 = str(datetime.timedelta(seconds=timestamp2))
                    hmmss = timestamp_str2[0:7]
                    print(f"Suspected bomb at: {hmmss} (p={p2:.3f})")

                    write_start = max(start_index2 - new_sample_rate, 4000)
                    write_end = max(start_index2 + (3 * new_sample_rate), 44000)
                    audio_write = audio[write_start:write_end]

                    file_counter += 1
                    wav_name = f"{file_counter:07d}_{hmmss.replace(':', '.')}_{file_name[:-4]}.wav"
                    sf.write(output_folder / wav_name, audio_write, new_sample_rate)

                    suspected_bombs.append(
                        (file_name, timestamp_str2, p2, p2 - decision_threshold)
                    )

        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
            skipped_files.append(file_name)
            # Keep CSV shape consistent; leave Probability/Margin empty for skipped files
            suspected_bombs.append(
                (file_name, "file was skipped, check if corrupted", None, None)
            )
            continue

    # Update the CSV
    csv_path = output_folder / "temporary_results_table.csv"
    existing_table = pd.read_csv(csv_path)

    new_data = pd.DataFrame(
        suspected_bombs,
        columns=["File", "Timestamp", "Probability", "Margin"],
    )

    combined_table = pd.concat([existing_table, new_data], ignore_index=True)
    combined_table.index = np.arange(1, len(combined_table) + 1)
    combined_table.to_csv(csv_path, index=False)

    # Print findings
    print(
        f"\nBatch {batch_counter} of {total_num_batches} finished. Checked {checked_file_counter} files in this batch"
    )
    print(
        f"Found {len([r for r in suspected_bombs if isinstance(r[2], (int, float))])} suspected bombs"
    )
    print(f"Skipped {len(skipped_files)} files due to corruption:")
    for file in skipped_files:
        print(f"  {file}")
    print("")


if __name__ == "__main__":
    main()
