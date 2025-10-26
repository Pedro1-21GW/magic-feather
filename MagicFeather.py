# This is a personal project script to transcribe DnD sessions (recorded in separated FLAC tracks) using Whisper and Silero VAD.
# It processes multiple audio files, applies VAD to segment speech, transcribes using Whisper and then concatenates logs into a final transcript.

import os
import re
import gc
import numpy as np
import torch
import whisper
from pydub import AudioSegment
from datetime import timedelta


# Set your file paths here 
AUDIO_FOLDER = "dnd_audio" 
OUTPUT_LOG_FOLDER = "transcription_player_logs"
LOG_FOLDER = "dnd_audio/transcription_player_logs"
FINAL_TRANSCRIPT_PATH = "final_session_transcript.txt"

# --- Whisper Model Configuration ---
WHISPER_MODEL = "medium"

# --- Configuration for splitting large files (Dm's file was too large...) ---
# Put the filename of the large audio file here (without the .flac extension)
LARGE_FILE_PLAYER_NAME = "LARGE_FILE_PLAYER_NAME_HERE" 
# Set the chunk number where you want to split the file
CHUNK_SPLIT_LIMIT = 1200 

def format_timestamp(seconds):
    """Formats seconds into HH:MM:SS,ms format."""
    delta = timedelta(seconds=seconds)
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = delta.microseconds // 1000
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def timestamp_to_seconds(ts_str):
    """Converts HH:MM:SS,ms string back to total seconds."""
    try:
        h, m, s_ms = ts_str.split(':')
        s, ms = s_ms.split(',')
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
    except (ValueError, AttributeError):
        return 0.0

def merge_logs_from_folder():
    """
    Finds all '_log.txt' files in the specified folder, reads them,
    and merges them into a single, chronologically sorted file.
    """
    print(f"--- Scanning for log files in '{LOG_FOLDER}'... ---")
    
    if not os.path.exists(LOG_FOLDER):
        print(f"Error: Log folder not found at '{LOG_FOLDER}'")
        return

    # Find all files in the directory that end with '_log.txt'
    log_files = [f for f in os.listdir(LOG_FOLDER) if f.endswith("_log.txt")]

    if not log_files:
        print(f"No log files found in '{LOG_FOLDER}'. Nothing to merge.")
        return

    print(f"Found {len(log_files)} log files. Proceeding with merge.")
    all_transcripts = []
    
    for log_filename in log_files:
        log_file_path = os.path.join(LOG_FOLDER, log_filename)
        print(f"Reading {log_filename}...")
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Use regex to parse the line format "[TIMESTAMP] speaker: text"
                match = re.match(r'\[(.*?)\] (.*?): (.*)', line)
                if match:
                    timestamp_str, speaker, text = match.groups()
                    all_transcripts.append({
                        'speaker': speaker,
                        'start': timestamp_to_seconds(timestamp_str),
                        'text': text.strip()
                    })

    # Sort the combined list chronologically by the start time
    all_transcripts.sort(key=lambda x: x['start'])

    # Write the final, sorted transcript
    with open(FINAL_TRANSCRIPT_PATH, 'w', encoding='utf-8') as f:
        for entry in all_transcripts:
            start_time_str = format_timestamp(entry['start'])
            f.write(f"[{start_time_str}] {entry['speaker']}: {entry['text']}\n")
            
    print(f"\nSuccess! Final merged transcript saved to '{FINAL_TRANSCRIPT_PATH}'")


# ==============================================================================
def run_transcription():
    print("--- Starting Full Transcription Process ---")

    # --- Model Loading ---
    print(f"Loading Whisper model: '{WHISPER_MODEL}'...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: GPU not found. Process will be very slow specially with whisper >= than medium size.")
    whisper_model = whisper.load_model(WHISPER_MODEL, device=device)
    print(f"Whisper model loaded on '{device}'.")

    print("Loading Silero VAD model...")
    vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
    (get_speech_timestamps, _, _, _, _) = utils
    print("Silero VAD model loaded.")

    # --- File Processing for FLAC audio extension---
    os.makedirs(OUTPUT_LOG_FOLDER, exist_ok=True)
    audio_files = sorted([f for f in os.listdir(AUDIO_FOLDER) if f.endswith('.flac')])
    all_transcripts = []

    print(f"\nFound {len(audio_files)} audio files. Beginning processing...")

    for filename in audio_files:
        player_name = os.path.splitext(filename)[0]
        filepath = os.path.join(AUDIO_FOLDER, filename)
        
        #  Checkpointing logic now handles multipart files (specially the Dm's file that was too large)
        is_large_file = (player_name == LARGE_FILE_PLAYER_NAME)
        num_parts = 2 if is_large_file else 1
        parts_to_process = []

        for part in range(1, num_parts + 1):
            part_suffix = f"_part_{part}" if is_large_file else ""
            log_filename = os.path.join(OUTPUT_LOG_FOLDER, f"{player_name}{part_suffix}_log.txt")

            if os.path.exists(log_filename):
                print(f"\n--- Log file found for {player_name}{part_suffix}. Loading from log. ---")
                with open(log_filename, 'r', encoding='utf-8') as f:
                    for line in f:
                        match = re.match(r'\[(.*?)\] (.*?): (.*)', line)
                        if match:
                            timestamp_str, speaker, text = match.groups()
                            all_transcripts.append({
                                'speaker': speaker,
                                'start': timestamp_to_seconds(timestamp_str),
                                'text': text.strip()
                            })
            else:
                parts_to_process.append(part)
        
        if not parts_to_process:
            continue # Skip to next player if all parts are done

        # --- If logs not found, start transcription ---
        print(f"\n--- Processing for player: {player_name} ---")
        
        with torch.no_grad():
            try:
                audio = AudioSegment.from_file(filepath)
                audio_for_vad = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2) # This guarantee the audio is on mono and also with 16 bits depth
                audio_tensor = torch.from_numpy(np.array(audio_for_vad.get_array_of_samples())).float() / 32768.0 # Standard Normalization for 16 bits audio

                print(f"[{player_name}] Applying VAD...")
                # Very important to obtain the correct timestamps and order every concatenated speech transcription
                speech_timestamps = get_speech_timestamps(audio_tensor, vad_model, sampling_rate=16000)
                del audio_tensor
                total_chunks = len(speech_timestamps)
                print(f"[{player_name}] Found {total_chunks} speech segments.")
                
                # Determine which chunks to process for each part
                chunks_part1 = speech_timestamps[:CHUNK_SPLIT_LIMIT] if is_large_file else speech_timestamps
                chunks_part2 = speech_timestamps[CHUNK_SPLIT_LIMIT:] if is_large_file else []

                for part_num, chunks_to_process in [(1, chunks_part1), (2, chunks_part2)]:
                    if part_num not in parts_to_process or not chunks_to_process:
                        continue
                        
                    part_suffix = f"_part_{part_num}" if is_large_file else ""
                    log_filename = os.path.join(OUTPUT_LOG_FOLDER, f"{player_name}{part_suffix}_log.txt")
                    player_transcripts = []
                    
                    print(f"--- Transcribing {player_name}{part_suffix} ({len(chunks_to_process)} chunks) ---")

                    for i, ts in enumerate(chunks_to_process):
                        start_ms, end_ms = ts['start'] / 16, ts['end'] / 16
                        print(f"[{player_name}{part_suffix}] Transcribing chunk {i+1}/{len(chunks_to_process)}...")
                        
                        speech_chunk = audio[start_ms:end_ms]
                        temp_chunk_path = "temp_chunk.wav"
                        speech_chunk.export(temp_chunk_path, format="wav")

                        result = whisper_model.transcribe(temp_chunk_path, language="pt", fp16=torch.cuda.is_available())

                        for segment in result['segments']:
                            corrected_start = start_ms / 1000.0 + segment['start']
                            player_transcripts.append({
                                'speaker': player_name, 
                                'start': corrected_start,
                                'text': segment['text'].strip()
                            })
                        os.remove(temp_chunk_path)
                    
                    if player_transcripts:
                        print(f"[{player_name}{part_suffix}] Transcription complete. Saving individual log...")
                        player_transcripts.sort(key=lambda x: x['start'])
                        
                        with open(log_filename, 'w', encoding='utf-8') as log_file:
                            for entry in player_transcripts:
                                start_time = format_timestamp(entry['start'])
                                log_file.write(f"[{start_time}] {entry['speaker']}: {entry['text']}\n")
                        
                        print(f"✅ Log for {player_name}{part_suffix} saved to '{log_filename}'")
                        all_transcripts.extend(player_transcripts)

            except Exception as e:
                print(f"An error occurred while processing {filename}: {e}")
            
            finally:
                if 'audio' in locals(): del audio
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- Final Concatenation ---
    print("\n------------------------------------------------------")
    print("--- All players processed. Creating final merged transcript... ---")

    if all_transcripts:
        all_transcripts.sort(key=lambda x: x['start'])
        with open(FINAL_TRANSCRIPT_PATH, 'w', encoding='utf-8') as f:
            for entry in all_transcripts:
                start_time_str = format_timestamp(entry['start'])
                f.write(f"[{start_time_str}] {entry['speaker']}: {entry['text']}\n")
        print(f"Final merged transcript saved to '{FINAL_TRANSCRIPT_PATH}'")
    else:
        print("No transcripts were generated or found. Final file not created.")

if __name__ == "__main__":        
    run_transcription()
    #merge_logs_from_folder()