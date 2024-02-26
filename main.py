import numpy as np
import math
import torch
import gc 
import time
import os
import random
from dotenv import load_dotenv
import argparse
import subprocess

import speechmetrics #https://github.com/aliutkus/speechmetrics
import whisperx
from pydub import AudioSegment
from pyannote.audio import Model
from scipy.spatial.distance import cdist
from pyannote.core import Segment
from pyannote.audio import Inference
from unsilence import Unsilence

#https://github.com/ina-foss/inaSpeechSegmenter # sudo apt-get install ffmpeg
from inaSpeechSegmenter import Segmenter
from inaSpeechSegmenter.export_funcs import seg2csv, seg2textgrid
from df.enhance import enhance, init_df, load_audio, save_audio

import torchaudio
from resemble_enhance.enhancer.inference import denoise, enhance as resemble_enhancer
from scipy.io.wavfile import write
from scipy.spatial.distance import cdist
import yt_dlp
import copy
import chromadb

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")

#You need to agree to share your contact information to access pyannote embedding model
#https://huggingface.co/pyannote/embedding
#https://huggingface.co/pyannote/speaker-diarization

def get_device():
    if torch.cuda.is_available():
        #device = torch.device("cuda")
        device = "cuda"
        print("CUDA is available. Using GPU.")
    else:
        #device = torch.device("cpu")
        device = "cpu"
        print("CUDA is not available. Using CPU.")
        
    return device
    
class dataset_generator:

    def create_audio_segment(self, start, end, audio_file):
        
        ts = str(int(time.time()))

        #file_name = os.path.join(path_to_store_audio, ts_encoded + str(random.getrandbits(128)) + ".wav")
        file_name = os.path.join(self.wavs_directory, ts + str(self.generate_random_speaker_name(24)) + ".wav")

        t1 = start * 1000
        t2 = end * 1000

        extension = audio_file[-3:]

        if extension == "mp3":
            newAudio = AudioSegment.from_mp3(audio_file)
        elif extension == "m4a":
            newAudio = AudioSegment.from_file(audio_file)
        else:
            newAudio = AudioSegment.from_wav(audio_file)
            
        newAudio = newAudio[t1:t2]

        newAudio = newAudio.set_frame_rate(22050)
        newAudio = newAudio.set_channels(1)
        newAudio.export(file_name, format="wav")

        return file_name

    def write_data_to_csv(self, transcription, file_name, language):
        
        for segment in transcription["segments"]:
            
            newData = {
                'text': segment["text"],
                'audio_file': file_name,
                'speaker': segment["generated_speaker_name"],
                'gender': segment["gender"],
                'duration': segment["duration"],
                'language': language
            }

            with open(self.csv_file_name, 'a', encoding='utf-8') as csvFile:
                csvFile.write(str(newData) +"\n")

    def generate_random_speaker_name(self, digits):
        finalNumber = ""
        for i in range(digits // 16):
            finalNumber = finalNumber + str(math.floor(random.random() * 10000000000000000))
        finalNumber = finalNumber + str(math.floor(random.random() * (10 ** (digits % 16))))
        return f"speaker_{finalNumber}"

    def get_speaker_info(self, collection, audio_embeddings_infencer, file_name):

        current_speaker_embedding = audio_embeddings_infencer(file_name)
        
        # Normalize embeddings
        current_speaker_embedding = (current_speaker_embedding / np.linalg.norm(current_speaker_embedding)).tolist()

        results = collection.query(
            query_embeddings=[current_speaker_embedding],
            n_results=1,
            include=["metadatas", "distances",  "embeddings"]
        )
        
        if not results["distances"][0]:
            
            speaker_name = self.generate_random_speaker_name(24)

            collection.add(
                embeddings=np.array([current_speaker_embedding]),
                metadatas=[{"speaker_name": speaker_name }],
                ids=[speaker_name]
            )
        
            return speaker_name
        
        distance = cdist([current_speaker_embedding], [results["embeddings"][0][0]], metric="cosine")[0,0]

        if distance < 0.15:
            
            speaker_name = results["metadatas"][0][0]["speaker_name"]
            
            return speaker_name
        else:
            
            speaker_name = self.generate_random_speaker_name(24)

            collection.add(
                embeddings=np.array([current_speaker_embedding]),
                metadatas=[{"speaker_name": speaker_name }],
                ids=[speaker_name]
            )
        
            return speaker_name
            
    def get_gender(self, segmentation):

        labels = [item[0] for item in segmentation if item[0] in ('male', 'female')]

        if 'male' in labels:
            return 'male'
        elif 'female' in labels:
            return 'female'
        else:
            return 'no_gender'
        
    def get_transcription(self, enhanced_audio_file_path):
        device = get_device()
        batch_size = 16 # reduce if low on GPU mem
        #compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
        compute_type="int8"

        # 1. Transcribe with original whisper (batched)
        model = whisperx.load_model("large-v3", device, compute_type=compute_type)

        audio = whisperx.load_audio(enhanced_audio_file_path)
        result = model.transcribe(audio, batch_size=batch_size)
        
        language = result["language"]
        
        model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

        diarize_model = whisperx.DiarizationPipeline(model_name='pyannote/speaker-diarization@2.1', use_auth_token=HF_TOKEN, device=device)
        diarize_segments = diarize_model(audio)
                
        result = whisperx.assign_word_speakers(diarize_segments, result)
        
        # this is needed to fix some items that only have word, but don't have start, end, score and speaker.
        for i, segment in enumerate(result["segments"]):
            wordlevel_info = []
            for iw, word in enumerate(segment["words"]):
                
                if "start" not in word:
                    
                    if iw-1 >= 0:  # Check if iw-1 is a valid index
                        word["start"]   = round(segment["words"][iw-1]["end"] + 0.001, 3)
                        word['score']   = segment["words"][iw-1]["score"]
                        word['speaker'] = segment["words"][iw-1]["speaker"]
                    elif i-1 >= 0:
                        # Use the last word of the previous segment
                        word["start"]   = round(result["segments"][i-1]["words"][-1]["end"] + 0.001, 3)
                        word["score"]   = result["segments"][i-1]["words"][-1]["score"]
                        word["speaker"] = result["segments"][i-1]["words"][-1]["speaker"]
                    else:
                        word["start"] = 0.001
                        word["score"] = 1
                        word["speaker"] = segment["speaker"]
                    
                    if iw+1 < len(segment["words"]) and 'start' in segment["words"][iw+1]:  # Check if iw+1 is a valid index
                        word["end"]     = round(segment["words"][iw+1]["start"] - 0.001, 3)
                        word["score"]   = segment["words"][iw+1]["score"]
                        word["speaker"] = segment["words"][iw+1]["speaker"]
                    elif i+1 < len(result["segments"]) and 'start' in result["segments"][i+1]["words"][0]:
                        # Use the first word of the next segment
                        word["end"]     = round(result["segments"][i+1]["words"][0]["start"] - 0.001, 3)
                        word["score"]   = result["segments"][i+1]["words"][0]["score"]
                        word["speaker"]   = result["segments"][i+1]["words"][0]["speaker"]
                    else:
                        word["end"] = 0.001
                        word["score"] = 1
                        word["speaker"] = segment["speaker"]

                #print("current word", word)
                wordlevel_info.append({
                    'word':word["word"],
                    'start':word["start"],
                    'end':word["end"],
                    'speaker':word['speaker'],
                    'score':word['score']
                })
            
            segment["words"] = wordlevel_info 
                
        fixed_segments = []
        for segment in result["segments"]:
            
            current_speaker = None
            current_words = []
            duration = segment["end"] - segment["start"]

            # Iterate over each word in the segment
            for word in segment["words"]:
                speaker = word["speaker"]

                if not current_speaker or current_speaker == speaker:
                    current_words.append(word)
                else:
                    fixed_segments.append({
                        "speaker": current_speaker,
                        "start": current_words[0]["start"] if current_words else None,
                        "end": current_words[-1]["end"] if current_words else None,
                        "text": " ".join(w["word"] for w in current_words),
                        "words": current_words,
                    })

                    # Start a new list for the current speaker
                    current_words = [word]
                
                current_speaker = speaker

            # Save the words for the last speaker in the segment
            if current_speaker is not None:
                fixed_segments.append({
                    "speaker": current_speaker,
                    "start": current_words[0]["start"] if current_words else None,
                    "end": current_words[-1]["end"] if current_words else None,
                    "text": " ".join(w["word"] for w in current_words),
                    "words": current_words,
                })
                    
        result["segments"] = fixed_segments
        
        del model; gc.collect(); torch.cuda.empty_cache()
        del diarize_segments
        
        return result, language
        
    def assign_name_to_each_speaker(self, transcription, collection):
        
        existing_speakers = {}

        for segment in transcription["segments"]:
            
            speaker = segment["speaker"]
            duration = segment["end"] - segment["start"]

            if "speaker" not in existing_speakers or ("speaker" in existing_speakers and duration > existing_speakers[speaker]["end"] - existing_speakers[speaker]["start"]):
                existing_speakers[speaker] = {
                    "speaker": speaker,
                    "audio_file": segment["audio_file"],
                    "start": segment["start"],
                    "end": segment["end"],
                }
                
        model = Model.from_pretrained("pyannote/embedding", use_auth_token=HF_TOKEN)
        audio_embeddings_infencer = Inference(model, window="whole")
        
        #for existing_speaker in existing_speakers:
        for speaker, existing_speaker in existing_speakers.items():
            
            generated_speaker_name = self.get_speaker_info(collection, audio_embeddings_infencer, existing_speaker["audio_file"])
            
            existing_speaker["generated_speaker_name"] = generated_speaker_name
        
        for segment in transcription["segments"]:
            
            segment["generated_speaker_name"] = existing_speaker["generated_speaker_name"]
        
        return transcription
        
    def filter_transcription_segments_and_assign_values(self, transcription, range_start, range_end, enhanced_audio_file_path):
       
        seg = Segmenter()
        valid_segments = []
        for segment in transcription["segments"]:
            start = segment["start"]
            end = segment["end"]
            
            duration = segment["end"] - segment["start"]
            
            if duration < range_start or duration > range_end:
                continue
            
            file_name = self.create_audio_segment(start, end, enhanced_audio_file_path)
            
            segmentation = seg(file_name)
            has_music = self.audio_manager_instance.has_music(segmentation)

            if has_music:
                print(f"Audio has music. Discarted from {start} to {end} of {enhanced_audio_file_path}")
                os.remove(file_name)
                continue
            
            #Verify the quality of the audio here
            has_quality = self.audio_manager_instance.has_speech_quality(file_name)
            
            if not has_quality:
                print(f"Audio does not have enough quality. Discarted from {start} to {end} of {enhanced_audio_file_path}")
                os.remove(file_name)
                continue
            
            gender = self.get_gender(segmentation)

            segment["audio_file"]   = file_name
            segment["gender"]       = gender
            segment["duration"]     = duration
            
            valid_segments.append(segment)
            
        transcription["segments"] = valid_segments
        
        return transcription
    
    def process(self, path_to_audio_file, output_directory, range_start, range_end, filter_types, collection):    
        
        # STEPS        
        # check the audio quality of the whole file
        # transcribe
        # check where is the speech
        # check the quality of each individual audio file 
        # get speakers number of the speakers. Just one can be speaking. Identify each one with chromadb embeddings
        # clustering voices (it is an improvement. TODO in a near future)
        # discard audios that only are music or has too poor quality
        # discard parts of the audio that are music if there is speech too

        self.audio_manager_instance = audio_manager()
        
        self.wavs_directory = os.path.join(output_directory, 'wavs')
        if not os.path.exists(self.wavs_directory):
            os.makedirs(self.wavs_directory)

        self.csv_file_name = os.path.join(output_directory, "dataset.csv") 
        if not os.path.exists(path_to_audio_file):
            raise Exception(f"File {path_to_audio_file} does not exist")

        enhanced_audio_file_path = self.audio_manager_instance.process(path_to_audio_file, output_directory, filter_types)
        
        torch.cuda.empty_cache()
        gc.collect()        
        if not enhanced_audio_file_path:
            return
        
        transcription, language = self.get_transcription(enhanced_audio_file_path)
        
        transcription = self.filter_transcription_segments_and_assign_values(transcription, range_start, range_end, enhanced_audio_file_path)
        
        transcription = self.assign_name_to_each_speaker(transcription, collection)

        self.write_data_to_csv(transcription, self.csv_file_name, language)
        
class audio_manager:
    
    def __init__(self):
        window_length = 30 # seconds
        self.metrics = speechmetrics.load('absolute', window_length)
        
    def process(self, input_audio, output_directory, filter_types):
        
        if "None" not in filter_types:
            output_audio_file = self.get_output_file_name(input_audio, output_directory)
            self.enhance_audio(input_audio, output_audio_file, filter_types)
        else:
            output_audio_file = input_audio
        
        if not self.has_speech_quality(output_audio_file):
            return None
        
        return output_audio_file
    
    def get_output_file_name(self, input_audio, output_directory):
        # Extract the input file name without extension
        file_name_without_extension, extension = os.path.splitext(os.path.basename(input_audio))

        # Create the output file name by adding "_enhanced" suffix
        output_file_name = f"{file_name_without_extension}_enhanced{extension}"
        
        enhanced_directory = os.path.join(output_directory, "enhanced")

        if not os.path.exists(enhanced_directory):
            os.makedirs(enhanced_directory)
            
        return os.path.join(enhanced_directory, output_file_name)

    #https://github.com/Rikorose/DeepFilterNet
    #alternatives to deepfilternet 
    #https://github.com/resemble-ai/resemble-enhance
    #https://github.com/shahules786/mayavoz 
    def enhance_audio(self, noisy_audio, output_audio_file, types=["deepfilternet"]):
            
        temp_output = noisy_audio  # Initial audio loading

        for enhancement_type in types:
            if enhancement_type == "deepfilternet":
                temp_output = self.enhance_audio_deepfilternet(temp_output, output_audio_file)
            elif enhancement_type == "enhance_audio_resembleai":
                temp_output = self.enhance_audio_resembleai(temp_output, output_audio_file)
            elif enhancement_type == "mayavoz":
                temp_output = self.enhance_audio_mayavoz(temp_output, output_audio_file)
        
        self.remove_sliences(temp_output)

    def enhance_audio_deepfilternet(self, noisy_audio, output_audio_file):

        model, df_state, _ = init_df()  # Load default model

        audio, info = load_audio(noisy_audio, sr=df_state.sr())
        
        minutes = 1 # it is easy to edit time in minutes than seconds
        seconds = minutes * 60

        # Split audio into 5min chunks
        audio_chunks = [audio[:, i:i + seconds * info.sample_rate]
                        for i in range(0, audio.shape[1], seconds * info.sample_rate)]

        enhanced_chunks = []
        for ac in audio_chunks:
            enhanced_chunks.append(enhance(model, df_state, ac))
            torch.cuda.empty_cache()
            gc.collect()

        enhanced = torch.cat(enhanced_chunks, dim=1)

        assert enhanced.shape == audio.shape, 'Enhanced audio shape does not match original audio shape.'

        save_audio(output_audio_file, enhanced, sr=df_state.sr())  
        
        del model, df_state, audio, info, enhanced_chunks, enhanced
        torch.cuda.empty_cache()
        gc.collect()
        
        return output_audio_file
       
    def split_audio_into_chunks(self, audio, chunk_size):
        """
        Split audio into chunks of specified size.

        Parameters:
            audio (np.ndarray): The audio data.
            chunk_size (int): Size of each chunk in samples.

        Returns:
            List[np.ndarray]: List of audio chunks.
        """
        num_samples = len(audio)
        num_chunks = num_samples // chunk_size
        chunks = [audio[i * chunk_size: (i + 1) * chunk_size] for i in range(num_chunks)]
        return chunks
    
    def enhance_audio_resembleai(self, noisy_audio, output_audio_file):
        
        solver = "midpoint" #rk4, euler, midpoint
        denoising = True
        nfe = 128
        tau = 0
        chunk_duration = 20  # 1 minute in seconds

        if noisy_audio is None:
            return None, None

        solver = solver.lower()
        nfe = int(nfe)
        lambd = 0.9 if denoising else 0.1

        # Load the entire audio file
        dwav, sr = torchaudio.load(noisy_audio)
        dwav = dwav.mean(dim=0)

        device = get_device()
        
        # Calculate chunk size based on duration
        chunk_size = int(sr * chunk_duration)
        
        # Split the audio into chunks
        audio_chunks = self.split_audio_into_chunks(dwav.cpu().numpy(), chunk_size)

        enhanced_chunks = []

        for chunk in audio_chunks:
            
            chunk_tensor = torch.tensor(chunk)

            # Apply enhancement to each chunk
            wav2_chunk, new_sr = resemble_enhancer(chunk_tensor, sr, device, nfe=nfe, solver=solver, lambd=lambd, tau=tau)
            
            # Save the enhanced chunk to the list
            enhanced_chunks.append(wav2_chunk)

        # Concatenate all enhanced chunks
        enhanced_audio = np.concatenate(enhanced_chunks)

        # Write the concatenated enhanced audio to the output file
        write(output_audio_file, new_sr, enhanced_audio)

        return output_audio_file
        
    def enhance_audio_mayavoz(self, noisy_audio, output_audio_file):
        #TODO
        return output_audio_file
        
    # https://github.com/lagmoellertim/unsilence
    def remove_sliences(self, path_to_audio_file):
        
        u = Unsilence(path_to_audio_file)
        
        u.detect_silence()
        
        #rewrite the file with no silences
        u.render_media(path_to_audio_file, audio_only=True)  # Audio only specified
        
        del u
        
    #https://github.com/aliutkus/speechmetrics
    def has_speech_quality(self, path_to_audio_file):

        scores = self.metrics(path_to_audio_file)

        average_scores = {}

        for metric_name, scores_array in scores.items():
            # Calculate the average of the array/list
            average_score = np.mean(scores_array)

            # Print the result
            #print(f"Average {metric_name} score: {average_score}")
            
            average_scores[metric_name] = average_score
            
        if average_scores['mosnet'] >= 3:
            return True

        return False
    
    def has_music(self, segmentation):

        labels = [item[0] for item in segmentation if item[0] in ('music')]

        if 'music' in labels:
            return True

        return False

def parse_range(value):
    try:
        start, end = map(int, value.split('-'))
        return start, end
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid range format. Please use 'start-end'.")

def get_local_audio_files(input_folder):
    all_files = os.listdir(input_folder)
    return [os.path.join(input_folder, file) for file in all_files if file.lower().endswith(('.mp3', '.wav', '.flac', '.ogg', '.aac', '.wma'))]

def get_youtube_audio_files(urls, output_directory):
    
    downloaded_files = []
    if not urls:
        return downloaded_files
    
    youtube_files_output_directory = os.path.join(output_directory, "youtube_files") 
    
    if not os.path.exists(youtube_files_output_directory):
        os.makedirs(youtube_files_output_directory)
        
    audio_format = "wav"
    
    for url in urls:
        output_template = os.path.join(youtube_files_output_directory, f"%(title)s.{audio_format}")
            
        ydl_opts = {
            'format': 'bestaudio/best',
            'extractaudio': True,
            'audioformat': 'wav',
            'outtmpl': output_template,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            
    downloaded_files = [os.path.join(youtube_files_output_directory, file_name) for file_name in os.listdir(youtube_files_output_directory)]

    return downloaded_files

def process_audio_files(audio_files, output_directory, start, end, filter_types):
    
    client = chromadb.PersistentClient(path=os.path.join(output_directory, "chroma_database"))
    collection = client.get_or_create_collection(name="speakers")

    for audio_file in audio_files:
        print("Processing:", audio_file)
        dataset_generator().process(audio_file, output_directory, start, end, filter_types, collection)
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='')

    # Define positional arguments
    #parser.add_argument('--input_file_path', type=str, help='Path to the input audio file.')
    #parser.add_argument('--input_folder', type=str, help='Path to the input folder containing audio files.')
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--input_file_path', type=str, help='Path to the input audio file.')
    group.add_argument('--input_folder', type=str, help='Path to the input folder containing audio files.')

    parser.add_argument("--youtube_download", nargs="*", help="YouTube playlist or video URLs")

    parser.add_argument("--output_directory", type=str, help="Output directory for audio files", default=".")
    parser.add_argument('--range_times', nargs='?', type=parse_range, default=(4, 10), help='Specify a range of two integers in the format "start-end". Default is 4-10.')
    parser.add_argument('--types', nargs='+', default=["deepfilternet"], help='List of types. Default is deepfilternet. You can combine filters too: --types deepfilternet enhance_audio_resembleai  . You can disable it with --types None')

    args = parser.parse_args()
    
    input_file_path     = args.input_file_path
    input_folder        = args.input_folder
    youtube_download    = args.youtube_download
    output_directory    = args.output_directory
    start, end          = args.range_times
    filter_types        = args.types

    youtube_audio_files = get_youtube_audio_files(youtube_download, output_directory)

    if input_folder:
        
        local_audio_files = get_local_audio_files(input_folder)
        audio_files = local_audio_files + youtube_audio_files
        process_audio_files(audio_files, output_directory, start, end, filter_types)
        
    elif input_file_path:
        
        audio_files = [input_file_path] + youtube_audio_files
        process_audio_files(audio_files, output_directory, start, end, filter_types)
    else:
        
        process_audio_files(youtube_audio_files, output_directory, start, end, filter_types)
