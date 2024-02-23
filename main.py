import numpy as np
import math
import torch
import gc 
import time
import os
import random
from dotenv import load_dotenv
import argparse

import speechmetrics #https://github.com/aliutkus/speechmetrics
import whisperx
from pydub import AudioSegment
#import chromadb
from pyannote.audio import Model
from scipy.spatial.distance import cdist
from pyannote.core import Segment
from pyannote.audio import Inference
from unsilence import Unsilence

#https://github.com/ina-foss/inaSpeechSegmenter # sudo apt-get install ffmpeg
from inaSpeechSegmenter import Segmenter
from inaSpeechSegmenter.export_funcs import seg2csv, seg2textgrid
from df.enhance import enhance, init_df, load_audio, save_audio

#TODO
import torchaudio
from resemble_enhance.enhancer.inference import denoise, enhance as resemble_enhancer
from scipy.io.wavfile import write
from scipy.spatial.distance import cdist

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
        file_name = os.path.join(self.wavs_directory, ts + str(self.generateRandomNumber(24)) + ".wav")

        print("file_name")
        print(file_name)

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

    def write_data_to_csv(self, text, file_name, speaker_name, gender, duration):
        newData = {
            'text': text,
            'audio_file': file_name,
            'speaker_name': speaker_name,
            'gender': gender,
            'duration': duration
        }

        with open(self.csv_file_name, 'a', encoding='utf-8') as csvFile:
            csvFile.write(str(newData) +"\n")

    def generateRandomNumber(self, digits):
        finalNumber = ""
        for i in range(digits // 16):
            finalNumber = finalNumber + str(math.floor(random.random() * 10000000000000000))
        finalNumber = finalNumber + str(math.floor(random.random() * (10 ** (digits % 16))))
        return int(finalNumber)

    #TODO Search existing speakers in the database with speaker embeddings + pyannote.
    def get_speaker_name(self, collection, audio_embeddings_infencer, file_name, speakers_list):

        current_speaker_embedding = audio_embeddings_infencer(file_name)

        if not speakers_list:
            speaker_name = self.generateRandomNumber(24)
        
            speakers_list.append({
                'speaker_embeddings': current_speaker_embedding,
                'speaker_name': speaker_name
            })
            
            return
        
        for existing_speaker in speakers_list:
            
            distance = cdist([current_speaker_embedding], [existing_speaker['speaker_embeddings']], metric="cosine")[0,0]

            if distance < 0.15:
                return existing_speaker['speaker_name']

        speaker_name = self.generateRandomNumber(24)
        
        speakers_list.append({
            'speaker_embeddings': current_speaker_embedding,
            'speaker_name': speaker_name
        })
        
        return speaker_name
    
        #TODO get speaker names from chroma database. Search by embeddings. Do this inside get_speaker_name
        current_speaker_embedding = audio_embeddings_infencer(file_name)

        #search similar embedding in the database or create a new speaker
        
        #pending to filter by embeddings distance
        # do nearest neighbor search to find similar embeddings or documents, supports filtering
        results = collection.query(
            query_embeddings=[current_speaker_embedding],
            n_results=1,
            #where={"cosine": "gt1"},
        )
                
        if results:
            speaker_name = results.ids[0][0]
        else:
            speaker_name = self.generateRandomNumber(24)
            
            collection.add(
                embeddings=[current_speaker_embedding],
                #metadatas=[{"speaker_name": speaker_name ],
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
        
    def process(self, path_to_audio_file, output_directory, range_start, range_end, filter_types):    
        
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

        print("enhance audio")
        enhanced_audio_file_path = self.audio_manager_instance.process(path_to_audio_file, output_directory, filter_types)
        
        print("enhanced_audio_file_path", enhanced_audio_file_path)
        
        torch.cuda.empty_cache()
        gc.collect()        
        if not enhanced_audio_file_path:
            return
        
        #TODO
        #client = chromadb.Client()
        #collection = client.get_or_create_collection(name="audios")
        #collection.peek() # returns a list of the first 10 items in the collection
        #collection.count() # returns the number of items in the collection
        collection = None
        
        device = get_device()
        batch_size = 16 # reduce if low on GPU mem
        #compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
        compute_type="int8"

        # 1. Transcribe with original whisper (batched)
        model = whisperx.load_model("large-v3", device, compute_type=compute_type)

        audio = whisperx.load_audio(enhanced_audio_file_path)
        result = model.transcribe(audio, batch_size=batch_size)
        
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        
        del model; gc.collect(); torch.cuda.empty_cache()
        
        #TODO
        #diarize_model = whisperx.DiarizationPipeline(model_name='pyannote/speaker-diarization@2.1', use_auth_token=HF_TOKEN, device=device)
        ## TO GET THE NUMBER OF SPEAKERS
        #diarize_segments = diarize_model(audio)
        #print("diarize_segments")
        #print(diarize_segments)
        
        model = Model.from_pretrained("pyannote/embedding", use_auth_token=HF_TOKEN)

        audio_embeddings_infencer = Inference(model, window="whole")
        
        seg = Segmenter()
        
        speakers_list = []

        for segment in result["segments"]:
            start = segment["start"]
            end = segment["end"]

            duration = segment["end"] - segment["start"]
            
            if duration < range_start or duration > range_end:
                print("current duration:", duration)
                print("text:", segment["text"])
                continue
            
            file_name = self.create_audio_segment(start, end, enhanced_audio_file_path)
            
            segmentation = seg(file_name)
            has_music = self.audio_manager_instance.has_music(segmentation)

            if has_music:
                print(f"Audio has music. Discarted from {start} to {end} of {enhanced_audio_file_path}")
                continue
            
            #Verify the quality of the audio here
            has_quality = self.audio_manager_instance.has_speech_quality(file_name)
            
            if not has_quality:
                print(f"Audio does not have enough quality. Discarted from {start} to {end} of {enhanced_audio_file_path}")
                continue
            
            speaker_name = self.get_speaker_name(collection, audio_embeddings_infencer, file_name, speakers_list)
                        
            gender = self.get_gender(segmentation)
            
            self.write_data_to_csv(segment["text"], self.csv_file_name, speaker_name, gender, duration)
    
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

        print("audio_chunks")
        print(audio_chunks)

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
        
        print("chunk_size", chunk_size)

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
        
        #estimated_time = u.estimate_time(audible_speed=5, silent_speed=2)  # Estimate time savings
        #print(estimated_time)

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
            print(f"Average {metric_name} score: {average_score}")
            
            average_scores[metric_name] = average_score
            
        if average_scores['mosnet'] >= 3:
            print("valid audio")
            return True

        del self.metrics
        torch.cuda.empty_cache()
        gc.collect()

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

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='')

    # Define positional arguments
    parser.add_argument('--input_file_path', type=str, help='Path to the input audio file.')
    parser.add_argument('--input_folder', type=str, help='Path to the input folder containing audio files.')
    parser.add_argument('--output_directory', type=str, help='Output directory for audio files.')
    parser.add_argument('--range_times', nargs='?', type=parse_range, default=(4, 10), help='Specify a range of two integers in the format "start-end". Default is 4-10.')
    parser.add_argument('--types', nargs='+', default=["deepfilternet"], help='List of types. Default is deepfilternet. You can combine filters too: --types deepfilternet enhance_audio_resembleai  . You can disable it with --types None')

    args = parser.parse_args()
    
    input_file_path     = args.input_file_path
    input_folder        = args.input_folder
    output_directory    = args.output_directory
    start, end          = args.range_times
    filter_types        = args.types

    if input_folder and input_file_path:
        raise Exception("You must choose either a file or a folder, not both.")
    
    if input_folder:
        all_files = os.listdir(input_folder)
        
        audio_files = [file for file in all_files if file.lower().endswith(('.mp3', '.wav', '.flac', '.ogg', '.aac', '.wma'))]

        for audio_file in audio_files:
            # You can perform operations on each audio file here
            # For example, print the file name
            print("Processing:", audio_file)
            
            # If you want to get the full path to the file, you can use os.path.join
            current_input_file_path = os.path.join(input_folder, audio_file)
            
            dataset_generator().process(current_input_file_path, output_directory, start, end, filter_types)

            # Now, you can use 'current_input_file_path' to load and process the audio file as needed
            # For simplicity, let's just print the full path
            print("Full Path:", current_input_file_path)

    elif input_file_path:

        dataset_generator().process(input_file_path, output_directory, start, end, filter_types)
