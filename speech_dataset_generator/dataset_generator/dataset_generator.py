import numpy as np
import math
import torch
import gc 
import time
import os
import random
from dotenv import load_dotenv

import whisperx
from pydub import AudioSegment
from pyannote.audio import Model
from scipy.spatial.distance import cdist
from pyannote.audio import Inference

#https://github.com/ina-foss/inaSpeechSegmenter # sudo apt-get install ffmpeg
from inaSpeechSegmenter import Segmenter

from scipy.spatial.distance import cdist

from speech_dataset_generator.audio_manager.audio_manager import AudioManager
from speech_dataset_generator.utils.utils import get_device
from speech_dataset_generator.speech_rate.speech_rate import SpeechRate
import shutil
import csv

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")

class DatasetGenerator:

    def create_audio_segment(self, start, end, audio_file, wavs_directory):
        
        ts = str(int(time.time()))

        #file_name = os.path.join(path_to_store_audio, ts_encoded + str(random.getrandbits(128)) + ".wav")
        file_name = os.path.join(wavs_directory, ts + "_" + self.generate_random_number_as_string(24) + ".wav")

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

    def write_main_data_to_csv(self, transcription, csv_file_name, language):
        
        header = ['text', 'audio_file', 'speaker_id', 'gender', 'duration', 'language', 'syllables_per_minute', 'words_per_minute']
        with open(csv_file_name, 'w', encoding='utf-8', newline='') as csvFile:
            csv_writer = csv.DictWriter(csvFile, fieldnames=header)
            csv_writer.writeheader()
            
        for segment in transcription["segments"]:
            newData = {
                'text': segment["text"],
                'audio_file': segment["audio_file"],
                'speaker_id': segment["generated_speaker_name"],
                'gender': segment["gender"],
                'duration': segment["duration"],
                'language': language,
                'syllables_per_minute': segment["syllables_per_minute"],
                'words_per_minute': segment["words_per_minute"],
            }

            with open(csv_file_name, 'a', encoding='utf-8', newline='') as csvFile:
                
                csv_writer = csv.DictWriter(csvFile, fieldnames=header)

                csv_writer.writerow(newData)

    def write_data_to_csv_ljspeech(self, transcription, csv_file_name):
        
        with open(csv_file_name, 'w', encoding='utf-8', newline='') as csvFile:

            csv_writer = csv.writer(csvFile, delimiter='|')
            
            for segment in transcription["segments"]:
                
                audio_file_name, _ = os.path.splitext(os.path.basename(segment["audio_file"]))

                text = segment["text"]

                csv_writer.writerow([audio_file_name, text, text])

    def generate_random_number_as_string(self, digits):
        finalNumber = ""
        for i in range(digits // 16):
            finalNumber = finalNumber + str(math.floor(random.random() * 10000000000000000))
        finalNumber = finalNumber + str(math.floor(random.random() * (10 ** (digits % 16))))
        return str(finalNumber)

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
            
            speaker_name = self.generate_random_number_as_string(24)

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
            
            speaker_name = self.generate_random_number_as_string(24)

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
        batch_size = 8 # reduce if low on GPU mem
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
                
                if any(key not in word for key in ["start", "end", "speaker"]):
                    
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
                
                if "speaker" not in word:
                    word["speaker"] = segment["speaker"]

                wordlevel_info.append({
                    'word':     word["word"],
                    'start':    word["start"],
                    'end':      word["end"],
                    'speaker':  word['speaker'],
                    'score':    word['score']
                })
            
            segment["words"] = wordlevel_info 
                
        fixed_segments = []
        for segment in result["segments"]:
            
            current_speaker = None
            current_words = []

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
    
    def add_wpm_spm_to_each_segment(self, transcription, language):
        
        speech_rate_instance = SpeechRate()
        for segment in transcription["segments"]:
            
            duration = segment['duration']
            
            word_list = [word_info['word'] for word_info in segment['words']]
            
            syllables_per_minute = speech_rate_instance.get_syllables_per_minute(word_list, language, duration)
            words_per_minute     = speech_rate_instance.get_words_per_minute(word_list, duration)

            segment['syllables_per_minute'] = syllables_per_minute
            segment['words_per_minute'] = words_per_minute
            
        return transcription
        
    def get_existing_speakers(self, transcription, collection):
        
        existing_speakers = {}

        for segment in transcription["segments"]:
            
            speaker = segment["speaker"]
            duration = segment["end"] - segment["start"]

            if "speaker" not in existing_speakers or ("speaker" in existing_speakers and duration > existing_speakers[speaker]["end"] - existing_speakers[speaker]["start"]):
                existing_speakers[speaker] = {
                    "speaker":      speaker,
                    "audio_file":   segment["audio_file"],
                    "start":        segment["start"],
                    "end":          segment["end"],
                    "gender":       segment["gender"]
                }
                
        model = Model.from_pretrained("pyannote/embedding", use_auth_token=HF_TOKEN)
        audio_embeddings_infencer = Inference(model, window="whole")
        
        #for existing_speaker in existing_speakers:
        for speaker, existing_speaker in existing_speakers.items():
            
            generated_speaker_name = self.get_speaker_info(collection, audio_embeddings_infencer, existing_speaker["audio_file"])
            
            existing_speaker["generated_speaker_name"] = generated_speaker_name
        
        return existing_speakers
    
    def assign_name_to_each_speaker(self, transcription, existing_speakers):
        
        for segment in transcription["segments"]:
            
            segment["generated_speaker_name"] = existing_speakers[segment["speaker"]]["generated_speaker_name"]
        
        return transcription

    # This method adds new values such as audio_file, gender and duration  to each segment
    def filter_transcription_segments_and_assign_values(self, transcription, range_start, range_end, enhanced_audio_file_path, wavs_directory):
       
        seg = Segmenter()
        valid_segments = []
        for segment in transcription["segments"]:
            start = segment["start"]
            end = segment["end"]
            
            duration = segment["end"] - segment["start"]
            
            if duration < range_start or duration > range_end:
                print(f"Audio duration greater than range. Range {start} to {end}. Duration {duration}. Audio file: {enhanced_audio_file_path}")
                continue
            
            file_name = self.create_audio_segment(start, end, enhanced_audio_file_path, wavs_directory)
            
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
            segment["duration"]     = round(duration, 3)
            
            valid_segments.append(segment)
            
        transcription["segments"] = valid_segments
        
        return transcription
    
    def process(self, path_to_audio_file, output_directory, range_start, range_end, enhancers, collection, datasets):    
        
        # STEPS        
        # check the audio quality of the whole file
        # transcribe
        # check where is the speech
        # check the quality of each individual audio file 
        # get speakers number of the speakers. Just one can be speaking. Identify each one with chromadb embeddings
        # clustering voices (it is an improvement. TODO in a near future)
        # discard audios that only are music or has too poor quality
        # discard parts of the audio that are music if there is speech too

        ljspeech_directory = os.path.join(output_directory, "ljspeech")

        self.audio_manager_instance = AudioManager()
        
        wavs_directory = os.path.join(ljspeech_directory, 'wavs')
        if not os.path.exists(wavs_directory):
            os.makedirs(wavs_directory, exist_ok=True)

        if not os.path.exists(path_to_audio_file):
            raise Exception(f"File {path_to_audio_file} does not exist")

        enhanced_audio_file_path = self.audio_manager_instance.process(path_to_audio_file, output_directory, enhancers)
        
        if not enhanced_audio_file_path:
            return
        
        transcription, language = self.get_transcription(enhanced_audio_file_path)
        
        transcription = self.filter_transcription_segments_and_assign_values(transcription, range_start, range_end, enhanced_audio_file_path, wavs_directory)

        # words_per_minute and syllables_per_minute
        transcription = self.add_wpm_spm_to_each_segment(transcription, language)

        existing_speakers = self.get_existing_speakers(transcription, collection)
        
        transcription = self.assign_name_to_each_speaker(transcription, existing_speakers)

        csv_file_name = os.path.join(output_directory, "main_data.csv") 
        self.write_main_data_to_csv(transcription, csv_file_name, language)

        csv_file_name = os.path.join(ljspeech_directory, "metadata.csv") 
        self.write_data_to_csv_ljspeech(transcription, csv_file_name)

        self.iterate_datasets(datasets, transcription, output_directory, path_to_audio_file, existing_speakers)
        
    def iterate_datasets(self, datasets, transcription, output_directory, path_to_audio_file, existing_speakers):
        
        for dataset in datasets:
            # Dynamically call the function based on the dataset name
            function_name = f"{dataset}_dataset_generator"
            
            if hasattr(self, function_name):
                function_to_call = getattr(self, function_name)
                function_to_call(transcription, output_directory, path_to_audio_file, existing_speakers)
            else:
                print(f"No matching function found for dataset: {dataset}")
                
    #Work in progress
    def librispeech_dataset_generator(self, transcription, output_directory, path_to_audio_file, existing_speakers):
    
        librispeech_directory = os.path.join(output_directory, 'librispeech')
        
        # Path is librispeech_directory/generated_speaker_name/audio_id/
        # Inside audio_id there is the transcription and the audio files
        # The transcription file has n lines with:
        # generated_speaker_name-audio_id-number_of_audio transcription
        # The transcription file name is speaker_id-book_id.trans.txt
        
        filename = os.path.basename(path_to_audio_file)

        # Strip the file extension
        filename_without_extension, _ = os.path.splitext(filename)

        # Make it lowercase and remove non-alphabetic characters
        cleaned_folder_name = ''.join(char.lower() for char in filename_without_extension if char.isalpha())
        
        for existing_speaker in existing_speakers.values():
            current_speaker_audio_directory = os.path.join(librispeech_directory, existing_speaker["generated_speaker_name"], cleaned_folder_name)
            
            os.makedirs(current_speaker_audio_directory, exist_ok=True)
            
        for segment in transcription["segments"]:
            
            current_speaker_audio_directory = os.path.join(librispeech_directory, segment["generated_speaker_name"], cleaned_folder_name)

            speaker_id = segment["generated_speaker_name"]
            book_id    = cleaned_folder_name
            
            file_extension = os.path.splitext(segment["audio_file"])[1]
            
            max_number = -1
            for filename in os.listdir(current_speaker_audio_directory):
                if filename.startswith(f"{speaker_id}-{book_id}-") and not filename.lower().endswith('.txt'):
                    
                    current_number = int(filename.rsplit('-', 1)[1].rsplit('.', 1)[0])
                    max_number = max(max_number, current_number)

            new_number = max_number + 1

            new_filename = f"{speaker_id}-{book_id}-{new_number}{file_extension}" 
            
            new_full_path = os.path.join(current_speaker_audio_directory, new_filename)
            
            shutil.copy(segment["audio_file"], new_full_path)

            new_file_data = f"{speaker_id}-{book_id}-{new_number}.{file_extension}" 

            transcription_file_path = os.path.join(current_speaker_audio_directory, f"{speaker_id}-{book_id}.trans.txt")
            with open(transcription_file_path, 'a') as transcription_file:
                transcription_file.write(f"{new_file_data} {segment['text']}\n")
