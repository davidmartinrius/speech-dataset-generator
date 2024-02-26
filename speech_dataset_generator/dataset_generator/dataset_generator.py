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

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")

class DatasetGenerator:

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

    def write_data_to_csv(self, transcription, csv_file_name, language):
        
        for segment in transcription["segments"]:
            
            newData = {
                'text': segment["text"],
                'audio_file': segment["audio_file"],
                'speaker': segment["generated_speaker_name"],
                'gender': segment["gender"],
                'duration': segment["duration"],
                'language': language
            }

            with open(csv_file_name, 'a', encoding='utf-8') as csvFile:
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

        self.audio_manager_instance = AudioManager()
        
        self.wavs_directory = os.path.join(output_directory, 'wavs')
        if not os.path.exists(self.wavs_directory):
            os.makedirs(self.wavs_directory)

        csv_file_name = os.path.join(output_directory, "dataset.csv") 
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

        self.write_data_to_csv(transcription, csv_file_name, language)