import numpy as np
import torch
import gc 
import os

import speechmetrics #https://github.com/aliutkus/speechmetrics
from unsilence import Unsilence

#https://github.com/ina-foss/inaSpeechSegmenter # sudo apt-get install ffmpeg
from df.enhance import enhance, init_df, load_audio, save_audio

import torchaudio
from resemble_enhance.enhancer.inference import denoise, enhance as resemble_enhancer
from scipy.io.wavfile import write
from mayavoz.models import Mayamodel

from speech_dataset_generator.utils.utils import get_device

class AudioManager:
            
    def process(self, input_audio, output_directory, enhancers):
        
        output_audio_file = self.get_output_file_name(input_audio, output_directory)
        self.enhance_audio(input_audio, output_audio_file, enhancers)
        
        if not self.has_speech_quality(output_audio_file):
            return None
        
        return output_audio_file
    
    def get_output_file_name(self, input_audio, output_directory):
        # Extract the input file name without extension
        file_name_without_extension, extension = os.path.splitext(os.path.basename(input_audio))

        # Create the output file name by adding "_enhanced" suffix
        output_file_name = f"{file_name_without_extension}_enhanced{extension}"
        
        enhanced_directory = os.path.join(output_directory, "enhanced_audios")

        if not os.path.exists(enhanced_directory):
            os.makedirs(enhanced_directory)
            
        return os.path.join(enhanced_directory, output_file_name)

    #https://github.com/Rikorose/DeepFilterNet
    #alternatives to deepfilternet 
    #https://github.com/resemble-ai/resemble-enhance
    #https://github.com/shahules786/mayavoz 
    def enhance_audio(self, noisy_audio, output_audio_file, enhancers=["deepfilternet"]):
            
        temp_output = noisy_audio  # Initial audio loading

        for enhancement_type in enhancers:
            print(f"enhancing audio with {enhancement_type}")

            if enhancement_type == "deepfilternet":
                temp_output = self.enhance_audio_deepfilternet(temp_output, output_audio_file)
            elif enhancement_type == "resembleai":
                temp_output = self.enhance_audio_resembleai(temp_output, output_audio_file)
            elif enhancement_type == "mayavoz":
                temp_output = self.enhance_audio_mayavoz(temp_output, output_audio_file)
        
        self.remove_sliences(temp_output, output_audio_file)
        
        return temp_output

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

        enhanced = torch.cat(enhanced_chunks, dim=1)

        assert enhanced.shape == audio.shape, 'Enhanced audio shape does not match original audio shape.'

        save_audio(output_audio_file, enhanced, sr=df_state.sr())  
        
        # Free memory after inference
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
        
        model = Mayamodel.from_pretrained("shahules786/mayavoz-waveunet-valentini-28spk")
        waveform = model.enhance(noisy_audio)

        # this model only works with this sampling rate
        sr = 16000

        write(
            output_audio_file, rate=sr, data=waveform.detach().cpu().numpy()
        )
        
        # Free memory after inference
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        return output_audio_file
        
    # https://github.com/lagmoellertim/unsilence
    def remove_sliences(self, path_to_audio_file, output_audio_file):
        
        print("removing silences")
        u = Unsilence(path_to_audio_file)
        
        u.detect_silence()
        
        #rewrite the file with no silences
        u.render_media(output_audio_file, audio_only=True)  # Audio only specified
        
        # Free memory after inference
        del u
        torch.cuda.empty_cache()
        gc.collect()
        
    #https://github.com/aliutkus/speechmetrics
    def has_speech_quality(self, path_to_audio_file):
        
        window_length = 30 # seconds
        metrics = speechmetrics.load('absolute', window_length)

        scores = metrics(path_to_audio_file)

        average_scores = {}

        for metric_name, scores_array in scores.items():
            # Calculate the average of the array/list
            average_score = np.mean(scores_array)

            # Print the result
            #print(f"Average {metric_name} score: {average_score}")
            
            average_scores[metric_name] = average_score
            
        mos_value = average_scores['mosnet']
        if mos_value >= 3:
            return True
        
        print(f"Discarding audio {path_to_audio_file}. Not enough quality. MOS {mos_value} < 3")
       
        # Free memory after inference
        del metrics
        torch.cuda.empty_cache()
        gc.collect()
        
        return False
    
    def has_music(self, segmentation):

        labels = [item[0] for item in segmentation if item[0] in ('music')]

        if 'music' in labels:
            return True

        return False