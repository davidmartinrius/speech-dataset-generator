import os
import yt_dlp
import chromadb

from speech_dataset_generator.dataset_generator.dataset_generator import DatasetGenerator

    
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

def process_audio_files(audio_files, output_directory, start, end, enhancers, datasets):
    
    dataset_generator = DatasetGenerator()
    
    client = chromadb.PersistentClient(path=os.path.join(output_directory, "chroma_database"))
    collection = client.get_or_create_collection(name="speakers")

    for audio_file in audio_files:
        print("Processing:", audio_file)
        dataset_generator.process(audio_file, output_directory, start, end, enhancers, collection, datasets)