import os
import yt_dlp
import chromadb
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, unquote

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

def get_librivox_audio_files(urls, output_directory):
    
    downloaded_files = []
    if not urls:
        return downloaded_files
    
    librivox_files_output_directory = os.path.join(output_directory, "librivox") 

    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'sec-ch-ua': '"Not A(Brand";v="99", "Google Chrome";v="121", "Chromium";v="121"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Linux"'
    }
    
    if not os.path.exists(librivox_files_output_directory):
        os.makedirs(librivox_files_output_directory)

    for url in urls:
        
        parsed_url = urlparse(url)
        headers['GET'] = f'{parsed_url.path} HTTP/1.1'

        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        results = soup.find_all(class_='chapter-name')
            
        hrefs = [c['href'] for c in results]
            
        print('found {} chapters to download'.format(len(hrefs)))

        for audio in hrefs:
            
            audio_file_name = audio.split('/')[-1][:-3]+'wav'
            
            audio_path = os.path.join(librivox_files_output_directory, audio_file_name) 

            if os.path.exists(audio_path):
                print(f"Already exists: {audio_file_name}")
                continue

            print('Downloading {} to:'.format(audio),audio_path)
            file = requests.get(audio, headers=headers)
            with open(audio_path, 'wb') as f:
                f.write(file.content)
    
    downloaded_files = [os.path.join(librivox_files_output_directory, file_name) for file_name in os.listdir(librivox_files_output_directory)]

    return downloaded_files
            
def get_tedtalks_audio_files(urls, output_directory):
    
    downloaded_files = []
    if not urls:
        return downloaded_files
    
    tedtalks_files_output_directory = os.path.join(output_directory, "tedtalks") 

    if not os.path.exists(tedtalks_files_output_directory):
        os.makedirs(tedtalks_files_output_directory)

    for url in urls:
        
        print("processing ted talk", url)
        
        parsed_url = urlparse(url)
        filename = unquote(os.path.basename(parsed_url.path))

        response = requests.get(url)

        if response.status_code == 200:
            
            audio_path = os.path.join(tedtalks_files_output_directory, filename) 

            with open(audio_path, 'wb') as audio_file:
                audio_file.write(response.content)
                
            print(f"Downloaded '{url}' to '{audio_path}'")
        else:
            print(f"Failed to download the audio file from url {url}.")
            
    downloaded_files = [os.path.join(tedtalks_files_output_directory, file_name) for file_name in os.listdir(tedtalks_files_output_directory)]

    return downloaded_files
        
def process_audio_files(audio_files, output_directory, start, end, enhancers, datasets):
    
    dataset_generator = DatasetGenerator()
    
    client = chromadb.PersistentClient(path=os.path.join(output_directory, "chroma_database"))
    collection = client.get_or_create_collection(name="speakers")

    for audio_file in audio_files:
        print("Processing:", audio_file)
        dataset_generator.process(audio_file, output_directory, start, end, enhancers, collection, datasets)