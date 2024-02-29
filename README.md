# Speech Dataset Generator by [David Martin Rius](https://github.com/davidmartinrius/speech-dataset-generator)

[![00019-2780374442](https://github.com/davidmartinrius/speech-dataset-generator/assets/16558194/8091ba96-6017-4645-b001-a9e3310982e8)](https://github.com/davidmartinrius/speech-dataset-generator)

This repository is dedicated to creating datasets suitable for training text-to-speech or speech-to-text models. The primary functionality involves transcribing audio files, enhancing audio quality when necessary, and generating datasets.


## Here are the key functionalities of the project:

1. **Dataset Generation:** Creation of datasets with Mean Opinion Score (MOS).

2. **Silence Removal:** It includes a feature to remove silences from audio files, enhancing the overall quality.

3. **Sound Quality Improvement:** The project focuses on improving the quality of the audio.

4. **Audio Segmentation:** It can segment audio files within specified second ranges.

5. **Transcription:** The project transcribes the segmented audio, providing a textual representation.

6. **Gender Identification:** It identifies the gender of each speaker in the audio.

7. **Pyannote Embeddings:** Utilizes pyannote embeddings for speaker detection across multiple audio files.

8. **Automatic Speaker Naming:** Automatically assigns names to speakers detected in multiple audios.

9. **Multiple Speaker Detection:** Capable of detecting multiple speakers within each audio file.

10. **Store speaker embeddings:** The speakers are detected and stored in a Chroma database, so you do not need to assign a speaker name.

11. **Syllabic and words-per-minute metrics**

### Example of the ouput folder:
```plaintext
output
|-- wavs
|   |-- 1272-128104-0000.wav
|   |-- 1272-128104-0001.wav
|   |-- ...
|   |-- 1272-128104-0225.wav
|-- metadata.csv
```

### Example of the csv content:

Consider that the values provided are purely fictitious and intended solely for illustrative purposes in this example.

```plaintext

| text                    | audio_filename               | speaker_id     | gender     | duration    | language    | words_per_minute   | syllables_per_minute |
|-------------------------|------------------------------|----------------|------------|-------------|-------------|--------------------|----------------------|
| Hello, how are you?     | wavs/1272-128104-0000.wav    | Speaker12      | male       | 4.5         | en          | 22.22              | 1.11                 |
| Hola, ¿cómo estás?      | wavs/1272-128104-0001.wav    | Speaker45      | female     | 6.2         | es          | 20.97              | 0.81                 |
| This is a test.         | wavs/1272-128104-0002.wav    | Speaker23      | male       | 3.8         | en          | 26.32              | 1.32                 |
| ¡Adiós!                 | wavs/1272-128104-0003.wav    | Speaker67      | female     | 7.0         | es          | 16.43              | 0.57                 |
| ...                     | ...                          | ...            | ...        | ...         | ...         | ...                | ...                  |
| Goodbye!                | wavs/1272-128104-0225.wav    | Speaker78      | male       | 5.1         | en          | 1.41               | 1.18                 |

```
## Installation

Please note that this project has been tested and verified to work on Ubuntu 22. Although it has not been tested on macOS and Windows nor on other unix distributions.

```bash

python3.10 -m venv venv 

source venv/bin/activate

pip install -r requirements.txt

or

pip install -e .

```


### Needed agreement to run the code

**Important**: Make sure to agree to share your contact information to access the [pyannote embedding model](https://huggingface.co/pyannote/embedding). Similarly, access to the [pyannote speaker diarization model](https://huggingface.co/pyannote/speaker-diarization) may require similar agreement.

### Huggingface
You need to provide a HuggingFace token in a .env file

```
HF_TOKEN=yourtoken
```


## Usage

The main script `speech_dataset_generator/main.py` accepts command-line arguments for specifying the input file, output directory, time range, and types of enhancers. You can process a single file or an entire folder of audio files.
Also you can use a youtube video or a youtube playlist as input.

```bash

python speech_dataset_generator/main.py --input_file_path <path_to_audio_file> --output_directory <output_directory> --range_times <start-end> --enhancers <enhancer_types>

```

- `--input_file_path`: (required) Path to the input audio file. Cannot be used with input folder.

- `--input_folder`: (required) Path to the input folder containing audio files. Cannot be used with input_file_path

- `--youtube_download`: (optional) Link or links separated by space of youtube videos or playlists. Can be combined with --input_file_path or --input_folder

- `--output_directory`: Output directory for audio files.

- `--range_times`: Specify a range of two integers in the format "start-end". Default is 4-10.

- `--enhancers`: You use audio enhancers: --enhancers deepfilternet resembleai. Will be executed in the order you write it. By default no enhancer is set. By now deepfilternet gives the best results when enhancing and denoising an audio. 

**Examples:**

```bash

python speech_dataset_generator/main.py --input_file_path /path/to/audio/file.mp3 --output_directory /output/directory --range_times 4-10 --enhancers deepfilternet

python speech_dataset_generator/main.py --input_file_path /path/to/audio/file.mp3 --output_directory /output/directory --range_times 4-10

python speech_dataset_generator/main.py --input_file_path /path/to/audio/file.mp3 --output_directory /output/directory --range_times 4-10 --enhancers resembleai

python speech_dataset_generator/main.py --input_file_path /path/to/audio/file.mp3 --output_directory /output/directory --range_times 4-10 --enhancers deepfilternet resembleai

python speech_dataset_generator/main.py --input_folder /path/to/folder/of/audios --output_directory /output/directory --range_times 4-10 --enhancers deepfilternet 

python speech_dataset_generator/main.py --youtube_download https://www.youtube.com/watch\?v\=ID --output_directory /output/directory --range_times 5-15 --enhancers deepfilternet resembleai

python speech_dataset_generator/main.py --youtube_download https://www.youtube.com/watch\?v\=ID  --input_file_path /path/to/audio/file.mp3 --output_directory /output/directory --range_times 5-15 --enhancers deepfilternet resembleai

python speech_dataset_generator/main.py --youtube_download https://www.youtube.com/watch\?v\=ID  --input_folder /path/to/folder/of/audios --output_directory /output/directory --range_times 5-15 --enhancers deepfilternet resembleai

```

## Notes

### Audio enhancer argument
You can combine --enhancers. There are available "deepfilternet", "resembleai" and soon "mayavoz".

If you pass multiples those will be executed in the order that are passed. In the case you don't pass enhancers no enhancer will be used.

By default, no enhancer is used. I suggest using deepfilternet for noisy audios.

The output sound of resembleai sometimes can be a little distorted. So, it is not always a good choise.

You can play with those and combine them.

### The audio is not always 100% used to be splitted into files

An input audio may not be used completely. Here some reasons:
- The range_times do not fit a transcripted segment.
- The segment has music or not enough quality (MOS under 3), even when enhanced.

If you are not using enhancers and the segments are being discarted because of bad quality you can try --enhancers argument with deepfilternet, resembleai or both together.

### Gender detection

The gender detection is not accurate enough when probably mixed. If there is no clear gender due the voice, maybe return male or female at random.

# Next Steps

## External input sources

- [X] [**yt-dlp**](https://github.com/yt-dlp/yt-dlp)

## Vector database

- [X] **Store speaker embeddings in Chroma vector database**

## Refactor code

- [X] Everything is inside main.py The code needs to be reorganized.

## Speech rate 

- [X] Detect the speech speed rate for each sentence and add it to the csv output file. The metrics are words per minute (wpm) and syllables per minute (spm)

## Google colab

- [ ] Add a speech_dataset_generator_example.ipynb file with all available options applied to some noisy audios and good quality audios.
  
## Support multiple datasets

Generator of multiple types of datasets:

- [X] **LJSpeech** This is the default one. When you generate a new dataset a LJSpeech format is given. It still does not split by train/dev/test, but creates a metadata.csv
    
- [ ] **LibriSpeech** Currently in development. Work in progress

- [ ] **Common Voice 11**

- [ ] **VoxPopuli**

- [ ] **TED-LIUM**

- [ ] **GigaSpeech**

- [ ] **SPGISpeech**

- [ ] **Earnings-22**

- [ ] **AMI**
    
- [ ] **VCTK**

## Dataset converter.

For example, from LibriSpeech to Common Voice and vice versa, etc.

I have to look for a way to extract all the needed features for each dataset type. Also find the best way to divide the dataset into train, dev and test taking into account the input data provided by the user. 

## Gradio interface

- [ ] **Generate datasets**

- [ ] **Dataset converter**

## Docker image

- [ ] **Create a docker image for ease of use.**

## Runpod serverless instance

In case you do not have a gpu or you want to distribute this as a service.

runpod is a cloud GPU on demand. It has a good integration with python and docker. Also it has an affordable pricing.

- [ ] Explain how to create a storage in runpod
- [ ] Create a base installation to the storage with a Pod
- [ ] Launch a serverless instance with a Docker instance of this project
- [ ] Call the serverless custom API endpoints to upload files, download generated datasets, convert datasets to other types of datasets, etc

## Used packages in this project
This project uses several open-source libraries and tools for audio processing. Special thanks to the contributors of these projects.

- Python 3.10

- [whisperx](https://github.com/m-bain/whisperX?tab=readme-ov-file) (v3.1.1)

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (1.0.0) 

- [pydub](https://github.com/jiaaro/pydub) (v0.25.1)

- [python-dotenv](https://github.com/theskumar/python-dotenv) (v1.0.1)

- [inaSpeechSegmenter](https://github.com/ina-foss/inaSpeechSegmenter) (v0.7.7)

- [unsilence](https://github.com/lagmoellertim/unsilence) (v1.0.9)

- [deepfilternet](https://github.com/Rikorose/DeepFilterNet)

- [resemble-enhance](https://github.com/resemble-ai/resemble-enhance) (v0.0.1)

- [speechmetrics](https://github.com/aliutkus/speechmetrics)

- [pyannote](https://huggingface.co/pyannote) (embedding model and speaker diarization model)

- [yt-dlp](https://github.com/yt-dlp/yt-dlp)

- [Chroma](https://github.com/chroma-core/chroma)

## License

If you plan to use this project in yours: [whisperX](https://github.com/m-bain/whisperX?tab=BSD-4-Clause-1-ov-file) is currently under the BSD-4-Clause license, yt-dlp has no license and all others are under the MIT license or Apache 2.0 license.

This project is licensed under the [MIT License](LICENSE).
