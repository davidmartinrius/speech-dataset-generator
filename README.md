# Speech Dataset Generator by [David Martin Rius](https://github.com/davidmartinrius/speech-dataset-generator)

This repository is dedicated to creating datasets suitable for training text-to-speech or speech-to-text models. The primary functionality involves transcribing audio files, enhancing audio quality when necessary, and generating datasets.


## Here are the key functionalities of the project:

1. **Dataset Generation:** The project allows for the creation of datasets with Mean Opinion Score (MOS).

2. **Silence Removal:** It includes a feature to remove silences from audio files, enhancing the overall quality.

3. **Sound Quality Improvement:** The project focuses on improving the quality of the audio.

4. **Audio Segmentation:** It can segment audio files within specified second ranges.

5. **Transcription:** The project transcribes the segmented audio, providing a textual representation.

6. **Gender Identification:** It identifies the gender of each speaker in the audio.

7. **Pyannote Embeddings:** Utilizes pyannote embeddings for speaker detection across multiple audio files.

8. **Automatic Speaker Naming:** Automatically assigns names to speakers detected in multiple audios.

9. **Multiple Speaker Detection:** Capable of detecting multiple speakers within each audio file.

### Example of the ouput folder:
```plaintext
output
|-- wavs
|   |-- 1272-128104-0000.wav
|   |-- 1272-128104-0001.wav
|   |-- ...
|   |-- 1272-128104-0225.wav
|-- dataset.csv
```

### Example of the csv content:

```plaintext


|   text                |   audio_file                |   speaker_name  |   gender   |   duration   |
|-----------------------|-----------------------------|-----------------|------------|--------------|
|   Hello, how are you? |   wavs/1272-128104-0000.wav |   Speaker12     |   male     |   4.5        |
|   This is a test.     |   wavs/1272-128104-0001.wav |   Speaker45     |   female   |   6.2        |
|   ...                 |   ...                       |   ...           |   ...      |   ...        |
|   Goodbye!            |   wavs/1272-128104-0225.wav |   Speaker78     |   male     |   5.1        |

```
## Installation

Due to existing incompatibilities among packages, generating a requirements.txt file is currently not feasible. However, installing the packages in the specified order should still work, even in the presence of conflicting dependencies as reported by pip.

Please note that this project has been tested and verified to work on Ubuntu 22. While it has not been tested on macOS, and Windows.

```bash

python3.10 -m venv venv 

source venv/bin/activate

pip install whisperx==3.1.1

pip install --upgrade faster-whisper==1.0.0

pip install pydub==0.25.1

pip install python-dotenv==1.0.1

pip install inaSpeechSegmenter==0.7.7

pip install unsilence==1.0.9

pip install deepfilternet

pip install resemble-enhance==0.0.1

pip install git+https://github.com/aliutkus/speechmetrics#egg=speechmetrics

```

### Needed agreement to run the code

**Important**: Make sure to agree to share your contact information to access the [pyannote embedding model](https://huggingface.co/pyannote/embedding). Similarly, access to the [pyannote speaker diarization model](https://huggingface.co/pyannote/speaker-diarization) may require similar agreement.

### Huggingface
You need to provide a HuggingFace token in a .env file

```
HF_TOKEN=yourtoken
```


## Usage

The main script `main.py` accepts command-line arguments for specifying the input file, output directory, time range, and types of filters. You can process a single file or an entire folder of audio files.

```bash

python main.py --input_file_path <path_to_audio_file> --output_directory <output_directory> --range_times <start-end> --types <filter_types>

```

- `--input_file_path`: Path to the input audio file.

- `--input_folder`: Path to the input folder containing audio files.

- `--output_directory`: Output directory for audio files.

- `--range_times`: Specify a range of two integers in the format "start-end". Default is 4-10.

- `--types`: List of types. Default is deepfilternet. You can combine filters too, and disable with `--types None`.

**Examples:**

```bash

python main.py --input_file_path /path/to/audio/file.mp3 --output_directory /output/directory --range_times 4-10 --types deepfilternet enhance_audio_resembleai

python main.py --input_file_path /path/to/audio/file.mp3 --output_directory /output/directory --range_times 4-10 --types None

python main.py --input_file_path /path/to/audio/file.mp3 --output_directory /output/directory --range_times 4-10 --types enhance_audio_resembleai

python main.py --input_folder /path/to/folder/of/audios --output_directory /output/directory --range_times 4-10 --types deepfilternet 

```

# Next Steps

Generator of multiple types of dataset formats

Dataset converter. for example, from LibriSpeech to Common Voice and vice versa, etc.

- [ ] **LibriSpeech**

- [ ] **Common Voice 11**

- [ ] **VoxPopuli**

- [ ] **TED-LIUM**

- [ ] **GigaSpeech**

- [ ] **SPGISpeech**

- [ ] **Earnings-22**

- [ ] **AMI**


## Used packages in this project
This project uses several open-source libraries and tools for audio processing. Special thanks to the contributors of these projects.

- Python 3.10

- [whisperx](https://github.com/m-bain/whisperX?tab=readme-ov-file) (v3.1.1)

- [pydub](https://github.com/jiaaro/pydub) (v0.25.1)

- [python-dotenv](https://github.com/theskumar/python-dotenv) (v1.0.1)

- [inaSpeechSegmenter](https://github.com/ina-foss/inaSpeechSegmenter) (v0.7.7)

- [unsilence](https://github.com/lagmoellertim/unsilence) (v1.0.9)

- [deepfilternet](https://github.com/Rikorose/DeepFilterNet)

- [resemble-enhance](https://github.com/resemble-ai/resemble-enhance) (v0.0.1)

- [speechmetrics](https://github.com/aliutkus/speechmetrics)

- [pyannote](https://huggingface.co/pyannote) (embedding model and speaker diarization model)

## License

This project is licensed under the [MIT License](LICENSE).
