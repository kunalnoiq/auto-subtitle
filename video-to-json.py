from moviepy.editor import VideoFileClip
import whisper
import openai
from dotenv import load_dotenv
import json

load_dotenv()

detected_language = None

def video_to_audio(input_video, output_audio):

    video_clip = VideoFileClip(input_video)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(output_audio)
    video_clip.close()
    audio_clip.close()

def detect_language_and_transcribe(audio_path, output_path):
    model = whisper.load_model("base")

    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    _, probs = model.detect_language(mel)

    detected_language = max(probs, key=probs.get)
    
    print(f"Detected language: {detected_language}")

    result = model.transcribe(word_timestamps=True, audio=audio_path)

    segments = list(result["segments"])
    wordlevel_info = []

    for segment in segments:
        for word in segment['words']:
            wordlevel_info.append({"word": word['word'].upper(), "start": word['start'], "end": word['end']})

    with open('data.json', 'w') as f:
        json.dump(wordlevel_info, f, indent=4)
    
    print(f"Result stored in 'data.json' ")

# def translate_text(output_text_path, source_language, target_language):

#     with open(output_text_path, 'r', encoding='utf-8') as input_file:
#         input_text = input_file.read()
    
#     openai.api_key = os.getenv("OPENAI_API_KEY")

#     conversation = [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": f'Translate the following text from {source_language} to {target_language}: "{input_text}"'}
#     ]

#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=conversation
#     )

#     translated_text = response['choices'][0]['message']['content']

#     file_name = f"./text/{target_language}.txt"
#     with open(file_name, 'w', encoding='utf-8') as file:
#         file.write(translated_text)



if __name__ == "__main__":
    input_video_file = './video/sample_video.mp4'
    output_audio_file = './audio/sample_output_audio.mp3'

    video_to_audio(input_video_file, output_audio_file)
    print(f"Conversion complete. Audio saved to {output_audio_file}")

    output_text_path='./text/sample_text.txt'
    
    detect_language_and_transcribe(output_audio_file, output_text_path)
    
    target_language = "es"

    # translate_text(output_text_path, detected_language, target_language)

    # print(f"Translated text saved to file: {target_language}.txt")


