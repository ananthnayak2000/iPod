import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from streamlit_player import st_player
import pyaudio
import wave
import os
import asyncio
from threading import Thread
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from deepgram import DeepgramClient, PrerecordedOptions
from lmnt.api import Speech

# API keys and initial setup
DEEPGRAM_API_KEY = ""
API_KEY = ""
LMNT_API_KEY = ""  # Add your LMNT API key

chat = ChatAnthropic(temperature=0, api_key=API_KEY, model_name="claude-3-sonnet-20240229")
deepgram = DeepgramClient(DEEPGRAM_API_KEY)

def record_audio(filename="output.wav", duration=5):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = duration
    audio_interface = pyaudio.PyAudio()
    stream = audio_interface.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    audio_interface.terminate()
    waveFile = wave.open(filename, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio_interface.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

def get_question_transcript(file_path):
    with open(file_path, 'rb') as audio_data:
        options = PrerecordedOptions(smart_format=False, model="nova-2", language="en-US")
        response = deepgram.listen.prerecorded.v('1').transcribe_file({'buffer': audio_data}, options)
        return response['results']['channels'][0]['alternatives'][0]['transcript']

def get_anthropic_response(question, transcript):
    system = f"Imagine you are a podcast host. In the podcast you are talking about a certain topic. At some point in your podcast, I ask you a question based on what you were talking in the podcast. You need to answer my question taking in context from the podcast transcript and help me understand what you were just talking about. Remember, answer has to be in first person view. Like you were talking to me. Attached here is the podcast transcript and all the contents of the podcast: {transcript}"
    human = f"{question}"
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    chain = prompt | chat
    response = chain.invoke({"transcript": transcript, "question": question})
    return response.content

def synthesize_speech(text, filename='response_audio.wav'):
    async def async_synthesize(text, filename):
        async with Speech(LMNT_API_KEY) as speech:
            synthesis = await speech.synthesize(text, voice='d9b944b9-fff7-4f38-8b68-27abd33eedab', format='wav') 
            with open(filename, 'wb') as f:
                f.write(synthesis['audio'])

    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)
    try:
        new_loop.run_until_complete(async_synthesize(text, filename))
    finally:
        new_loop.close()

st.title('IPod')
link = st.text_input('Enter your YouTube link', value='')

if link:
    try:
        video_id = link.split('v=')[1].split('&')[0]
        transcripts = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = ' '.join([item['text'] for item in transcripts])
        #st.text_area('Transcript', value=transcript_text, height=300)
        st_player(link)

        if st.button('Ask me'):
            with st.spinner('Recording...'):
                record_audio()
            st.success('Recording complete.')

            questionasked = get_question_transcript('output.wav')
            st.write('Your Question:', questionasked)
            
            response = get_anthropic_response(questionasked, transcript_text)
            synthesize_speech(response)
            st.audio('response_audio.wav', format='audio/wav')

            st.write('Answer :', response)

            

    except Exception as e:
        st.error(f"Error: {e}")