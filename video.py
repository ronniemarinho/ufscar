import yt_dlp
import whisper
import tempfile
import os

# 🔹 COLOQUE O LINK DO YOUTUBE AQUI:
YOUTUBE_URL = "https://www.youtube.com/watch?v=5nGlCbBTi8c&list=PLrUTNCwfEIbc_d4415nVR-jLBP4fIMPuM&index=7"

def extrair_audio_temporario(url):
    print("[1/2] Baixando áudio do YouTube...")
    temp_audio = tempfile.NamedTemporaryFile(suffix=".m4a", delete=False)

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': temp_audio.name,
        'quiet': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return temp_audio.name

def transcrever_youtube(url, modelo="small", idioma="pt"):
    audio_file = extrair_audio_temporario(url)

    print("[2/2] Transcrevendo áudio...")
    model = whisper.load_model(modelo)
    result = model.transcribe(audio_file, language=idioma)

    os.remove(audio_file)  # Apaga o arquivo temporário
    return result["text"]

if __name__ == "__main__":
    texto = transcrever_youtube(YOUTUBE_URL)
    print("\n--- TRANSCRIÇÃO ---\n")
    print(texto)
