from moviepy.editor import VideoFileClip

def convert_mp4_to_wav(mp4_file, wav_file):
    try:
        # Tải tệp video
        video = VideoFileClip(mp4_file)
            # Trích xuất và lưu âm thanh
        video.audio.write_audiofile(wav_file)
        video.close()
        return wav_file
    except Exception as e:
        print(f"Có lỗi xảy ra: {e}")
