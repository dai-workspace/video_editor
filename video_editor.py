from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips
from langchain_ollama import OllamaLLM
import whisper
import os
import re
import librosa
import numpy as np

# OpenMP重複ライブラリエラーを回避
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def extract_audio_from_video(video_path):
    print("動画から音声を抽出して一時ファイルに保存")
    video = VideoFileClip(video_path)
    if video.audio is None:
        return None, "動画に音声が含まれていません"

    temp_audio_path = "temp_audio.wav"
    video.audio.write_audiofile(temp_audio_path)
    return temp_audio_path, video


def transcribe_audio(audio_path):
    print("音声ファイルを文字起こし")
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        print(f"文字起こしエラー: {e}")
        return None


def analyze_audio_features(audio_path):
    print("音声の特徴（音量、ピッチ、テンポなど）を分析")
    try:
        # 音声ファイルを読み込み
        y, sr = librosa.load(audio_path, sr=None)

        y = y / np.max(np.abs(y))  # 音量を-1〜1の範囲に正規化

        # 音量レベル（RMS）を計算
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]

        # 音量の正規化
        rms_normalized = (rms - np.mean(rms)) / np.std(rms)

        # 高音量の区間を特定（閾値: 平均 + 1.0標準偏差）
        threshold = 1.0
        high_volume_indices = np.where(rms_normalized > threshold)[0]

        # 時間軸に変換
        time_stamps = librosa.frames_to_time(high_volume_indices, sr=sr)

        # 連続する高音量区間をグループ化
        volume_sections = []
        if len(time_stamps) > 0:
            start_time = time_stamps[0]
            prev_time = start_time

            for i in range(1, len(time_stamps)):
                current_time = time_stamps[i]
                # 1秒以内の連続区間をグループ化
                if current_time - prev_time > 1.0:
                    if prev_time - start_time > 2.0:  # 2秒以上の区間のみ
                        volume_sections.append([start_time, prev_time])
                    start_time = current_time
                prev_time = current_time

            # 最後の区間を追加
            if prev_time - start_time > 2.0:
                volume_sections.append([start_time, prev_time])

        # スペクトラル特徴量（ピッチ、テンポなど）
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]

        # テンポ推定
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        print(f"音声分析完了:")
        print(
            f"  テンポ: {float(tempo[0] if hasattr(tempo, '__len__') else tempo):.1f} BPM"
        )
        print(f"  高音量区間数: {len(volume_sections)}")
        print(f"  平均音量: {float(np.mean(rms)):.3f}")

        return {
            "volume_sections": volume_sections,
            "tempo": tempo,
            "rms": rms,
            "spectral_centroids": spectral_centroids,
            "spectral_rolloff": spectral_rolloff,
            "sample_rate": sr,
        }

    except ImportError:
        print("librosaがインストールされていません。音声分析をスキップします。")
        return None
    except Exception as e:
        print(f"音声分析エラー: {e}")
        return None


def get_interesting_sections(transcript, video_duration, audio_features=None):
    print("LLMを使用して面白い場面の時間区間を特定")
    # llm = OllamaLLM(model="llama2:7b", base_url="http://localhost:11434")
    llm = OllamaLLM(model="qwen2.5", base_url="http://localhost:11434")

    # 音声分析の結果を含めたプロンプト
    audio_info = ""
    if audio_features:
        try:
            audio_info = f"""Audio analysis results:
- Tempo: {float(audio_features['tempo'][0] if hasattr(audio_features['tempo'], '__len__') else audio_features['tempo']):.1f} BPM
- Number of high volume sections: {len(audio_features['volume_sections'])}
- Details of high volume sections: {audio_features['volume_sections']}

Please pay attention to the following points considering the audio features:
1. High volume sections (such as laughter, cheers, excited conversations)
2. Parts where the tempo changes (moments of excitement)
3. Parts where the volume changes abruptly (surprise or emotional peaks)
Please consider that the audio volume can be low at times, so pay attention to subtle changes or peaks in volume when identifying interesting segments."""
        except Exception as e:
            print(f"音声分析結果の表示でエラー: {e}")
            audio_info = ""

    prompt = f"""As a video editing expert, please identify the interesting time segments from the following video.
Please select at least 5 interesting segments distributed throughout the entire video of length {video_duration} seconds.

Transcript: {transcript}
Video length: {video_duration} seconds

{audio_info}

Please output at least 5 interesting time segments in the following exact format:
[[start_second, end_second], [start_second, end_second], [start_second, end_second], [start_second, end_second], [start_second, end_second]]

Conditions:
- Each segment should be at least 60 seconds long
- The segments should be distributed throughout the entire video
- When selecting segments, consider including a few seconds before and after any detected peaks or exciting moments to better capture the full context
- Output numbers only, no explanations, no extra text, no brackets other than those shown

Example: [[15,90], [120,180], [300,360], [450,510], [600,660]]

Make sure to output exactly the list of segments with no additional text."""

    response = llm.invoke(prompt)
    print(f"LLMの応答: {response}")

    # より柔軟で正確な時間区間の抽出
    sections = []

    # 例：すべての [start,end] を抽出
    section_pattern = r"\[(\d+),\s*(\d+)\]"
    matches = re.findall(section_pattern, response)

    sections = []
    for start, end in matches:
        sections.append([int(start), int(end)])

    return sections


def create_highlight_video(video_path, edit_sections):
    print("指定された区間を切り出してハイライト動画を作成")
    video = VideoFileClip(video_path)
    clips = [video.subclipped(start, end) for start, end in edit_sections]
    final_clip = concatenate_videoclips(clips)

    # fps_value = video.fps if video.fps is not None else 30
    # final_clip = final_clip.set_fps(fps_value)

    final_clip.write_videofile(
        "output.mp4",
        codec="libx264",
        audio_codec="aac",
    )

    for clip in clips:
        clip.close()
    final_clip.close()
    video.close()


def main():
    input_path = r"C:\Users\balnobi\Downloads\2398203250-1210583252-7cc15fb0-91da-4e21-a004-06f3846054ef.mp4"

    # 音声抽出
    temp_audio_path, video = extract_audio_from_video(input_path)
    if temp_audio_path is None:
        print(video)  # エラーメッセージ
        return

    try:
        # 文字起こし
        transcript = transcribe_audio(temp_audio_path)
        if transcript is None:
            transcript = "動画の字幕テキスト（エラーが発生しました）"

            # 音声特徴の分析
        audio_features = analyze_audio_features(temp_audio_path)
        if audio_features is None:
            print("音声分析に失敗したため、文字起こしのみで分析を行います")

        # 面白い場面の特定（音声分析結果も含める）
        edit_sections = get_interesting_sections(
            transcript, video.duration, audio_features
        )
        print(f"編集区間: {edit_sections}")

        # ハイライト動画の作成
        create_highlight_video(input_path, edit_sections)
        print("ハイライト動画の作成が完了しました: output.mp4")

    finally:
        # 一時ファイルの削除
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        if video:
            video.close()


if __name__ == "__main__":
    main()
