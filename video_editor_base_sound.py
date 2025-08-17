import ast
import json
import re
import cv2
from langchain_ollama import OllamaLLM
from moviepy import CompositeVideoClip, TextClip, VideoFileClip, concatenate_videoclips


def analyze_video_color_changes(video_path, frame_skip=5, change_threshold=30):
    print("映像の色彩変化を解析")
    cap = cv2.VideoCapture(video_path)
    prev_frame_hsv = None
    color_diffs = []
    timestamps = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            # BGR→HSV変換し平均色を取得
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            avg_hue = hsv_frame[:, :, 0].mean()  # 色相(H)の平均

            if prev_frame_hsv is not None:
                diff = abs(avg_hue - prev_frame_hsv)
                color_diffs.append(diff)
                timestamps.append(frame_idx / fps)

            prev_frame_hsv = avg_hue

        frame_idx += 1

    cap.release()

    # 色変化の大きい地点を区間として抽出
    sections = []
    start = None
    for i, diff in enumerate(color_diffs):
        if diff > change_threshold:
            if start is None:
                start = timestamps[i]
        else:
            if start is not None:
                end = timestamps[i]
                if end - start > 2:  # 2秒以上の区間
                    sections.append([start, end])
                start = None

    if start is not None:
        sections.append([start, timestamps[-1]])

    print(f"色彩変化で検出した編集区間候補: {sections}")
    return sections


def parse_sections(response: str):
    try:
        # JSONが含まれていれば取り出してパース
        json_pattern = r"\{.*\}"
        match = re.search(json_pattern, response, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return data.get("segments", [])
    except Exception as e:
        print(f"区間解析エラー: {e}")
    return []


def fallback_pairs(response: str):
    pairs = []
    matches = re.findall(r"(\d+)\s*[-–]\s*(\d+)", response)  # 930-934 など
    for s, e in matches:
        pairs.append([int(s), int(e)])
    return pairs


def get_interesting_sections_video_color(video_path, video_duration):
    print("LLMで映像色彩変化を含めて面白いシーンを推定")

    # 映像解析
    color_sections = analyze_video_color_changes(video_path)

    prompt = f"""As a video editing expert, identify interesting time segments from the following video based on color changes.
Video length: {video_duration} seconds
Significant color change segments: {color_sections}
Please output at least 5 interesting time segments formatted as:
[[start_second, end_second], [start_second, end_second], [start_second, end_second], [start_second, end_second], [start_second, end_second]]

Conditions:
- Each segment at least 60 seconds long
- Segments distributed throughout the video
- Include a few seconds before/after peaks for full context
- Output only list of segments, no extra text"""

    llm = OllamaLLM(model="qwen2.5", base_url="http://localhost:11434")
    response = llm.invoke(prompt)
    print(f"LLMの応答: {response}")

    sections = parse_sections(response)

    if not sections:
        sections = fallback_pairs(response)

    # すべての [start,end] を抽出
    section_pattern = r"\[(\d+),\s*(\d+)\]"
    matches = re.findall(section_pattern, response)

    for start, end in matches:
        sections.append([int(start), int(end)])

    return sections


def add_title_and_combine(clips, title):
    # タイトルを3秒間表示
    title_clip = TextClip(
        title, fontsize=60, color="yellow", bg_color="black", size=(1280, 120)
    ).set_duration(3)
    return concatenate_videoclips([title_clip] + clips)


def add_subtitle(clip, text):
    # 各クリップ下部に字幕を表示
    subtitle = TextClip(
        text, fontsize=40, color="white", bg_color="black", size=(1280, 70)
    ).set_duration(clip.duration)
    subtitle = subtitle.set_position(("center", "bottom"))
    return CompositeVideoClip([clip, subtitle])


def create_highlight_video(video_path, edit_sections, title=None, subtitles=None):
    video = VideoFileClip(video_path)
    clips = []
    for i, (start, end) in enumerate(edit_sections):
        sub_clip = video.subclipped(start, end)
        # 字幕があれば追加
        if subtitles and i < len(subtitles):
            sub_clip = add_subtitle(sub_clip, subtitles[i])
        clips.append(sub_clip)
    # タイトルがあれば動画先頭に追加
    if title:
        final_clip = add_title_and_combine(clips, title)
    else:
        final_clip = concatenate_videoclips(clips)
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
    input_path = r"input.mp4"

    video = VideoFileClip(input_path)
    video_duration = video.duration

    try:

        # 映像色彩変化を利用して編集区間を取得
        edit_sections = get_interesting_sections_video_color(input_path, video_duration)
        edit_sections = sorted(edit_sections, key=lambda x: x[0])
        print(f"編集区間: {edit_sections}")

        # ハイライト動画の作成
        create_highlight_video(input_path, edit_sections)
        print("ハイライト動画の作成が完了しました: output.mp4")

    finally:
        video.close()


if __name__ == "__main__":
    main()
