import cv2
from moviepy import (
    VideoFileClip,
    concatenate_videoclips,
)
import numpy as np


def analyze_battle_scenes(
    video_path, frame_skip=50, change_threshold=40, motion_threshold=40
):
    print("戦闘シーンの抽出を開始します")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("動画ファイルを開けませんでした")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0

    prev_gray = None
    timestamps = []
    diffs = []

    while True:
        for _ in range(frame_skip):
            grabbed = cap.grab()
            if not grabbed:
                cap.release()
                print("動画の終端に到達しました")
                return extract_sections(timestamps)

        ret, frame = cap.retrieve()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_std = np.std(gray)

        motion_mag = 0
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion_mag = np.sum(diff > 25) / diff.size * 100

        prev_gray = gray
        # ----------ここが「動きが大きい時だけ戦闘判定」----------

        if motion_mag > motion_threshold and brightness_std > change_threshold:
            timestamps.append(frame_idx / fps)
            diffs.append(brightness_std)

        frame_idx += frame_skip

    cap.release()
    return extract_sections(timestamps)


def extract_sections(timestamps):
    print(f"timestamps: {timestamps}")

    margin = 10  # 10秒前後の余裕
    intervals = []

    for t in timestamps:
        start = max(0, t - margin)
        end = t + margin
        intervals.append([start, end])

    # 重複・連続区間をマージする関数
    def merge_intervals(intervals):
        if not intervals:
            return []
        # 開始時間でソート
        intervals.sort(key=lambda x: x[0])
        merged = [intervals]

        for current in intervals[1:]:
            last = merged[-1]
            # 重複・連続していればマージ
            if current[0] <= last[1]:
                last[1] = max(last[1], current[1])
            else:
                merged.append(current)
        return merged

    merged_sections = merge_intervals(intervals)

    # 2秒以上の区間だけ抽出（安全策）
    final_sections = [
        [round(s, 2), round(e, 2)] for s, e in merged_sections if (e - s) > 2
    ]

    print(f"final_sections: {final_sections}")

    return final_sections


def create_highlight_video(video_path, edit_sections, title=None, subtitles=None):
    video = VideoFileClip(video_path)
    clips = []
    for i, (start, end) in enumerate(edit_sections):
        sub_clip = video.subclipped(start, end)
        clips.append(sub_clip)

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


if __name__ == "__main__":
    video_file = r"input.mp4"
    sections = analyze_battle_scenes(video_file)

    print(f"sections: {sections}")

    # ハイライト動画の作成
    create_highlight_video(video_file, sections)
    print("ハイライト動画の作成が完了しました: output.mp4")
