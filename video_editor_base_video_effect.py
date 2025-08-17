import cv2
from moviepy import (
    VideoFileClip,
    concatenate_videoclips,
)
import numpy as np


def analyze_battle_scenes(
    video_path,
    frame_skip=20,
    brightness_threshold=30,
    motion_threshold=50,
    margin=10,
    scene_time=30,
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

    while True:
        for _ in range(frame_skip):
            grabbed = cap.grab()
            if not grabbed:
                cap.release()
                print("動画の終端に到達しました")
                # 終了時に連続区間を抽出して返す処理へ
                return extract_intervals(
                    intervals=timestamps, merge_gap=margin, scene_time=scene_time
                )

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

        if motion_mag > motion_threshold and brightness_std > brightness_threshold:
            # print("motion_mag", motion_mag)
            # print("brightness_std", brightness_std)
            print("timestamps", frame_idx / fps)
            timestamps.append(frame_idx / fps)

        frame_idx += frame_skip


def extract_intervals(intervals, merge_gap=10, scene_time=30):
    intervals = sorted(intervals)
    clusters = []
    cluster = [intervals[0]]

    for t in intervals[1:]:
        if t - cluster[-1] <= 10:
            cluster.append(t)
        else:
            # clusterのサイズが複数なら結果に追加
            if len(cluster) > 1:
                clusters.append(cluster)
            # 1つだけのクラスタは捨てる
            cluster = [t]

    # 最後のclusterの処理
    if len(cluster) > 1:
        clusters.append(cluster)

    highlights = []
    for cluster in clusters:
        center = sum(cluster) / len(cluster)
        start = max(0, center - scene_time)
        end = center + scene_time
        highlights.append([start, end])

    return highlights


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

    print("ハイライト区間", sections)

    # ハイライト動画の作成
    create_highlight_video(video_file, sections)
    print("ハイライト動画の作成完了！: output.mp4")
