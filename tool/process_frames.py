"""
批处理视频，按间隔采样生成单帧数据集：
1) <dataset_name>_{x}: 每 x 帧截取 1 帧，单帧保存
2) 额外生成 grouping 索引（每 5 张一组），供 LLM 端 (doubao_image_api) 拼接与描述。

说明：
- 仅保存单帧，不再在此处拼接；拼接与描述交由 doubao_image_api 完成。
- grouping 索引文件保存为 groups_of_five.json，位于每个视频文件夹下。
"""

import argparse
import os
import re
import shutil
from pathlib import Path

import cv2
import json


def save_sampled_frames(video_path: str, output_folder: str, sample_interval: int) -> int:
	"""每 sample_interval 帧保存 1 帧到 output_folder，并返回保存数量。"""
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		print(f"    错误：无法打开视频文件 {video_path}")
		return 0

	os.makedirs(output_folder, exist_ok=True)
	frame_count = 0
	saved = 0

	while True:
		ret, frame = cap.read()
		if not ret:
			break

		if frame_count % sample_interval == 0:
			fname = f"{saved + 1:06d}.jpg"
			cv2.imwrite(os.path.join(output_folder, fname), frame)
			saved += 1

		frame_count += 1

	cap.release()
	return saved


def write_group_index(output_folder: str, total_frames: int, group_size: int = 5) -> None:
	"""生成分组索引文件 groups_of_five.json，便于下游 5 张一组处理。"""
	groups = []
	frame_files = [f"{i:06d}.jpg" for i in range(1, total_frames + 1)]
	for i in range(0, len(frame_files), group_size):
		chunk = frame_files[i:i + group_size]
		groups.append(chunk)

	index_path = os.path.join(output_folder, "groups_of_five.json")
	with open(index_path, "w", encoding="utf-8") as f:
		json.dump(groups, f, indent=2, ensure_ascii=False)


def process_dataset(source_base: str, target_base: str, dataset_name: str, sample_interval: int, clear_target: bool) -> None:
	"""遍历源数据集，生成 <dataset_name>_<x>（仅单帧）并写出 5 张一组索引供 LLM 使用。"""
	target_frames_base = os.path.join(target_base, f"{dataset_name}_{sample_interval}")

	os.makedirs(target_frames_base, exist_ok=True)

	if clear_target:
		print("清空目标目录...")
		if os.path.exists(target_frames_base):
			shutil.rmtree(target_frames_base)
			os.makedirs(target_frames_base, exist_ok=True)

	folder_counter = 1
	total_videos = 0
	total_frames_saved = 0

	# 遍历源目录下所有 .avi/.mp4 文件，按照 Pxx/cam01 结构（若存在）
	for p_dir in sorted(os.listdir(source_base)):
		p_path = os.path.join(source_base, p_dir)
		if not os.path.isdir(p_path):
			continue

		# 如果存在 cam01 结构则进入，否则直接在当前目录找视频
		candidate_dirs = [os.path.join(p_path, "cam01")] if os.path.exists(os.path.join(p_path, "cam01")) else [p_path]

		for cdir in candidate_dirs:
			for file in sorted(os.listdir(cdir)):
				if not (file.endswith('.avi') or file.endswith('.mp4')):
					continue

				source_file = os.path.join(cdir, file)
				folder_name = f"{folder_counter:06d}"

				frames_folder = os.path.join(target_frames_base, folder_name)

				os.makedirs(frames_folder, exist_ok=True)

				print(f"处理视频：{file} -> {folder_name}/ (间隔 {sample_interval})")

				saved_frames = save_sampled_frames(source_file, frames_folder, sample_interval)

				# 写出 5 张一组索引，供 doubao_image_api 拼接与描述
				if saved_frames > 0:
					write_group_index(frames_folder, saved_frames, group_size=5)

				total_frames_saved += saved_frames
				total_videos += 1
				folder_counter += 1

	print("\n完成！")
	print(f"总视频数：{total_videos}")
	print(f"{dataset_name}_{sample_interval}：保存 {total_frames_saved} 张单帧")
	print(f"输出目录：\n  {target_frames_base}")


def parse_args():
	parser = argparse.ArgumentParser(description="按固定间隔提帧，输出 <dataset_name>_<x> 单帧与分组索引")
	parser.add_argument('--source-base', required=True, help='源视频根目录，例如 BreakfastII_15fps_qvga_sync')
	parser.add_argument('--target-base', required=True, help='输出根目录，例如 dataset/breakfast')
	parser.add_argument('--dataset-name', required=True, help='数据集名称前缀，如 breakfast 或 youcook')
	parser.add_argument('--sample-interval', type=int, default=5, help='每 N 帧取 1 帧 (默认 5)')
	parser.add_argument('--clear-target', action='store_true', help='先清空输出目录')
	return parser.parse_args()


def main():
	args = parse_args()
	process_dataset(args.source_base, args.target_base, args.dataset_name, args.sample_interval, args.clear_target)


if __name__ == "__main__":
	main()
