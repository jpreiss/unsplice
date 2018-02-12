#!/usr/bin/env python

from __future__ import print_function
import argparse
import cPickle as pickle
import os
import sys
import time

import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import moviepy.video.tools.tracking as tracking
import moviepy.editor as mpe
import pygame

def underscore(s):
	print(s)
	print("-" * len(s))

def rightpad(s, n):
	return s + " " * (n - len(s))

def zip_max(a, b):
	for x, y in zip(a, b):
		yield max(x, y)

def tabular(rows):
	if len(rows) == 0:
		return
	nc = len(rows[0])
	for row in rows:
		assert len(row) == nc
	rows = [[str(x) for x in row] for row in rows]
	colmax = [0 for _ in range(nc)]
	for row in rows:
		colmax = list(zip_max(colmax, [len(x) for x in row]))
	for row in rows:
		pads = (rightpad(s, cmax) for s, cmax in zip(row, colmax))
		rowstr = " ".join(pads)
		yield rowstr

def print_info(clip):
	print("")
	underscore("clip info:")
	for s in tabular([
		["name:", clip.filename],
		["size:", clip.size],
		["duration:", str(clip.duration) + " secs"],
		["fps:", clip.fps],
	]):
		print(s)
	print("")

def sample_tiles(img, size, N):
	h, w, _ = img.shape
	h_range, w_range = ((int(0.25 * x), int(0.75 * x)) for x in (h, w))
	def make_tile():
		y = np.random.randint(*h_range)
		x = np.random.randint(*w_range)
		return x, y, img[y:(y+size),x:(x+size),:]
	return [make_tile() for _ in range(N)]

def find_likely_cuts(clip, n_top=None, tile_size=8, n_tiles=20, radius=15):
	dt = 1.0 / clip.fps
	match_sums = []
	if n_top is None:
		# most FPV video does not contain fast cuts.
		# might need to increase n_top for other kinds of video.
		n_top = clip.duration / 5
	frames = clip.iter_frames(progress_bar=True)
	prev_frame = None
	for i, frame in enumerate(frames):
		# TODO cleanup
		if prev_frame is None:
			prev_frame = frame
			continue

		tiles = sample_tiles(prev_frame, tile_size, n_tiles)
		match_tot = 0
		for x, y, pattern in tiles:
			r = radius
			tsz = tile_size
			pic = frame[(y-r):(y+tsz+r),(x-r):(x+tsz+r),:]
			matches = cv2.matchTemplate(pattern, pic, cv2.TM_CCOEFF_NORMED)
			best_match = np.max(matches.flatten())
			match_tot += best_match

		match_sums.append((match_tot, i * dt))
		prev_frame = frame

	return [t for weight, t in sorted(match_sums)[:n_top]]

# FPV videos do not have extremely rapid shots,
# so if we detect two "cuts" nearby, it's more likely
# that we're in an extreme roll
def remove_nearby(times):
	times = np.array(times)
	for t in times:
		delta = times - t
		close = np.logical_and(np.abs(delta) < 1.0, delta != 0)
		if close.any():
			others = np.delete(times, np.where(close))
			return remove_nearby(others)
	return list(times)

decide_frame = None

def pg_events_forever():
	while True:
		for ev in pygame.event.get():
			yield ev

def get_user_decision(size):
	global decide_frame
	if not decide_frame:
		msg = """
		was the previous video clip a cut?
		   y = yes, n = no, r = replay
		"""
		# TODO figure out why the text rendering is really ugly
		decide_frame = mpe.TextClip(msg, 
			size=size, color='white', bg_color='black', font='Monaco',)

	decide_frame.show()
	key2char = { pygame.K_y : 'y', pygame.K_n : 'n', pygame.K_r : 'r' }
	for ev in pg_events_forever():
		if ev.type == pygame.KEYDOWN and ev.key in key2char:
			return key2char[ev.key]


def show_padded_clips(clip, times, n_repeat=1, pad_sec=1.0):
	dt = 1.0 / clip.fps
	black = mpe.ColorClip(clip.size, (0, 0, 0))
	accept = [None for _ in times]
	i = 0
	while i < len(times):
		t = times[i]
		t0, t1 = t - pad_sec, t + pad_sec
		c = clip.subclip(t0, t1)

		for _ in range(n_repeat):
			c.preview()

		u = get_user_decision(clip.size)
		if u == 'y':
			print("accepting cut")
			accept[i] = True
			i += 1
		elif u == 'n':
			print("rejecting cut")
			accept[i] = False
			i += 1
		elif u == 'r':
			print("replaying")
		else:
			assert False
	return accept

# generates objects suitable for JSON serialization
def cuts2clips(begin, end, cut_times, accept):
	clips = []
	tprev = begin
	for t, acc in zip(cut_times, accept):
		if acc:
			yield { "begin" : tprev, "end" : t }
			tprev = t
	yield { "begin" : tprev, "end" : end }

def save_results(filename, clip, cut_times, accept):
	begin = 0 # TODO get from clip
	end = clip.duration
	cuts = list(cuts2clips(begin, end, cut_times, accept))
	obj = {
		"filename" : clip.filename,
		"cuts" : cuts,
	}
	with open(filename, 'w') as f:
		json.dump(obj, f)

def main():
	argparser = argparse.ArgumentParser()
	argparser.add_argument('clip')
	args = argparser.parse_args()

	# TODO convert 60fps videos to 30fps?
	clip = mpe.VideoFileClip(args.clip, 
		target_resolution=(320, None),
		audio=False)
	print_info(clip)

	if not os.path.exists('./cut_hyps'):
		clip_lowres = mpe.VideoFileClip(
			args.clip, target_resolution=(64, None), audio=False)
		times = find_likely_cuts(clip_lowres)
		times = remove_nearby(times)
		pickle.dump(times, open('./cut_hyps', 'wb'))
	else:
		times = pickle.load(open('./cut_hyps', 'rb'))

	accept = show_padded_clips(clip, times)
	save_results('test.json', clip, times, accept)

if __name__ == '__main__':
	main()
