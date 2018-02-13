#!/usr/bin/env python

from __future__ import print_function
import argparse
import cPickle as pickle
import os
import sys
import time

from collections import namedtuple
import cv2
import itertools
import json
import numpy as np
import matplotlib.pyplot as plt
import moviepy.video.tools.tracking as tracking
import moviepy.editor as mpe
import multiprocessing
import pygame

from guieditor import interactive_editor

#
# terminology used:
# -----------------
# video - a full multi-shot video, to be split
# cut   - a single moment in time when the shot changes
# clip  - (begin, end) time boundaries of a shot
#

Clip = namedtuple('Clip', 'begin end')

# input: array of arrays, each same length, any type
# output: a table with justified columns, as array of strings
def tabular(rows):
	if len(rows) == 0:
		return
	nc = len(rows[0])
	for row in rows:
		assert len(row) == nc
	rows = [[str(x) for x in row] for row in rows]
	colmax = [0 for _ in range(nc)]
	for row in rows:
		colmax = list(itertools.imap(max, colmax, [len(x) for x in row]))
	for row in rows:
		cells = (s.ljust(cmax) for s, cmax in zip(row, colmax))
		yield " ".join(cells)

# pretty-print some basic facts about the video.
def print_info(video):
	rowstrs = list(tabular([
		["name:", video.filename],
		["size:", video.size],
		["duration:", str(video.duration) + " secs"],
		["fps:", video.fps],
	]))
	underscore_len = max(len(row) for row in rowstrs)
	print("")
	print("video info:")
	print("-" * underscore_len)
	for s in rowstrs:
		print(s)
	print("")

# random sample of N (size, size) tiles in the middle of the image
def sample_tiles(img, size, N):
	h, w, _ = img.shape
	h_range, w_range = ((int(0.25 * x), int(0.75 * x)) for x in (h, w))
	def make_tile():
		y = np.random.randint(*h_range)
		x = np.random.randint(*w_range)
		return x, y, img[y:(y+size),x:(x+size),:]
	return [make_tile() for _ in range(N)]

# TODO this can be done better. use an optical flow
# or multi-view geometric method to recognize camera movements.
def find_likely_cuts(video, n_top=None, tile_size=8, n_tiles=20, radius=15):
	dt = 1.0 / video.fps
	match_sums = []
	if n_top is None:
		# most FPV video does not contain fast cuts.
		# might need to increase n_top for other kinds of video.
		n_top = int(video.duration)
	frames = video.iter_frames(progress_bar=True)
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
def remove_nearby(cuts):
	cuts = np.array(cuts)
	for t in cuts:
		delta = cuts - t
		close = np.logical_and(np.abs(delta) < 1.0, delta != 0)
		if close.any():
			others = np.delete(cuts, np.where(close))
			return remove_nearby(others)
	return list(cuts)

decide_frame = None

def pg_events_forever():
	while True:
		for ev in pygame.event.get():
			yield ev

# display message asking user to approve or reject cut via keyboard input.
# input: resolution of video. output: 'y' - yes, 'n' - no, 'r' - replay
def get_user_decision(size):
	global decide_frame
	if not decide_frame:
		msg = """
		was the previous video clip a cut?
		   y = yes, n = no, r = replay,
		   m = mark for manual attention
		"""
		# TODO figure out why the text rendering is really slow + ugly
		decide_frame = mpe.TextClip(msg, 
			size=size, color='white', bg_color='black', font='Monaco',)

	decide_frame.show()
	key2char = { pygame.K_y : 'y', pygame.K_n : 'n', pygame.K_r : 'r' }
	for ev in pg_events_forever():
		if ev.type == pygame.KEYDOWN and ev.key in key2char:
			return key2char[ev.key]

# 
def show_padded_cuts(video, cuts, n_repeat=1, pad_sec=1.0):
	dt = 1.0 / video.fps
	black = mpe.ColorClip(video.size, (0, 0, 0))
	accept = [None for _ in cuts]
	i = 0
	while i < len(cuts):
		t = cuts[i]
		t0, t1 = t - pad_sec, t + pad_sec
		c = video.subclip(t0, t1)

		for _ in range(n_repeat):
			c.preview(fps=video.fps)

		u = get_user_decision(video.size)
		if u == 'y':
			print("accepting cut")
			accept[i] = True
			i += 1
		elif u == 'n':
			print("rejecting cut")
			accept[i] = False
			i += 1
		elif u == 'm':
			print("marking cut for manual attention")
			# TODO: do it
			i += 1
		elif u == 'r':
			print("replaying")
		else:
			assert False
	return accept

# input: clip begin&end, list of proposed cuts, user's accept/reject decisions
# yields: NamedTuples of (begin, end) defining clips
def cuts2clips(begin, end, cuts, accept):
	clips = []
	tprev = begin
	for t, acc in zip(cuts, accept):
		if acc:
			yield Clip(tprev, t)
			tprev = t
	yield Clip(tprev, end)

def approve_clips(video, clips):
	black = mpe.ColorClip(video.size, (0, 0, 0))
	accept = [True for _ in clips]
	for i, c in enumerate(clips):
		clip = video.subclip(*c)
		reject_keys = [pygame.K_DELETE, pygame.K_BACKSPACE]
		def event_handler(ev):
			if ev.type == pygame.KEYDOWN and ev.key in reject_keys:
				print("rejecting video")
				accept[i] = False
		clip.preview(fps=video.fps, event_handler=event_handler)
		black.show()
		time.sleep(1)
	return accept

def save_clips(video, clips):
	name = video.filename
	dirpath = "./{}_clips".format(name)
	if not os.path.exists(dirpath):
		os.mkdir(dirpath)
	for c in clips:
		path = "{}/{}_{:.4f}_{:.4f}.mp4".format(dirpath, name, *c)
		video.subclip(*c).write_videofile(path)

#
# JSON "schema": {
#   filename: <str>
#   proposed_cuts: [floats]
#   accept: [bools] or null
#   clips: [[t0, t1], ...] or null
#
def save_results(video, cuts, accept=None, clips=None):
	obj = {
		"filename" : video.filename,
		"proposed_cuts" : cuts,
		"accept" : accept,
		"clips" : clips,
	}
	with open(video.filename + '.json', 'w') as f:
		json.dump(obj, f)

def load_or_construct_results(video_filename):
	path = video_filename + ".json"
	if os.path.exists(path):
		with open(path) as f:
			return json.load(f), path
	results = {
		"filename" : video_filename,
		"proposed_cuts" : None,
		"accept" : None,
		"clips" : None,
	}
	return results, path

def user_cancel(msg):
	cmd = raw_input(msg + "\nenter to continue, q to exit.\n")
	return cmd in ['q', 'Q']

def main():
	argparser = argparse.ArgumentParser()
	argparser.add_argument('video')
	args = argparser.parse_args()

	# TODO convert 60fps videos to 30fps?
	video = mpe.VideoFileClip(args.video, 
		target_resolution=(320, None), audio=False)
	video_lowres = mpe.VideoFileClip(
		args.video, target_resolution=(64, None), audio=False)
	print_info(video)

	results, path = load_or_construct_results(args.video)
	if not results["proposed_cuts"]:
		if user_cancel("search for proposed cuts?"):
			return
		cuts = find_likely_cuts(video_lowres)
		cuts = sorted(remove_nearby(cuts))
		results["proposed_cuts"] = cuts
		with open(path, 'w') as f:
			json.dump(results, f)
	if not results["accept"]:
		if user_cancel("review proposed cuts?"):
			return
		cuts = results["proposed_cuts"]
		accept = show_padded_cuts(video, cuts)
		clips = list(cuts2clips(0, video.duration, cuts, accept))
		results["accept"] = accept
		results["clips"] = clips
		with open(path, 'w') as f:
			json.dump(results, f)

	clips = results["clips"]
	#if user_cancel("review clips?"):
		#return
	#keep = approve_clips(video, clips)

	#if user_cancel("save clips?"):
		#return
	#save_clips(video_lowres, clips)

	clip = clips[0]
	edit_clips = interactive_editor(video, clip)
	print("edited clips:", edit_clips)

if __name__ == '__main__':
	main()
