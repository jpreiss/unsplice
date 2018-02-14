from copy import deepcopy
import pygame as pg
import moviepy.editor as mpe

BAR_H = 20
BAR_COLOR = pg.Color(100, 100, 100, 255)
CLIP_COLOR = pg.Color(200, 130, 0, 255)
CURSOR_COLOR = pg.Color(255, 255, 255, 255)

PAUSE = 0
PLAY = 1
SCRUB_LEFT = 2
SCRUB_RIGHT = 3
SCRUB_MAX_SPEED = 4.0

ED_ACCEPT = 0
ED_REJECT = 1
ED_CONTINUE = 2

# keyboard driven editor for speed.
class Editor(object):

	def __init__(self, imgseq, fps, clip, screen):
		self.screen = screen
		self.imgseq = imgseq
		self.domain = deepcopy(clip)
		self.clip = deepcopy(clip)
		self.fps = fps
		self.cursor = clip[0]

		w, h = screen.get_size()
		vw, vh, _ = imgseq[0].shape
		self.bar_top = h - BAR_H
		x_scale = w / float(clip[1] - clip[0])
		self.t2x = lambda t : x_scale * (t - clip[0])
		self.winsz = (w, h)

		self.state = PAUSE

	def cursor_clip(self):
		if self.cursor < self.domain[0]:
			self.cursor = self.domain[0]
		elif self.cursor >= self.domain[1]:
			self.cursor = self.domain[1] - 1

	def step(self, dt):
		frames = dt * self.fps
		if self.state == PAUSE:
			pass
		if self.state == PLAY:
			self.cursor += frames
		if self.state in [SCRUB_LEFT, SCRUB_RIGHT]:
			self.scrub_speed = min(self.scrub_speed * 1.05, SCRUB_MAX_SPEED)
			factor = self.scrub_speed * (-1 if self.state == SCRUB_LEFT else 1)
			self.cursor += factor * frames
		self.cursor_clip()

	def draw(self):
		frame = self.imgseq[int(self.cursor)]
		surface = pg.surfarray.make_surface(frame)
		r = pg.transform.rotate(surface, -90)
		f = pg.transform.flip(r, True, False)
		s = pg.transform.scale(f, self.winsz)
		self.screen.blit(s, (0, 0))
		self.screen.fill(BAR_COLOR, self.rect(*self.domain))
		self.screen.fill(CLIP_COLOR, self.rect(*self.clip))
		self.screen.fill(CURSOR_COLOR, self.cursor_rect())

	def cursor_rect(self):
		x = self.t2x(self.cursor)
		return pg.Rect(x - 2, self.bar_top, 4, BAR_H)

	def rect(self, t0, t1):
		x = self.t2x(t0)
		w = self.t2x(t1) - x
		return pg.Rect(x, self.bar_top, w, BAR_H)

	def keyscan(self):
		keys = pg.key.get_pressed()
		if keys[pg.K_LEFT] and keys[pg.K_RIGHT]:
			self.state = PAUSE
		elif keys[pg.K_LEFT]:
			self.state = SCRUB_LEFT
			self.scrub_speed = 1.0
		elif keys[pg.K_RIGHT]:
			self.state = SCRUB_RIGHT
			self.scrub_speed = 1.0
		else:
			self.state = PAUSE

	def keyevent(self, key):
		if key == pg.K_RIGHTBRACKET:
			self.clip[1] = int(self.cursor + 1)
		if key == pg.K_LEFTBRACKET:
			self.clip[0] = int(self.cursor)
		if key == pg.K_COMMA:
			self.cursor = int(self.cursor) - 1
			self.cursor_clip()
		if key == pg.K_PERIOD:
			self.cursor = int(self.cursor) + 1
			self.cursor_clip()
		if key == pg.K_b:
			self.cursor = self.clip[0]
		if key == pg.K_e:
			self.cursor = self.clip[1]
		if key == pg.K_SPACE:
			self.state = PLAY

		return ED_CONTINUE


class EditorTree(object):
	def __init__(self, editor):
		self.ed = editor
		self.left = self.right = None
		self.active = self.ed

	def step(self, dt):
		self.active.step(dt)

	def draw(self):
		self.active.draw()

	def keyscan(self):
		self.active.keyscan()

	def get_clips(self, arr):
		if self.left is not None:
			print("recursing left")
			self.left.get_clips(arr)
		if self.right is not None:
			print("recursing right")
			self.right.get_clips(arr)
		if (self.left, self.right) == (None, None):
			arr.append(self.ed.clip)

	def keyevent(self, key):
		ret = self.active.keyevent(key)

		if self.active is self.left:
			if ret == ED_REJECT:
				self.left = None
			if ret in [ED_ACCEPT, ED_REJECT]:
				self.active = self.right

		elif self.active is self.right:
			if ret == ED_REJECT:
				self.right = None
				self.active = None
				return ED_ACCEPT if self.left is not None else ED_REJECT
			if ret == ED_ACCEPT:
				self.active = None
				return ED_ACCEPT

		else:
			assert self.active == self.ed
			# split
			if key == pg.K_s:
				clip0 = [self.ed.clip[0], self.ed.cursor]
				clip1 = [self.ed.cursor, self.ed.clip[1]]
				self.left = EditorTree(Editor(self.ed.imgseq, self.ed.fps, clip0, self.ed.screen))
				self.right = EditorTree(Editor(self.ed.imgseq, self.ed.fps, clip1, self.ed.screen))
				self.active = self.left
				return ED_CONTINUE

			if key == pg.K_RETURN:
				return ED_ACCEPT

			if key == pg.K_BACKSPACE:
				return ED_REJECT

		return ED_CONTINUE

def interactive_editor(video, clip):
	ratio = float(video.size[0]) / video.size[1]
	h = 600
	w = int(ratio * h)
	screen = pg.display.set_mode((w, h), 
		pg.DOUBLEBUF | pg.HWSURFACE | pg.FULLSCREEN)

	imgseq = list(video.subclip(*clip).iter_frames())
	ed = Editor(imgseq, video.fps, [0, len(imgseq)], screen)
	root = EditorTree(ed)

	fps = video.fps/2
	running = True
	clock = pg.time.Clock()
	clips = []

	while running:
		dt = clock.tick(fps) / 1000.0
		root.keyscan()
		root.step(dt)
		root.draw()
		pg.display.flip()

		for event in pg.event.get():
			if event.type == pg.QUIT:
				running = False   
			if event.type == pg.KEYDOWN:
				ret = root.keyevent(event.key)
				if ret == ED_ACCEPT:
					root.get_clips(clips)
					running = False
				elif ret == ED_REJECT:
					running = False
	
	frame2time = lambda f: clip[0] + (1.0 / video.fps) * f
	clip2time = lambda c: (frame2time(c[0]), frame2time(c[1]))
	print("real end:", clip[1])
	print("approx end:", frame2time(len(imgseq)))
	return [clip2time(c) for c in clips]
