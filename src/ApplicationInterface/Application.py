import kivy
from kivy.clock import Clock
import os
import time as t
import threading as thr
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Color, Rectangle
from kivy.clock import Clock
import random

from src.Gwen.gwen import Gwen


class VisualizerWidget(Widget):
    def __init__(self, **kwargs):
        super(VisualizerWidget, self).__init__(**kwargs)
        self.rectangles = []

    def on_size(self, *args):
        self.rectangles.clear()
        for _ in range(10):  # replace with the number of outputs
            color = (random.random(), random.random(), random.random())
            with self.canvas:
                Color(*color)
                rect = Rectangle(pos=self.pos, size=self.size)
                self.rectangles.append(rect)

    def update_rectangles(self, dt):
        for rect in self.rectangles:
            rect.pos = self.pos
            rect.size = self.size
            rect.size = (rect.size[0], rect.size[1] * (0.5 + 0.5 * random.random()))  # randomize the height

    def start_visualization(self):
        # self.sound.play()
        Clock.schedule_interval(self.update_rectangles, 0.1)  # adjust as needed

class VisualizerApp(App):
    def build(self):
        visualizer = VisualizerWidget()
        visualizer.start_visualization()
        self.Gwen = Gwen.Gwen()
        self._app_thread = thr.Thread(target=self.backend_run, args=(0,))
        self._app_thread.start()
        return visualizer

    def backend_run(self, dt):
        while True:
            self.Gwen.run_context()
            t.sleep(0.08)