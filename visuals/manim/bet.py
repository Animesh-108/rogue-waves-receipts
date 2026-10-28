"""The bet (C4, new form): my prediction drawn on the empty chart BEFORE any buoy data is read.

    cd rogue_waves/visuals/manim
    py -3.12 -m manim render bet.py Bet --resolution 1920,1080 --frame_rate 30 --media_dir ../manim_media

Clean version (2026-10-17, your note "too many things not clean"): one curve, one line, one dot, the bet, a stamp.
Numbers come from the pre-registration (PRODUCTION_LOG.md), never from data: the dice line exp(-2k^2) and the bet
band 4e-5 .. 1e-4 at k = 2. STAMP (bet_stamp.json) is written at the freeze with the pre-registration's SHA-256.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "../../../channel/channel/motion_kit/manim")))
from t108 import *                                                   # noqa: E402,F401,F403
from manim import (Scene, Axes, VGroup, VMobject, DashedLine, Dot, Create, Write, FadeIn,  # noqa: E402
                   LaggedStart, DOWN, LEFT, RIGHT, UP, DR, rate_functions)
import numpy as np                                                   # noqa: E402

_stamp = os.path.join(HERE, "bet_stamp.json")
STAMP = json.load(open(_stamp)) if os.path.exists(_stamp) else {"when": "", "sha": ""}
BET_LO, BET_HI = 4e-5, 1e-4                                          # pre-registration, "The bet"
DICE_2 = math.exp(-8)                                                # exp(-2k^2) at k = 2


def one_in(p):
    return f"1 in {round(1 / p):,}"


Y0 = 6                                                               # chart floor at 1 in a million (10^-6)


def ly(k):                                                           # log10 of the dice line, lifted so the x axis sits at the floor
    return -2 * k * k / math.log(10) + Y0


class Bet(Scene):
    def construct(self):
        ax = Axes(x_range=[1.0, 2.5, 0.5], y_range=[0, 6, 1], x_length=11.0, y_length=5.6,
                  axis_config=dict(color=C["wall"], stroke_width=3, include_ticks=False), tips=False)
        ax.shift(0.3 * DOWN)
        xl = VGroup(*[text(s, "min", "cream").scale(0.7).next_to(ax.c2p(v, 0), DOWN, buff=0.15)
                      for v, s in ((1.0, "1×"), (2.0, "2×"))])
        self.play(Create(ax), FadeIn(xl), run_time=1.0)

        dice = ax.plot(ly, x_range=[1.0, 2.5], color=C["cream"], stroke_width=6)
        dlab = text("chance alone", "label", "cream").next_to(ax.c2p(1.35, ly(1.35)), UP + RIGHT, 0.2)
        self.play(Create(dice), run_time=2.0, rate_func=rate_functions.ease_in_out_sine)
        self.play(FadeIn(dlab), run_time=0.6)

        rogue = DashedLine(ax.c2p(2.0, 0), ax.c2p(2.0, Y0), color=C["cyan"], stroke_width=3, dash_length=0.12)
        rlab = text("monster", "min", "cyan").scale(0.8).next_to(ax.c2p(2.0, Y0 - 0.35), RIGHT, 0.15)
        self.play(Create(rogue), FadeIn(rlab), run_time=0.8)

        dot = Dot(ax.c2p(2.0, math.log10(DICE_2) + Y0), radius=0.1, color=C["cream"])
        dval = text(one_in(DICE_2), "min", "cream").scale(0.8).next_to(dot, RIGHT, 0.2)
        self.play(FadeIn(dot, scale=0.5), FadeIn(dval), run_time=0.7)
        self.wait(0.5)

        rng = np.random.default_rng(17)                              # fixed seed: the same "hand" every render
        lo, hi = math.log10(BET_LO) + Y0, math.log10(BET_HI) + Y0

        def pen(points):
            m = VMobject(color=C["orange"], stroke_width=8)
            m.set_points_smoothly([ax.c2p(x, y) + np.array([rng.normal(0, 0.01), rng.normal(0, 0.01), 0])
                                   for x, y in points])
            return m
        w = 0.028
        bracket = VGroup(pen([(2.0 - w, hi), (2.0, hi + 0.01), (2.0 + w, hi)]),
                         pen([(2.0, hi), (2.002, (hi + lo) / 2), (2.0, lo)]),
                         pen([(2.0 - w, lo), (2.0, lo - 0.01), (2.0 + w, lo)]))
        bet = VGroup(text("my bet", "label", "orange"),
                     text(f"{one_in(BET_HI)} to {one_in(BET_LO)}", "min", "orange").scale(0.75))
        bet.arrange(DOWN, aligned_edge=RIGHT, buff=0.08).next_to(ax.c2p(1.96, (hi + lo) / 2), LEFT, 0.15)
        self.play(LaggedStart(*[Create(s) for s in bracket], lag_ratio=0.6), run_time=1.6)
        self.play(Write(bet), run_time=0.9)
        self.wait(0.8)

        s = (f"frozen {STAMP['when']}  ·  sha256 {STAMP['sha'][:12]}" if STAMP["sha"]
             else "draft  ·  not frozen  ·  no data read")
        stamp = text(s, "min", "wall" if STAMP["sha"] else "rose", mono=True).scale(0.45).to_corner(DR, buff=0.25)
        self.play(FadeIn(stamp), run_time=0.6)
        self.wait(2.5)
