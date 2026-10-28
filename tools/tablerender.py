"""THE TABLE (act 1, chance): dice and chips on a dark table, ray-marched with ModernGL (signed distance fields), the
brand's 3D way (channel/channel/tools/brand3d.py: soft key light up-left, cool rim, contact shading, AgX tone map,
light bloom). Z is up, the table top is z = 0, one die is 1 unit wide.

    import tablerender as tr
    sc = tr.Table()
    img, info = sc.render(eye, tgt, focal, dice=[(x, y, top_value, yaw_deg), ...])      # mode "dice"
    img, info = sc.render(eye, tgt, focal, chips=counts, lit_col=i)                       # mode "chips"

Dice are real dice (opposite faces sum to 7: +z 1, +y 2, +x 3, -x 4, -y 5, -z 6 in the die's own frame), turned so
the thrown value faces up. Chips: one chip per throw, stacked over its total (column pitch 1 unit).
"""
import math
import os
import sys

import moderngl
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "channel", "channel", "tools")))
from brand3d import agx, bloom, lin                                       # noqa: E402

MAXD, MAXC = 16, 64
CHIP_T = 0.08                                                             # chip pitch (thickness incl. gap)

QUAD_VS = "#version 330\nin vec2 v; void main(){ gl_Position = vec4(v, 0.0, 1.0); }"

FS = """#version 330
uniform vec2 res; uniform vec3 eye; uniform vec3 tgt; uniform float focal; uniform vec2 shift;
uniform int mode;                                   // 0 dice, 1 chips, 2 nothing but the page, 3 wave channel
uniform float chL; uniform float chW; uniform float chD; uniform sampler2D etaTex; uniform vec3 waterc; uniform float chWalls;
const float FLOOR_T = 0.06;
float eta(float x){ return texture(etaTex, vec2(clamp((x + chL * 0.5) / chL, 0.0, 1.0), 0.5)).r; }
uniform float pageOn; uniform vec3 pageC; uniform vec2 pageS; uniform vec2 pageR; uniform sampler2D pageTex;
uniform int nD; uniform vec4 dC[16]; uniform mat3 dM[16];   // die centre (w = half size), world -> die rotation
uniform int nC; uniform float cnt[64]; uniform float x0; uniform int litC; uniform float chipT;
uniform vec3 bg0; uniform vec3 bg1; uniform vec2 glowC; uniform vec3 diec; uniform vec3 pipc; uniform vec3 chipc;
uniform vec3 litc; uniform vec3 tablec; uniform float keyI; uniform float tableK;
out vec4 o;

float rbox(vec3 p, vec3 b, float r){ vec3 q = abs(p) - (b - r); return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r; }
float rcyl(vec3 p, float R, float h, float r){
  vec2 d = vec2(length(p.xy) - (R - r), abs(p.z) - (h - r));
  return min(max(d.x, d.y), 0.0) + length(max(d, 0.0)) - r;
}
float hash(vec2 p){ return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

// pips: dimples on the face the point is nearest to (value from the die's own frame)
float pips(vec3 q, float h){
  vec3 a = abs(q); int ax; float s;
  if (a.x > a.y && a.x > a.z){ ax = 0; s = sign(q.x); } else if (a.y > a.z){ ax = 1; s = sign(q.y); } else { ax = 2; s = sign(q.z); }
  int v = ax == 2 ? (s > 0.0 ? 1 : 6) : (ax == 1 ? (s > 0.0 ? 2 : 5) : (s > 0.0 ? 3 : 4));
  vec2 uv = (ax == 0 ? q.yz : (ax == 1 ? q.xz : q.xy)) / h;
  float n = (ax == 0 ? a.x : (ax == 1 ? a.y : a.z)) / h;
  float sp = 0.5, r = 0.2, c0 = 1.0 + 0.12;
  float d = 1e3;
  vec2 P[6]; int k = 0;
  if (v == 1){ P[0] = vec2(0.0); k = 1; }
  else if (v == 2){ P[0] = vec2(-sp, -sp); P[1] = vec2(sp, sp); k = 2; }
  else if (v == 3){ P[0] = vec2(-sp, -sp); P[1] = vec2(0.0); P[2] = vec2(sp, sp); k = 3; }
  else if (v == 4){ P[0] = vec2(-sp, -sp); P[1] = vec2(sp, sp); P[2] = vec2(-sp, sp); P[3] = vec2(sp, -sp); k = 4; }
  else if (v == 5){ P[0] = vec2(-sp, -sp); P[1] = vec2(sp, sp); P[2] = vec2(-sp, sp); P[3] = vec2(sp, -sp); P[4] = vec2(0.0); k = 5; }
  else { P[0] = vec2(-sp, -sp); P[1] = vec2(-sp, 0.0); P[2] = vec2(-sp, sp); P[3] = vec2(sp, -sp); P[4] = vec2(sp, 0.0); P[5] = vec2(sp, sp); k = 6; }
  for (int i = 0; i < 6; i++){ if (i >= k) break; d = min(d, length(vec3(uv - P[i], n - c0)) - r); }
  return d * h;
}

vec2 pageLocal(vec3 p){ vec2 d = p.xy - pageC.xy; return vec2(pageR.x * d.x + pageR.y * d.y, -pageR.y * d.x + pageR.x * d.y); }
// returns dist, material (1 die, 2 pip, 3 chip, 4 lit chip, 5 table, 6 page, 7 water, 8 tank), id
vec3 map(vec3 p){
  vec3 res = vec3(p.z, 5.0, 0.0);
  if (pageOn > 0.5){
    vec2 l = pageLocal(p);
    float dp = rbox(vec3(l, p.z - pageC.z), vec3(pageS, 0.006), 0.003);
    if (dp < res.x) res = vec3(dp, 6.0, 0.0);
  }
  if (mode == 3){                              // a lab wave channel: floor, back wall, posts; the front is glass
    float fl = rbox(p - vec3(0.0, 0.0, FLOOR_T * 0.5), vec3(chL * 0.5 + 0.15, chW * 0.5 + 0.08, FLOOR_T * 0.5), 0.01);
    float bw = rbox(p - vec3(0.0, chW * 0.5 + 0.04, (chD + 0.4) * 0.5), vec3(chL * 0.5 + 0.15, 0.04, (chD + 0.4) * 0.5), 0.01);
    float px = mod(p.x + 1.0, 2.0) - 1.0;
    float po = rbox(vec3(px, p.y + chW * 0.5 + 0.05, p.z - (chD + 0.4) * 0.5), vec3(0.025, 0.025, (chD + 0.4) * 0.5), 0.006);
    po = max(po, abs(p.x) - chL * 0.5 - 0.15);
    float tk = chWalls > 0.5 ? min(fl, min(bw, po)) : 1e3;
    if (tk < res.x) res = vec3(tk, 8.0, 0.0);
    float box = rbox(p - vec3(0.0, 0.0, FLOOR_T + (chD + 0.3) * 0.5), vec3(chL * 0.5, chW * 0.5, (chD + 0.3) * 0.5), 0.0);
    float dw = max(box, (p.z - (FLOOR_T + chD + eta(p.x))) * 0.7);
    if (dw < res.x) res = vec3(dw, 7.0, 0.0);
  }
  if (mode == 0){
    for (int i = 0; i < 16; i++){
      if (i >= nD) break;
      vec3 q = dM[i] * (p - dC[i].xyz); float h = dC[i].w;
      float db = length(p - dC[i].xyz) - 1.8 * h;
      if (db > res.x) continue;
      float d = rbox(q, vec3(h), 0.17 * h);
      float dp = pips(q, h);
      float dd = max(d, -dp);
      if (dd < res.x) res = vec3(dd, dp < 0.004 * h ? 2.0 : 1.0, float(i));
    }
  } else {
    float ci = clamp(floor(p.x - x0 + 0.5), 0.0, float(nC - 1));
    for (int s = -1; s <= 1; s++){
      float c = clamp(ci + float(s), 0.0, float(nC - 1));
      float n = cnt[int(c)];
      if (n < 0.5) continue;
      float k = clamp(floor(p.z / chipT), 0.0, n - 1.0);
      vec2 j = (vec2(hash(vec2(c, k)), hash(vec2(k, c + 7.0))) - 0.5) * 0.06;     // a hand-stacked wobble
      vec3 q = p - vec3(x0 + c + j.x, j.y, (k + 0.5) * chipT);
      float d = rcyl(q, 0.42, chipT * 0.46, 0.018);
      float top = (int(c) == litC && k > n - 1.5) ? 4.0 : 3.0;
      if (d < res.x) res = vec3(d, top, c);
    }
  }
  return res;
}

vec3 nrm(vec3 p){
  vec2 e = vec2(0.0006, 0.0);
  return normalize(vec3(map(p + e.xyy).x - map(p - e.xyy).x, map(p + e.yxy).x - map(p - e.yxy).x, map(p + e.yyx).x - map(p - e.yyx).x));
}
vec3 march(vec3 ro, vec3 rd, float tmax){
  float t = 0.0;
  for (int i = 0; i < 260; i++){
    vec3 h = map(ro + rd * t);
    if (h.x < 0.0003 * t) return vec3(t, h.yz);
    t += h.x * 0.85;
    if (t > tmax) break;
  }
  return vec3(-1.0, 0.0, 0.0);
}
float objs(vec3 p){ vec3 h = map(p); return h.y > 4.5 ? 1e3 : h.x; }
float shadow(vec3 ro, vec3 rd){
  float s = 1.0, t = 0.02;
  for (int i = 0; i < 80; i++){
    float h = objs(ro + rd * t);
    s = min(s, 8.0 * h / t);
    t += clamp(h, 0.01, 0.5);
    if (s < 0.002 || t > 40.0) break;
  }
  return clamp(s, 0.0, 1.0);
}
float ao(vec3 p, vec3 n){
  float o = 0.0, w = 1.0;
  for (int i = 1; i <= 5; i++){ float h = 0.06 * float(i); o += w * (h - map(p + n * h).x); w *= 0.6; }
  return clamp(1.0 - 2.4 * o, 0.0, 1.0);
}
uniform vec3 farDir; uniform float farI;
vec3 env(vec3 d){              // a dark studio: one big soft box up-left, a faint cool strip right (+ the channel's far light)
  vec3 sb = normalize(vec3(-0.55, -0.35, 0.76));
  float box = smoothstep(0.86, 0.96, dot(d, sb));
  vec3 rb = normalize(vec3(0.9, 0.35, 0.15));
  float strip = smoothstep(0.93, 0.99, dot(d, rb));
  float far = farI > 0.0 ? smoothstep(0.90, 0.995, dot(d, farDir)) : 0.0;
  return bg0 * 0.6 + vec3(1.0, 0.94, 0.86) * box * 1.4 + vec3(0.55, 0.75, 1.0) * strip * 0.4 + vec3(0.85, 0.92, 1.0) * far * farI;
}
vec3 background(vec2 frag){
  vec2 p = (frag / res - glowC) * vec2(res.x / res.y, 1.0);
  return bg0 + bg1 * exp(-dot(p, p) * 2.0);
}
uniform vec3 litP;
vec3 shade(vec3 p, vec3 rd, vec3 hit){
  vec3 n = nrm(p); float mat = hit.y;
  vec3 alb; float rough, spec; vec3 emit = vec3(0.0);
  if (mat < 1.5){ alb = diec; rough = 0.35; spec = 0.45; }
  else if (mat < 2.5){ alb = pipc; rough = 0.5; spec = 0.2; }
  else if (mat < 3.5){ alb = chipc; rough = 0.4; spec = 0.35; }
  else if (mat < 4.5){ alb = litc * 0.5; rough = 0.3; spec = 0.4; emit = litc * 1.6; }
  else if (mat < 5.5){ alb = tablec * tableK; rough = 0.3; spec = 0.5 * tableK; }
  else if (mat > 7.5){ alb = vec3(0.020, 0.018, 0.024); rough = 0.35; spec = 0.5; }
  else if (mat > 6.5){                          // water: lit from above, glowing faintly through its side
    float depthK = smoothstep(FLOOR_T, FLOOR_T + chD, p.z);
    vec3 vol = waterc * (0.10 + 0.90 * depthK * depthK);
    if (n.z > 0.45){ alb = vol * 0.35; rough = 0.08; spec = 1.0; emit = vol * 0.10; }
    else { alb = vol * 0.4; rough = 0.5; spec = 0.25; emit = vol * 0.55; }
  }
  else {                                        // paper: the page image, slightly warm, matte
    vec2 uv = pageLocal(p) / (2.0 * pageS) + 0.5;
    alb = pow(texture(pageTex, vec2(uv.x, 1.0 - uv.y)).rgb, vec3(2.2)) * vec3(0.80, 0.76, 0.70); rough = 0.75; spec = 0.12;
  }
  vec3 key = normalize(vec3(-0.55, -0.35, 0.76));
  vec3 rim = normalize(vec3(0.75, 0.6, 0.25));
  float occ = ao(p, n);
  float sh = shadow(p + n * 0.004, key);
  float kd = max(dot(n, key), 0.0) * sh;
  vec3 h = normalize(key - rd);
  float sp = pow(max(dot(n, h), 0.0), mix(160.0, 12.0, rough)) * sh * spec;
  float fr = 0.04 + 0.96 * pow(1.0 - max(dot(n, -rd), 0.0), 5.0);
  vec3 col = alb * (keyI * kd * vec3(1.0, 0.93, 0.84) + 0.10 * occ)
           + keyI * sp * 1.1
           + env(reflect(rd, n)) * fr * spec * occ * (1.0 - rough * 0.6)
           + vec3(0.35, 0.5, 0.75) * pow(max(dot(n, rim), 0.0), 2.0) * 0.14 * occ;
  if (litC >= 0){                               // light spilled by the lit chip
    vec3 lv = litP - p; float d = length(lv);
    if (d > 0.3) col += alb * litc * max(dot(n, lv / d), 0.0) * 1.2 / (1.0 + 3.0 * d * d) * occ;
  }
  return col + emit;
}
void main(){
  vec2 frag = gl_FragCoord.xy;
  vec2 ndc = (2.0 * frag - res) / res.y - shift;
  vec3 fw = normalize(tgt - eye), rt = normalize(cross(fw, vec3(0.0, 0.0, 1.0))), up = cross(rt, fw);
  vec3 rd = normalize(fw * focal + rt * ndc.x + up * ndc.y);
  vec3 col = background(frag);
  vec3 hit = march(eye, rd, 200.0);
  if (hit.x > 0.0){
    vec3 p = eye + rd * hit.x;
    vec3 c = shade(p, rd, hit);
    if (hit.y > 4.5 && hit.y < 5.5){             // the table: soft gloss reflection, fading into the night
      vec3 n = vec3(0.0, 0.0, 1.0); vec3 r = reflect(rd, n);
      vec3 h2 = march(p + n * 0.01, r, 60.0);
      vec3 rc = h2.x > 0.0 && h2.y < 4.5 ? shade(p + n * 0.01 + r * h2.x, r, h2) : vec3(0.0);
      float f = (0.05 + 0.35 * pow(1.0 - max(dot(n, -rd), 0.0), 5.0)) * tableK;
      float fade = exp(-0.0025 * dot(p.xy - tgt.xy, p.xy - tgt.xy));
      c = mix(col, c + rc * f, fade);
    }
    col = c;
  }
  if (mode == 3 && chWalls > 0.5){             // the front glass: a faint reflection and a bright top edge
    float yg = -chW * 0.5 - 0.012;
    float tg = (yg - eye.y) / rd.y;
    vec3 pg = eye + rd * tg;
    if (tg > 0.0 && (hit.x < 0.0 || tg < hit.x) && abs(pg.x) < chL * 0.5 + 0.15 && pg.z > 0.0 && pg.z < chD + 0.4){
      vec3 n = vec3(0.0, -1.0, 0.0);
      float fr = 0.04 + 0.96 * pow(1.0 - abs(dot(n, rd)), 5.0);
      col = col * 0.92 + env(reflect(rd, n)) * fr * 0.5;
      col += vec3(0.8, 0.9, 1.0) * 0.5 * exp(-pow((chD + 0.4 - pg.z) / 0.004, 2.0));
    }
  }
  o = vec4(col, 1.0);
}
"""


def rot_top(v, yaw_deg):
    """World -> die rotation for a die resting with value v on top, turned by yaw about z."""
    a = math.radians(yaw_deg)
    Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])
    q = math.pi / 2

    def rx(t):
        return np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])

    def ry(t):
        return np.array([[math.cos(t), 0, math.sin(t)], [0, 1, 0], [-math.sin(t), 0, math.cos(t)]])
    face = {1: np.eye(3), 6: rx(math.pi), 2: rx(q), 5: rx(-q), 3: ry(-q), 4: ry(q)}[v]
    R = Rz @ face                                                         # die -> world
    assert np.allclose(R @ {1: [0, 0, 1], 6: [0, 0, -1], 2: [0, 1, 0], 5: [0, -1, 0], 3: [1, 0, 0], 4: [-1, 0, 0]}[v],
                       [0, 0, 1], atol=1e-9)
    return R.T


class Table:
    def __init__(self, w=1920, h=1080, ss=2):
        self.w, self.h, self.ss = w, h, ss
        self.W, self.H = w * ss, h * ss
        ctx = self.ctx = moderngl.create_standalone_context()
        self.prog = ctx.program(vertex_shader=QUAD_VS, fragment_shader=FS)
        quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], "f4").tobytes())
        self.vao = ctx.vertex_array(self.prog, [(quad, "2f", "v")])
        self.fbo = ctx.framebuffer([ctx.texture((self.W, self.H), 4, dtype="f4")])

    def set(self, name, val):
        if name in self.prog:
            self.prog[name].value = val

    def render(self, eye, tgt, focal, dice=None, chips=None, x0=0.0, lit_col=-1, shift=(0.0, 0.0),
               glow_c=(0.55, 0.6), exposure=1.0, bloom_k=0.05, grain=0.008, page=None, channel=None, table_k=1.0,
               ground="#14111A", wall="#3B3643", key=1.0):
        """page: dict(image=PIL image, center=(x, y), width=units, yaw=deg): a sheet of paper lying on the table.
        channel: dict(eta=array (surface height along the channel, units), length, width, depth): a wave tank.
        ground / wall: the video ground and the table's base colour (defaults: the brand night and night wall).
        key: the key light on the table and paper (1.0 as approved; lower = a dimmer lamp, the ground unchanged)."""
        eye, tgt = np.asarray(eye, float), np.asarray(tgt, float)
        self.set("res", (float(self.W), float(self.H)))
        self.set("eye", tuple(map(float, eye)))
        self.set("tgt", tuple(map(float, tgt)))
        self.set("focal", float(focal))
        self.set("shift", tuple(map(float, shift)))
        self.set("bg0", tuple(lin(ground)))
        self.set("bg1", tuple(0.05 * a + 0.05 * b for a, b in zip(lin("#2D697B"), lin("#543C52"))))
        self.set("glowC", tuple(map(float, glow_c)))
        self.set("diec", tuple(c * 0.85 for c in lin("#FCE2CA")))       # cream = chance (STORYBOARD §1)
        self.set("pipc", tuple(lin("#14111A")))
        self.set("chipc", tuple(c * 0.8 for c in lin("#FCE2CA")))
        self.set("litc", tuple(lin("#71DDEE")))                          # cyan = "look here"
        self.set("tablec", tuple(c * 0.55 for c in lin(wall)))
        self.set("keyI", float(key))
        self.set("tableK", float(table_k))
        info = {}
        self.set("mode", 2)
        self.set("litC", -1)
        self.set("nD", 0)
        self.set("pageOn", 0.0)
        self.set("farI", 0.0)
        if page is not None:
            im = page["image"].convert("RGB")
            if getattr(self, "_page_src", None) is not im:
                self.page_tex = self.ctx.texture(im.size, 3, im.tobytes())
                self.page_tex.build_mipmaps()
                self.page_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
                self.page_tex.anisotropy = 16.0
                self._page_src = im
            self.page_tex.use(0)
            self.set("pageTex", 0)
            w_ = float(page["width"])
            h_ = w_ * im.size[1] / im.size[0]
            a = math.radians(page.get("yaw", 0.0))
            self.set("pageOn", 1.0)
            self.set("pageC", (float(page["center"][0]), float(page["center"][1]), 0.006))
            self.set("pageS", (w_ / 2, h_ / 2))
            self.set("pageR", (math.cos(a), math.sin(a)))
            self.page = dict(center=page["center"], half=(w_ / 2, h_ / 2), yaw=a, size=im.size)
        if channel is not None:
            e = np.asarray(channel["eta"], "f4")
            self.eta_tex = self.ctx.texture((len(e), 1), 1, e.tobytes(), dtype="f4")
            self.eta_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.eta_tex.repeat_x = False
            self.eta_tex.use(1)
            self.set("etaTex", 1)
            self.set("mode", 3)
            for k_ in ("length", "width", "depth"):
                self.set({"length": "chL", "width": "chW", "depth": "chD"}[k_], float(channel[k_]))
            self.set("waterc", tuple(c * 1.0 for c in lin("#79B8CE")))     # steel blue = the trick (STORYBOARD §1)
            fd = np.array(channel.get("far_dir", (1.0, 0.0, 0.12)), float)
            self.set("farDir", tuple(fd / np.linalg.norm(fd)))
            self.set("farI", float(channel.get("far_i", 1.6)))
            self.set("chWalls", 1.0 if channel.get("walls", True) else 0.0)
        if dice is not None:
            self.set("mode", 0)
            self.set("nD", len(dice))
            C = [(float(x), float(y), 0.5, 0.5) for x, y, v, yaw in dice] + [(0.0, 0.0, -9.0, 0.5)] * (MAXD - len(dice))
            M = [tuple(rot_top(int(v), yaw).T.ravel()) for x, y, v, yaw in dice] + [tuple(np.eye(3).ravel())] * (MAXD - len(dice))
            self.set("dC", C)
            self.set("dM", M)                                                # GLSL column-major: pass R.T rows
            self.set("litC", -1)
        if chips is not None:
            cnt = list(map(float, chips)) + [0.0] * (MAXC - len(chips))
            self.set("mode", 1)
            self.set("nC", len(chips))
            self.set("cnt", cnt)
            self.set("x0", float(x0))
            self.set("chipT", CHIP_T)
            self.set("litC", int(lit_col))
            if lit_col >= 0:
                lp = (x0 + lit_col, 0.0, (chips[lit_col] - 0.5) * CHIP_T)
                self.set("litP", tuple(map(float, lp)))
                info["lit_xyz"] = lp
        self.fbo.use()
        self.ctx.viewport = (0, 0, self.W, self.H)
        self.fbo.clear()
        T = 256                                                              # tiles keep each draw short (GPU watchdog)
        for y in range(0, self.H, T):
            for x in range(0, self.W, T):
                self.ctx.viewport = (x, y, min(T, self.W - x), min(T, self.H - y))
                self.vao.render(moderngl.TRIANGLE_STRIP)
                self.ctx.finish()
        a = np.frombuffer(self.fbo.read(viewport=(0, 0, self.W, self.H), components=4, dtype="f4"), "f4")
        a = a.reshape(self.H, self.W, 4)[::-1].reshape(self.h, self.ss, self.w, self.ss, 4).mean((1, 3))[..., :3]
        img = agx(bloom(a, bloom_k), exposure, sat=1.1)
        if grain:
            img = np.clip(img + np.random.default_rng(11).normal(0, grain, img.shape[:2] + (1,)), 0, 1)
        self.cam = (eye, tgt, focal, shift)
        return Image.fromarray((img * 255 + 0.5).astype(np.uint8)), info

    def page_point(self, u, v):
        """Page image pixel (u right, v down) -> world point on the paper."""
        pg = self.page
        lx = (u / pg["size"][0] - 0.5) * 2 * pg["half"][0]
        ly = (0.5 - v / pg["size"][1]) * 2 * pg["half"][1]
        c, s_ = math.cos(pg["yaw"]), math.sin(pg["yaw"])
        return (pg["center"][0] + c * lx - s_ * ly, pg["center"][1] + s_ * lx + c * ly, 0.013)

    def screen(self, p):
        """World point -> pixel (x, y) in the output image, for labels beside their object."""
        eye, tgt, focal, shift = self.cam
        fw = (tgt - eye) / np.linalg.norm(tgt - eye)
        rt = np.cross(fw, [0, 0, 1.0])
        rt /= np.linalg.norm(rt)
        up = np.cross(rt, fw)
        d = np.asarray(p, float) - eye
        if d @ fw <= 1e-6:
            return None                                                      # behind the camera
        x, y = d @ rt / (d @ fw) * focal + shift[0], d @ up / (d @ fw) * focal + shift[1]
        return float((x * self.h + self.w) / 2), float((self.h - y * self.h) / 2)
