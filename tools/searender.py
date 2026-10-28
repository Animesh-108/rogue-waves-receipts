"""Sea renderer for the rogue-wave key frames and renders (P6 onward). Grown from packaging/p0/mock_art.py.

    import searender as sr
    sea = sr.chance_sea()                           # the P0 seed search: seed 79 of 300, cos^3 spreading
    img, info = sr.render(sea, dict(dist=..., eye_h=..., fov=...), grade="storm", buoy=dict(r=0.45))

Heights are the linear random sea itself (JONSWAP, Hs 12 m, Tp 12 s, gamma 3.3): waves added up with random
phases, nothing else. Look only, never changing a height: the "choppy" sideways shift of FFT oceans (sharper
crests), tiled wind chop, foam (folds, steep crests, wind streaks), sky clouds, grain, vignette.
`point_record` gives what a buoy at one point records from the same waves, evolved in time (linear dispersion).
"""
import json
import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
EP = os.path.dirname(HERE)
G = 9.81
HS, TP, GAMMA = 12.0, 12.0, 3.3
W, H = 1920, 1080


def jonswap(w, tp=TP):
    wp = 2 * np.pi / tp
    s = np.where(w <= wp, 0.07, 0.09)
    r = np.exp(-((w - wp) ** 2) / (2 * s ** 2 * wp ** 2))
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        S = G ** 2 * w ** -5.0 * np.exp(-1.25 * (wp / w) ** 4) * GAMMA ** r
    return np.nan_to_num(S)


def spectrum_grid(N, L, tp=TP, spread=10):
    k1 = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    kx, kz = np.meshgrid(k1, k1, indexing="xy")
    k = np.hypot(kx, kz)
    th = np.arctan2(kz, kx)
    w = np.sqrt(G * k)
    with np.errstate(divide="ignore", invalid="ignore"):
        F = jonswap(w, tp) * G / (2 * np.maximum(w, 1e-9)) / np.maximum(k, 1e-9)
    D = np.where(np.abs(th) < np.pi / 2, np.cos(th) ** spread, 0.0)
    amp = np.sqrt(np.nan_to_num(F * D))
    amp[0, 0] = 0
    return kx, kz, k, amp


def chance_sea(N=1024, L=4000.0):
    """The P0 chance sea: the tallest crest of 300 random seeds (packaging/p0/peak_search.json). Hs scaled to 12 m."""
    best = json.load(open(os.path.join(EP, "packaging", "p0", "peak_search.json")))
    kx, kz, k, amp = spectrum_grid(N, L, spread=best["spread_cos_power"])
    rng = np.random.default_rng(best["seed"])
    z = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))) * amp
    z *= (HS / 4) / np.fft.ifft2(z).real.std()
    kk = np.where(k > 0, k, 1)
    eta = np.fft.ifft2(z).real
    iz, ix = np.unravel_index(np.argmax(eta), eta.shape)
    return dict(z=z, kx=kx, kz=kz, k=k, L=L, N=N, seed=best["seed"], seeds_tried=best["seeds_tried"],
                spread=best["spread_cos_power"], eta=eta, crest_ij=(int(iz), int(ix)),
                Dx=np.fft.ifft2(-1j * kx / kk * z).real, Dz=np.fft.ifft2(-1j * kz / kk * z).real)


def sea_at(sea, t):
    """The same sea t seconds later (t < 0: earlier): every wave moved on by its own speed (linear dispersion).
    The buoy point and the camera aim stay those of the t = 0 crest, so frames at different times line up."""
    w = np.sqrt(G * sea["k"])
    z = sea["z"] * np.exp(-1j * w * t)
    kk = np.where(sea["k"] > 0, sea["k"], 1)
    iz, ix = sea["crest_ij"]
    return dict(sea, z=z, eta=np.fft.ifft2(z).real, Dx=np.fft.ifft2(-1j * sea["kx"] / kk * z).real,
                Dz=np.fft.ifft2(-1j * sea["kz"] / kk * z).real, aim_h=float(sea.get("aim_h", sea["eta"][iz, ix])), t=t)


def point_record(sea, x0, z0, t, keep=1e-3):
    """Surface height at (x0, z0) over times t (s): every wave of the sea moved on by its own speed (linear)."""
    z, kx, kz, k, N = sea["z"], sea["kx"], sea["kz"], sea["k"], sea["N"]
    sel = np.abs(z) > np.abs(z).max() * keep
    a = z[sel] * np.exp(1j * (kx[sel] * x0 + kz[sel] * z0)) / N ** 2
    w = np.sqrt(G * k[sel])
    return np.array([np.real(np.sum(a * np.exp(-1j * w * tt))) for tt in t])


def zero_up_waves(t, eta):
    """Zero-upcrossing waves: (t_start, t_end, height, crest)."""
    up = np.where((eta[:-1] < 0) & (eta[1:] >= 0))[0]
    return [(t[i0], t[i1], eta[i0:i1].max() - eta[i0:i1].min(), eta[i0:i1].max()) for i0, i1 in zip(up[:-1], up[1:])]


def detail_slopes(N=256, L=90.0, seed=7, t=0.0):
    """Short wind chop for the surface texture only (look, not physics): tiled, slopes scaled to rms 0.12. Moved on
    in time like the big waves (each ripple at its own deep-water speed), so in motion the glints travel with the
    water instead of popping (P9 cold-open test, 2026-10-18); t = 0 gives the key frames' texture unchanged."""
    kx, kz, k, amp = spectrum_grid(N, L, tp=2.6, spread=4)
    rng = np.random.default_rng(seed)
    z = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))) * amp
    if t:
        z = z * np.exp(-1j * np.sqrt(G * k) * t)
    sx = np.fft.ifft2(1j * kx * z).real
    sz = np.fft.ifft2(1j * kz * z).real
    c = 0.12 / np.sqrt((sx ** 2 + sz ** 2).mean())
    return np.dstack([sx * c, sz * c, np.zeros_like(sx), np.zeros_like(sx)]).astype("f4"), L


def cloud_field(n=512, seed=3):
    """Overcast cloud texture for the sky (look only): filtered noise, zero mean, unit std, tiled."""
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(n)
    k = np.hypot(*np.meshgrid(f, f))
    k[0, 0] = 1
    c = np.fft.ifft2(np.fft.fft2(rng.standard_normal((n, n))) * k ** -1.7).real
    return (c - c.mean()) / c.std()


# Grades (look only): sky horizon / zenith, diffuse light behind the wave, deep water, light through the crest,
# exposure, saturation, contrast, light height (rad), detail, reflection cap, fog distance (m), foam on steep
# crests, wind streaks, cloud contrast, grain, vignette. "clean" = the P2 thumbnail; "storm" = P6, after the look
# board (04 overcast light + wind streaks, 05 light through the crest, 01 grain + vignette).
GRADES = {
    "clean": dict(hor=(0.98, 0.97, 0.94), zen=(0.30, 0.38, 0.48), sunc=(0.22, 0.20, 0.17),
                  deepc=(0.006, 0.035, 0.060), sssc=(0.04, 0.46, 0.50), exp=1.8, sat=1.05, con=1.12, sun_el=0.12,
                  det=1.1, fcap=0.55, fog=9000.0, cap=0.0, streak=0.0, cloud=0.0, grain=0.0, vig=0.0,
                  disc=1.0, sssp=1.5),
    "storm": dict(hor=(0.42, 0.45, 0.47), zen=(0.08, 0.10, 0.13), sunc=(0.12, 0.12, 0.10),
                  deepc=(0.004, 0.022, 0.036), sssc=(0.02, 0.24, 0.25), exp=1.75, sat=0.95, con=1.10, sun_el=0.15,
                  det=1.25, fcap=0.42, fog=5200.0, cap=0.85, streak=0.5, cloud=0.10, grain=0.016, vig=0.30,
                  disc=0.0, sssp=2.0),
    # "lit" (2026-10-17, your note "looks dull" on storm): light breaking through behind the wave, so the crest
    # glows teal against it (look board 05); deep navy water, white foam, a sparkle path; the upper left stays dark.
    "lit": dict(hor=(0.30, 0.39, 0.48), zen=(0.04, 0.07, 0.12), sunc=(1.0, 0.93, 0.80), broad=0.10, halo=1.6,
                hexp=2500.0, disc=0.0, sun_at_crest=True, deepc=(0.003, 0.028, 0.058), sssc=(0.05, 0.62, 0.60),
                sssp=2.0, exp=1.8, sat=1.15, con=1.18, sun_el=0.035, det=1.25, fcap=0.5, fog=6000.0, cap=1.0,
                streak=0.45, cloud=0.14, grain=0.012, vig=0.32, foamc=(0.93, 0.96, 0.97), glint=1.6),
    # "studio" (2026-10-17, test after "not look good" on lit): DESIGN §5's own look, not photoreal. The sea as a lit
    # matte shape on the night ground, the far sea fading into the dark, one teal light on the tallest wave,
    # a soft back light for a rim on the crest. No streaks, no clouds, little ripple.
    "studio": dict(hor=(0.085, 0.080, 0.110), zen=(0.045, 0.040, 0.065), sunc=(0.55, 0.60, 0.62), broad=0.0,
                   halo=0.35, hexp=1500.0, disc=0.0, sun_at_crest=True, deepc=(0.004, 0.006, 0.012),
                   matte=(0.075, 0.095, 0.130), sssc=(0.03, 0.42, 0.44), sssp=3.0, exp=1.9, sat=1.05, con=1.12,
                   sun_el=0.03, det=0.35, fcap=0.18, fog=1300.0, cap=0.0, streak=0.0, cloud=0.0, grain=0.010,
                   vig=0.35, foamc=(0.86, 0.89, 0.90), glint=0.4),
    # "studio2": the studio look made to read as water, not dunes: no fold blotches, less matte (sand) shading,
    # more reflection of the light, a sharp glitter path toward the camera, fine ripples kept up close.
    "studio2": dict(hor=(0.13, 0.13, 0.17), zen=(0.040, 0.036, 0.060), sunc=(0.70, 0.74, 0.76), broad=0.0,
                    halo=0.45, hexp=1500.0, disc=0.0, sun_at_crest=True, deepc=(0.003, 0.005, 0.010),
                    matte=(0.030, 0.040, 0.058), sssc=(0.03, 0.42, 0.44), sssp=3.0, exp=1.9, sat=1.05, con=1.15,
                    sun_el=0.03, det=0.7, fcap=0.45, fog=1300.0, cap=0.0, streak=0.0, cloud=0.0, grain=0.010,
                    vig=0.35, foamc=(0.86, 0.89, 0.90), glint=2.2, gexp=900.0, jfoam=0.0, sssflat=0.3),
}

SKY_GLSL = """
    uniform vec3 hor; uniform vec3 zen; uniform vec3 sunc; uniform float disc;
    uniform float broad; uniform float halo; uniform float hexp;
    vec3 sky(vec3 d, vec3 sun){
        float t = clamp(d.y, 0.0, 1.0);
        vec3 c = mix(hor, zen, pow(t, 0.45));
        float g = max(dot(d, sun), 0.0);
        c += sunc * (pow(g, 8.0) * broad + pow(g, hexp) * halo + pow(g, 600.0) * 8.0 * disc);
        return c;
    }"""

SEA_VS = """
    #version 330
    uniform mat4 mvp; uniform sampler2D A; uniform float L; uniform float texel; uniform float chop;
    in vec3 in_p; out vec3 wp; out vec2 uv; out float dist;
    void main(){
        uv = in_p.xy / L;
        float lod = max(0.0, log2(in_p.z * 0.0008 / texel));
        vec4 a = textureLod(A, uv, lod);
        wp = vec3(in_p.x + chop * a.g, a.r, in_p.y + chop * a.b); dist = in_p.z;
        gl_Position = mvp * vec4(wp, 1.0);
    }"""

SEA_FS = """
    #version 330
    uniform sampler2D A; uniform sampler2D B; uniform sampler2D C;
    uniform float Ld; uniform vec3 eye; uniform vec3 sun; uniform float hs; uniform vec3 deepc; uniform vec3 sssc;
    uniform float det; uniform float fcap; uniform float fogd; uniform float capf; uniform float streak;
    uniform float sssp; uniform vec3 foamc; uniform float glint; uniform vec3 matte; uniform vec3 key;
    uniform float jfoam; uniform float gexp; uniform float sssflat;
    in vec3 wp; in vec2 uv; in float dist; out vec4 o;""" + SKY_GLSL + """
    void main(){
        vec3 n0 = normalize(texture(B, uv).xyz);
        vec2 s = texture(C, wp.xz / Ld).xy * exp(-dist / 2500.0) * det
               + texture(C, wp.xz / (Ld * 0.19)).xy * 0.7 * exp(-dist / 150.0) * det;     // fine ripples up close
        vec3 n = normalize(n0 + vec3(-s.x, 0.0, -s.y));
        float J = texture(A, uv).a;
        vec3 v = normalize(eye - wp);
        float fr = min(0.02 + 0.98 * pow(1.0 - max(dot(n, v), 0.0), 5.0), fcap);
        vec3 refl = sky(reflect(-v, n), sun);
        float hh = clamp(wp.y / hs, -1.0, 1.6);
        float back = pow(max(dot(-v, sun), 0.0), 3.0);
        vec3 sss = sssc * pow(max(hh, 0.0), sssp) * (0.25 + 1.6 * back)
                 * (sssflat + (1.0 - sssflat) * smoothstep(0.995, 0.88, n0.y));   // light through steep faces
        vec3 col = mix(deepc + sss + matte * (0.15 + max(dot(n, key), 0.0)), refl, fr);   // matte: lit-shape look
        float foam = smoothstep(0.75, 0.15, J) * (0.6 + 0.4 * smoothstep(0.2, 0.9, hh)) * jfoam;
        foam += capf * smoothstep(0.85, 1.30, hh) * smoothstep(0.95, 0.80, n0.y);
        float sm = texture(C, vec2(wp.x / (Ld * 8.0), wp.z / (Ld * 0.3))).x / 0.12;
        foam += streak * smoothstep(0.7, 1.7, sm) * exp(-dist / 2500.0) * smoothstep(40.0, 160.0, dist);
        foam *= 0.75 + 0.5 * texture(C, wp.xz / (Ld * 0.37)).x / 0.12;
        col = mix(col, foamc, clamp(foam, 0.0, 1.0));
        col += vec3(1.0, 0.9, 0.75) * pow(max(dot(reflect(-sun, n), v), 0.0), gexp) * 3.0 * (1.0 - fcap) * glint;
        float fog = 1.0 - exp(-dist / fogd);
        col = mix(col, sky(normalize(vec3(-v.x, 0.02, -v.z)), sun), fog);
        o = vec4(col, 1.0);
    }"""

BUOY_VS = """
    #version 330
    uniform mat4 mvp; uniform vec3 c; uniform float r; in vec3 in_n; out vec3 nn;
    void main(){ nn = in_n; gl_Position = mvp * vec4(c + in_n * r, 1.0); }"""

BUOY_FS = """
    #version 330
    uniform vec3 sun; uniform vec3 eye; uniform vec3 c; in vec3 nn; out vec4 o;
    void main(){
        vec3 n = normalize(nn);
        vec3 base = vec3(1.0, 0.72, 0.05);
        float dif = max(dot(n, sun), 0.0) * 0.5 + 0.55 * (0.5 + 0.5 * n.y);
        vec3 v = normalize(eye - c);
        float sp = pow(max(dot(reflect(-sun, n), v), 0.0), 40.0);
        float stripe = step(abs(n.y), 0.06);
        o = vec4(mix(base * dif, vec3(0.08), stripe) + sp * 0.4, 1.0);
    }"""


def render(sea, cam, grade="storm", chop=1.4, buoy=None):
    """cam: dist (m, from the crest along +x, the way the waves travel: + = the wave comes at you), side (fraction
    of dist sideways), or pos=(along, across) in metres from the crest, eye_h (m above the local surface, or above mean sea level with eye_abs=True), fov (deg),
    look (aim height as a fraction of the crest height), frame_x / frame_y (where the crest lands, 0-1 across /
    down). buoy: dict(r=0.45 (m; a 0.9 m Waverider), at=None for the crest or (along, across) metres from the
    camera, light=True). Returns (PIL RGB image, info with screen positions)."""
    import moderngl
    gr = GRADES[grade]
    eta, Dx, Dz, L, N = sea["eta"], sea["Dx"], sea["Dz"], sea["L"], sea["N"]
    dx = L / N

    def cd(a, ax):
        return (np.roll(a, -1, ax) - np.roll(a, 1, ax)) / 2

    Pi = np.dstack([dx + chop * cd(Dx, 1), cd(eta, 1), chop * cd(Dz, 1)])
    Pj = np.dstack([chop * cd(Dx, 0), cd(eta, 0), dx + chop * cd(Dz, 0)])
    nrm = np.cross(Pj, Pi)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    J = ((1 + chop * cd(Dx, 1) / dx) * (1 + chop * cd(Dz, 0) / dx)
         - (chop * cd(Dx, 0) / dx) * (chop * cd(Dz, 1) / dx))
    iz, ix = sea["crest_ij"]
    crest = np.array([ix * dx + chop * Dx[iz, ix], eta[iz, ix], iz * dx + chop * Dz[iz, ix]], "f4")

    def surface_at(x, z):
        """Height of the shifted (choppy) surface above world point x, z: find the grid point that moves there."""
        x0, z0 = x, z
        for _ in range(8):
            i, j = int(round(x0 / dx)) % N, int(round(z0 / dx)) % N
            x0, z0 = x - chop * Dx[j, i], z - chop * Dz[j, i]
        return float(eta[int(round(z0 / dx)) % N, int(round(x0 / dx)) % N])

    ctx = moderngl.create_standalone_context()
    ms = ctx.framebuffer(color_attachments=[ctx.renderbuffer((W, H), 4, samples=4)],
                         depth_attachment=ctx.depth_renderbuffer((W, H), samples=4))
    res = ctx.framebuffer(color_attachments=[ctx.renderbuffer((W, H), 4)])

    def tex(arr, unit):
        t = ctx.texture((arr.shape[1], arr.shape[0]), 4, np.ascontiguousarray(arr, "f4").tobytes(), dtype="f4")
        t.repeat_x = t.repeat_y = True
        t.build_mipmaps()
        t.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        t.anisotropy = 16.0                   # the wave face is seen edge-on: without this the middle goes soft
        t.use(unit)
        return t

    det, Ld = detail_slopes(t=float(sea.get("t", 0.0)))
    tex(np.dstack([eta, Dx, Dz, J]), 0)
    tex(np.dstack([nrm, np.zeros_like(eta)]), 1)
    tex(det, 2)

    if "pos" in cam:                          # explicit (along x, across z) offset from the crest, metres
        eye = crest + np.array([cam["pos"][0], 0, cam["pos"][1]], "f4")
    else:
        eye = crest + np.array([cam["dist"], 0, cam.get("side", 0.0) * abs(cam["dist"])], "f4")
    eye[1] = cam["eye_h"] + (0.0 if cam.get("eye_abs") else surface_at(eye[0], eye[2]))
    target = crest.copy()
    target[1] = sea.get("aim_h", crest[1]) * cam.get("look", 0.6)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    f = 1 / np.tan(np.radians(cam["fov"]) / 2)
    # turn the camera so the crest lands at frame_x across (rule of thirds), then tilt for frame_y
    yaw = np.arctan((2 * cam.get("frame_x", 0.5) - 1) * (W / H) / f)
    r0 = np.cross(fwd, [0, 1, 0])
    r0 /= np.linalg.norm(r0)
    fwd = np.cos(yaw) * fwd - np.sin(yaw) * r0
    pitch = np.arctan((2 * cam.get("frame_y", 0.5) - 1) / f)
    up0 = np.cross(np.cross(fwd, [0, 1, 0]), fwd)
    up0 /= np.linalg.norm(up0)
    fwd = np.cos(pitch) * fwd + np.sin(pitch) * up0
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 1, 0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    near, far = 0.2, 9000.0
    P = np.zeros((4, 4), "f4")
    P[0, 0], P[1, 1] = f / (W / H), f
    P[2, 2], P[2, 3], P[3, 2] = (far + near) / (near - far), 2 * far * near / (near - far), -1
    V = np.eye(4, dtype="f4")
    V[0, :3], V[1, :3], V[2, :3] = right, up, -fwd
    V[:3, 3] = -V[:3, :3] @ eye
    PV = P @ V
    MVP = PV.T.astype("f4")

    def screen(p):
        c = PV @ np.array([*p, 1.0])
        return float((c[0] / c[3] + 1) / 2 * W), float((1 - c[1] / c[3]) / 2 * H)

    # projected grid: log-spaced rows along the horizontal view direction
    hf = np.array([fwd[0], 0, fwd[2]], "f4")
    hf /= np.linalg.norm(hf)
    hr = np.cross(hf, [0, 1, 0])
    nu, nv = 1800, 1600
    d = (far * 0.8) ** np.linspace(0, 1, nv)
    sx = np.linspace(-1, 1, nu)
    D_, S_ = np.meshgrid(d, sx, indexing="ij")
    lat = S_ * (D_ + 20) * np.tan(np.radians(cam["fov"]) / 2) * (W / H) * 1.5
    px = eye[0] + hf[0] * D_ + hr[0] * lat
    pz = eye[2] + hf[2] * D_ + hr[2] * lat
    verts = np.dstack([px, pz, D_]).reshape(-1, 3).astype("f4")
    a = np.arange(nv * nu).reshape(nv, nu)
    q0, q1, q2, q3 = a[:-1, :-1], a[:-1, 1:], a[1:, :-1], a[1:, 1:]
    idx = np.stack([q0, q2, q1, q1, q2, q3], -1).reshape(-1).astype("i4")

    sun_el = gr["sun_el"]
    if gr.get("sun_at_crest"):
        to_c = (crest - eye) * np.array([1, 0, 1], "f4")
        to_c /= np.linalg.norm(to_c)
        sun = (to_c * np.cos(sun_el) + np.array([0, np.sin(sun_el), 0], "f4")).astype("f4")
    else:
        sun = (hf * np.cos(sun_el) + np.array([0, np.sin(sun_el), 0], "f4") + hr * 0.25).astype("f4")
    sun /= np.linalg.norm(sun)
    prog = ctx.program(vertex_shader=SEA_VS, fragment_shader=SEA_FS)
    prog["mvp"].write(MVP.tobytes())
    for k_, v_ in dict(L=L, texel=dx, chop=chop, Ld=Ld, hs=HS, det=gr["det"], fcap=gr["fcap"], fogd=gr["fog"],
                       capf=gr["cap"], streak=gr["streak"], disc=gr["disc"], sssp=gr["sssp"],
                       broad=gr.get("broad", 0.55), halo=gr.get("halo", 0.0), hexp=gr.get("hexp", 400.0),
                       glint=gr.get("glint", 1.0), jfoam=gr.get("jfoam", 1.0), gexp=gr.get("gexp", 400.0),
                       sssflat=gr.get("sssflat", 0.3)).items():
        prog[k_].value = float(v_)
    prog["eye"].value = tuple(map(float, eye))
    prog["sun"].value = tuple(map(float, sun))
    prog["A"].value, prog["B"].value, prog["C"].value = 0, 1, 2
    for k_ in ("hor", "zen", "sunc", "deepc", "sssc"):
        prog[k_].value = tuple(map(float, gr[k_]))
    prog["foamc"].value = tuple(map(float, gr.get("foamc", (0.86, 0.89, 0.90))))
    prog["matte"].value = tuple(map(float, gr.get("matte", (0.0, 0.0, 0.0))))
    key = -hf * 0.4 + np.array([0, 0.75, 0], "f4") - hr * 0.6                 # key light from the upper left, front
    prog["key"].value = tuple(map(float, key / np.linalg.norm(key)))
    vao = ctx.vertex_array(prog, [(ctx.buffer(verts), "3f", "in_p")], ctx.buffer(idx))
    ms.use()
    ctx.clear(0, 0, 0, 0)
    ctx.enable(moderngl.DEPTH_TEST)
    vao.render()

    info = dict(crest_m=float(crest[1]), crest_xy=screen(crest), eye=[float(v) for v in eye], fov=cam["fov"],
                horizon_y=screen(eye + hf * 8000.0 + np.array([0, -eye[1], 0]))[1])
    if buoy is not None:                      # a wave buoy (the experiment's own object), true size by default
        rad = buoy.get("r", 0.45)
        if buoy.get("at") is None:
            # moored at the crest grid point: ride that water particle exactly (its own displaced vertex), so the
            # buoy sits on the rendered surface every frame; the nearest-node lookup snapped between 3.9 m nodes
            # and left it floating above or sunk into steep faces (P9, your note "buoy placement not correct")
            bp = crest.copy()
            bp[1] = float(eta[iz, ix]) + rad * 0.35
        else:
            bp = eye + hf * buoy["at"][0] + hr * buoy["at"][1]
            bp[1] = surface_at(bp[0], bp[2]) + rad * 0.35
        u, w_ = np.meshgrid(np.linspace(0, np.pi, 40), np.linspace(0, 2 * np.pi, 80), indexing="ij")
        sp = np.dstack([np.sin(u) * np.cos(w_), np.cos(u), np.sin(u) * np.sin(w_)]).reshape(-1, 3).astype("f4")
        aa = np.arange(40 * 80).reshape(40, 80)
        bidx = np.stack([aa[:-1, :-1], aa[1:, :-1], aa[:-1, 1:], aa[:-1, 1:], aa[1:, :-1], aa[1:, 1:]], -1)
        bprog = ctx.program(vertex_shader=BUOY_VS, fragment_shader=BUOY_FS)
        bprog["mvp"].write(MVP.tobytes())
        bprog["c"].value = tuple(map(float, bp))
        bprog["r"].value = float(rad)
        bprog["sun"].value = tuple(map(float, sun))
        bprog["eye"].value = tuple(map(float, eye))
        ctx.vertex_array(bprog, [(ctx.buffer(sp), "3f", "in_n")],
                         ctx.buffer(bidx.reshape(-1).astype("i4"))).render()
        top = bp + np.array([0, rad + 0.5, 0], "f4")
        info.update(buoy_xy=screen(bp), buoy_top_xy=screen(top), buoy_px=abs(screen(top)[1] - screen(bp)[1]) * rad / (rad + 0.5),
                    buoy_dist_m=float(np.linalg.norm(bp - eye)))

    ctx.copy_framebuffer(res, ms)
    img = Image.frombytes("RGBA", (W, H), res.read(components=4)).transpose(Image.FLIP_TOP_BOTTOM)
    ctx.release()

    # sky behind (same gradient as the shader, per pixel ray) + overcast clouds, same tone curve as the sea
    yy, xx = np.mgrid[0:H, 0:W].astype("f4")
    ndc_x, ndc_y = (xx + 0.5) / W * 2 - 1, 1 - (yy + 0.5) / H * 2
    dirs = (fwd[None, None] + right[None, None] * (ndc_x[..., None] * (W / H) / f)
            + up[None, None] * (ndc_y[..., None] / f))
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    t = np.clip(dirs[..., 1], 0, 1) ** 0.45
    c = (1 - t[..., None]) * np.array(gr["hor"]) + t[..., None] * np.array(gr["zen"])
    g = np.maximum(dirs @ sun, 0)
    c += np.array(gr["sunc"]) * (g ** 8.0 * gr.get("broad", 0.55) + g ** gr.get("hexp", 400.0) * gr.get("halo", 0.0)
                                 + g ** 600.0 * 8.0 * gr["disc"])[..., None]
    if gr["cloud"]:
        cl = cloud_field()
        dy = np.maximum(dirs[..., 1], 0.02)
        u_ = (dirs[..., 0] / dy * 0.06) % 1.0 * cl.shape[1]
        v_ = (dirs[..., 2] / dy * 0.06) % 1.0 * cl.shape[0]
        cv = cl[v_.astype(int) % cl.shape[0], u_.astype(int) % cl.shape[1]]
        fade = np.clip(dirs[..., 1] / 0.12, 0, 1)
        c *= (1 + gr["cloud"] * cv * fade)[..., None]

    def tone(x):
        y = 1 - np.exp(-x * gr["exp"])
        m = y.mean(-1, keepdims=True)
        y = m + (y - m) * gr["sat"]
        return (y - 0.5) * gr["con"] + 0.5

    sky_rgb = tone(c)
    if gr["cloud"]:                           # soften the cloud texture (it was sampled nearest)
        from PIL import ImageFilter
        sky_img = Image.fromarray((np.clip(sky_rgb, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(3))
        sky_rgb = np.asarray(sky_img).astype("f4") / 255
    sea_rgba = np.asarray(img).astype("f4") / 255
    al = sea_rgba[..., 3:4]
    rgb = sky_rgb * (1 - al) + tone(sea_rgba[..., :3]) * al
    if buoy is not None and buoy.get("light", True):
        bx, by = info["buoy_top_xy"]
        rr = np.hypot(xx - bx, yy - by)
        glow = np.exp(-(rr / 5.0) ** 2) * 0.9 + np.exp(-(rr / 18.0) ** 2) * 0.18
        rgb = rgb + glow[..., None] * np.array([1.0, 0.86, 0.55])
    if gr["vig"]:
        r2 = ((xx - W / 2) / (W / 2)) ** 2 * 0.6 + ((yy - H / 2) / (H / 2)) ** 2 * 0.4
        rgb = rgb * (1 - gr["vig"] * r2)[..., None]
    if gr["grain"]:
        rgb = rgb + np.random.default_rng(11).normal(0, gr["grain"], (H, W, 1))
    out = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), "RGB")
    if buoy is not None and buoy.get("light", True):     # the antenna mast under the light
        from PIL import ImageDraw
        ppm = info["buoy_px"] / rad
        ImageDraw.Draw(out).line([screen(bp + np.array([0, rad * 0.85, 0], "f4")),
                                  screen(top - np.array([0, 0.06, 0], "f4"))],
                                 fill=(38, 38, 40), width=max(1, int(round(0.035 * ppm))))
    return out, info
