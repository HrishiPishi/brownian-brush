from __future__ import annotations
import threading
import webbrowser
import html
import json
import mimetypes
import shutil
import uuid
import warnings
from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import default as email_default_policy
from io import BytesIO
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
from PIL import Image

from src.equations import dot_coordinates_text, stroke_equations_text
from src.image_processing import compute_darkness, compute_edges, load_image, resize_image, to_grayscale
from src.rendering import render_dots, render_strokes, side_by_side
from src.sampling import generate_brownian_strokes, make_probability_map, sample_edge_weighted_points, sample_points
from src.utils import RenderMetadata

warnings.filterwarnings("ignore", category=DeprecationWarning)


@dataclass
class UploadedFile:
    filename: str
    file: BytesIO


class SimpleForm:
    def __init__(self, fields: dict[str, str], files: dict[str, UploadedFile]):
        self.fields = fields
        self.files = files

    def getfirst(self, name: str, default: object | None = None) -> object:
        return self.fields.get(name, default)

    def __contains__(self, name: str) -> bool:
        return name in self.fields or name in self.files

    def __getitem__(self, name: str) -> UploadedFile:
        return self.files[name]


def parse_multipart(headers, rfile) -> SimpleForm:
    try:
        length = int(headers.get("Content-Length", "0"))
    except ValueError:
        length = 0
    body = rfile.read(length)
    content_type = headers.get("Content-Type", "")
    raw = (f"Content-Type: {content_type}\nMIME-Version: 1.0\n\n").encode("utf-8") + body
    message = BytesParser(policy=email_default_policy).parsebytes(raw)

    fields: dict[str, str] = {}
    files: dict[str, UploadedFile] = {}
    if not message.is_multipart():
        return SimpleForm(fields, files)

    for part in message.iter_parts():
        disposition = part.get("Content-Disposition", "")
        if "form-data" not in disposition:
            continue
        params = dict(part.get_params(header="content-disposition", unquote=True) or [])
        name = params.get("name")
        if not name:
            continue
        payload = part.get_payload(decode=True) or b""
        filename = params.get("filename")
        if filename is not None:
            files[name] = UploadedFile(filename=filename, file=BytesIO(payload))
        else:
            charset = part.get_content_charset() or "utf-8"
            fields[name] = payload.decode(charset, errors="replace")
    return SimpleForm(fields, files)


MODE_DOTS = "dots"
MODE_STROKES = "brownian lines"
MODE_EDGES = "edge dots"
VALID_MODES = {MODE_DOTS, MODE_STROKES, MODE_EDGES}

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

CSS = """
<style>
* { box-sizing: border-box; }
html, body {
  margin: 0;
  padding: 0;
  background: white;
  color: #111;
  font-family: Arial, Helvetica, sans-serif;
  font-size: 16px;
}
body { padding: 18px 28px 40px; }
h1 {
  margin: 0 0 10px;
  font-size: 42px;
  line-height: 1;
  letter-spacing: -1px;
}
p {
  margin: 0 0 18px;
  color: #555;
  font-size: 20px;
}
h2 {
  margin: 22px 0 8px;
  font-size: 28px;
  line-height: 1.1;
}
h3 {
  margin: 14px 0 8px;
  font-size: 20px;
}
hr {
  border: 0;
  border-top: 1px solid #bbb;
  margin: 18px 0;
}
form { margin: 0; padding: 0; }
.row {
  display: flex;
  flex-wrap: wrap;
  gap: 10px 12px;
  align-items: end;
  margin-bottom: 10px;
}
label {
  display: block;
  font-size: 14px;
  color: #555;
  margin-bottom: 3px;
}
input, select, button, textarea, a.button {
  font: inherit;
  color: #111;
  background: white;
  border: 1px solid #111;
  border-radius: 0;
  padding: 8px 10px;
  box-shadow: none;
  outline: none;
}
input[type="file"] { width: 300px; }
input[type="number"] { width: 110px; }
input[type="text"] { width: 92px; }
input[type="checkbox"] {
  width: auto;
  padding: 0;
  margin: 0 5px 0 0;
  vertical-align: middle;
}
select { min-width: 155px; }
button, a.button {
  cursor: pointer;
  text-decoration: none;
  display: inline-block;
}
button:hover, a.button:hover { background: #eee; }
.box {
  border: 1px solid #ccc;
  background: white;
  padding: 14px;
  margin-bottom: 14px;
}
.grid2 {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  align-items: start;
}
img {
  max-width: 100%;
  height: auto;
  border: 1px solid #ccc;
  background: white;
  display: block;
}
.small { font-size: 14px; color: #555; }
.error {
  border: 1px solid #111;
  background: white;
  padding: 10px;
  margin: 12px 0;
  color: #111;
}
textarea {
  width: 100%;
  min-height: 260px;
  font-family: Menlo, Consolas, monospace;
  font-size: 12px;
  line-height: 1.35;
  resize: vertical;
  background: white;
  color: #111;
  border: 1px solid #ccc;
}
.downloads {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin: 8px 0 0;
}
.math {
  max-width: 900px;
  line-height: 1.45;
}
.muted { color: #666; }
@media (max-width: 900px) {
  body { padding: 14px; }
  .grid2 { grid-template-columns: 1fr; }
  h1 { font-size: 34px; }
  p { font-size: 18px; }
}
</style>
"""


def defaults() -> dict[str, object]:
    return {
        "mode": MODE_STROKES,
        "max_size": 512,
        "seed": 17,
        "num_samples": 1600,
        "gamma": 1.35,
        "edge_weight": 2.5,
        "batch_size": 200,
        "frame_delay": 80,
        "invert": False,
        "background_color": "#ffffff",
        "drawing_color": "#000000",
        "dot_size": 1.5,
        "dot_opacity": 0.8,
        "stroke_length": 20,
        "stroke_jitter": 2.0,
        "stroke_width": 1,
        "stroke_opacity": 0.45,
    }


def clamp_int(value: object, low: int, high: int, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, parsed))


def clamp_float(value: object, low: float, high: float, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, parsed))


def first(form: SimpleForm, name: str, default: object) -> object:
    value = form.getfirst(name)
    return default if value is None else value


def parse_values(form: SimpleForm) -> dict[str, object]:
    d = defaults()
    mode = str(first(form, "mode", d["mode"]))
    d["mode"] = mode if mode in VALID_MODES else MODE_STROKES
    d["max_size"] = clamp_int(first(form, "max_size", d["max_size"]), 64, 1200, int(d["max_size"]))
    d["seed"] = clamp_int(first(form, "seed", d["seed"]), 0, 2_147_483_647, int(d["seed"]))
    d["num_samples"] = clamp_int(first(form, "num_samples", d["num_samples"]), 1, 50_000, int(d["num_samples"]))
    d["gamma"] = clamp_float(first(form, "gamma", d["gamma"]), 0.2, 5.0, float(d["gamma"]))
    d["edge_weight"] = clamp_float(first(form, "edge_weight", d["edge_weight"]), 0.0, 8.0, float(d["edge_weight"]))
    d["batch_size"] = clamp_int(first(form, "batch_size", d["batch_size"]), 1, 10_000, int(d["batch_size"]))
    d["frame_delay"] = clamp_int(first(form, "frame_delay", d["frame_delay"]), 10, 2000, int(d["frame_delay"]))
    d["invert"] = form.getfirst("invert") == "1"
    d["background_color"] = str(first(form, "background_color", d["background_color"])).strip() or "#ffffff"
    d["drawing_color"] = str(first(form, "drawing_color", d["drawing_color"])).strip() or "#000000"
    d["dot_size"] = clamp_float(first(form, "dot_size", d["dot_size"]), 0.1, 20.0, float(d["dot_size"]))
    d["dot_opacity"] = clamp_float(first(form, "dot_opacity", d["dot_opacity"]), 0.05, 1.0, float(d["dot_opacity"]))
    d["stroke_length"] = clamp_int(first(form, "stroke_length", d["stroke_length"]), 2, 80, int(d["stroke_length"]))
    d["stroke_jitter"] = clamp_float(first(form, "stroke_jitter", d["stroke_jitter"]), 0.0, 12.0, float(d["stroke_jitter"]))
    d["stroke_width"] = clamp_int(first(form, "stroke_width", d["stroke_width"]), 1, 8, int(d["stroke_width"]))
    d["stroke_opacity"] = clamp_float(first(form, "stroke_opacity", d["stroke_opacity"]), 0.05, 1.0, float(d["stroke_opacity"]))
    return d


def save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def grayscale_image(gray: np.ndarray) -> Image.Image:
    arr = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")


def render_progress_frame(mode: str, sample_data: np.ndarray | list[np.ndarray], image_shape: tuple[int, int], fraction: float, values: dict[str, object]) -> Image.Image:
    fraction = float(np.clip(fraction, 0.0, 1.0))
    if mode in {MODE_DOTS, MODE_EDGES}:
        points = np.asarray(sample_data)
        end = int(round(len(points) * fraction))
        return render_dots(
            points[:end],
            image_shape,
            dot_size=float(values["dot_size"]),
            opacity=float(values["dot_opacity"]),
            dot_color=str(values["drawing_color"]),
            background_color=str(values["background_color"]),
        )

    strokes = list(sample_data)
    end = int(round(len(strokes) * fraction))
    return render_strokes(
        strokes[:end],
        image_shape,
        stroke_width=int(values["stroke_width"]),
        opacity=float(values["stroke_opacity"]),
        stroke_color=str(values["drawing_color"]),
        background_color=str(values["background_color"]),
    )


def build_sample_data(values: dict[str, object], image_shape: tuple[int, int], darkness: np.ndarray, edges: np.ndarray, probability_map: np.ndarray):
    mode = str(values["mode"])
    if mode == MODE_DOTS:
        return sample_points(probability_map, int(values["num_samples"]), seed=int(values["seed"]))

    if mode == MODE_EDGES:
        return sample_edge_weighted_points(
            darkness,
            edges,
            int(values["num_samples"]),
            edge_weight=float(values["edge_weight"]),
            gamma=float(values["gamma"]),
            seed=int(values["seed"]),
        )

    targets = sample_points(probability_map, int(values["num_samples"]), seed=int(values["seed"]))
    return generate_brownian_strokes(
        targets,
        image_shape=image_shape,
        stroke_length=int(values["stroke_length"]),
        jitter=float(values["stroke_jitter"]),
        seed=int(values["seed"]) + 991,
    )


def export_text(mode: str, sample_data: np.ndarray | list[np.ndarray], image_shape: tuple[int, int], *, preview: bool = False) -> str:
    if mode == MODE_STROKES:
        return stroke_equations_text(list(sample_data), image_shape, max_segments=35 if preview else None)
    return dot_coordinates_text(np.asarray(sample_data), image_shape, max_points=80 if preview else None)


def build_result(values: dict[str, object], form: SimpleForm) -> dict[str, object]:
    if "image" not in form:
        raise ValueError("upload an image first")
    uploaded = form["image"]
    if isinstance(uploaded, list):
        uploaded = uploaded[0]
    if not getattr(uploaded, "filename", ""):
        raise ValueError("upload an image first")

    uploaded.file.seek(0)
    original = resize_image(load_image(uploaded.file), max_size=int(values["max_size"]))
    gray = to_grayscale(original)
    darkness = compute_darkness(gray, invert=bool(values["invert"]))
    edges = compute_edges(gray)
    probability_map = make_probability_map(darkness, gamma=float(values["gamma"]))
    image_shape = gray.shape

    sample_data = build_sample_data(values, image_shape, darkness, edges, probability_map)
    final = render_progress_frame(str(values["mode"]), sample_data, image_shape, 1.0, values)
    gray_preview = grayscale_image(gray)
    comparison = side_by_side(original, final, gutter=12, background_color="white")

    job_id = uuid.uuid4().hex
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    save_png(original, job_dir / "original.png")
    save_png(gray_preview, job_dir / "grayscale.png")
    save_png(final, job_dir / "brownian_brush_output.png")
    save_png(comparison, job_dir / "side_by_side.png")

    full_text = export_text(str(values["mode"]), sample_data, image_shape, preview=False)
    preview_text = export_text(str(values["mode"]), sample_data, image_shape, preview=True)
    (job_dir / "brownian_brush_equations.txt").write_text(full_text, encoding="utf-8")

    metadata = RenderMetadata(
        mode=str(values["mode"]),
        samples=int(values["num_samples"]),
        gamma=float(values["gamma"]),
        seed=int(values["seed"]),
        width=original.width,
        height=original.height,
    )
    metadata_dict = json.loads(metadata.to_json())
    metadata_dict.update({
        "max_size": int(values["max_size"]),
        "invert": bool(values["invert"]),
        "edge_weight": float(values["edge_weight"]),
        "background_color": str(values["background_color"]),
        "drawing_color": str(values["drawing_color"]),
        "dot_size": float(values["dot_size"]),
        "dot_opacity": float(values["dot_opacity"]),
        "stroke_length": int(values["stroke_length"]),
        "stroke_jitter": float(values["stroke_jitter"]),
        "stroke_width": int(values["stroke_width"]),
        "stroke_opacity": float(values["stroke_opacity"]),
    })
    (job_dir / "brownian_brush_metadata.json").write_text(json.dumps(metadata_dict, indent=2), encoding="utf-8")

    total_units = len(sample_data) if str(values["mode"]) == MODE_STROKES else len(np.asarray(sample_data))
    frame_count = max(2, min(40, int(np.ceil(total_units / max(1, int(values["batch_size"])))) + 1))
    fractions = np.linspace(0.0, 1.0, frame_count)

    frame_urls: list[str] = []
    for idx, frac in enumerate(fractions):
        frame = render_progress_frame(str(values["mode"]), sample_data, image_shape, float(frac), values)
        frame_name = f"frame_{idx:03d}.png"
        save_png(frame, job_dir / frame_name)
        frame_urls.append(f"/outputs/{job_id}/{frame_name}")

    return {
        "job_id": job_id,
        "original_url": f"/outputs/{job_id}/original.png",
        "gray_url": f"/outputs/{job_id}/grayscale.png",
        "output_download": f"/download/{job_id}/image",
        "equations_download": f"/download/{job_id}/equations",
        "metadata_download": f"/download/{job_id}/metadata",
        "frames": frame_urls,
        "preview_text": preview_text,
    }


def selected(value: object, target: object) -> str:
    return " selected" if str(value) == str(target) else ""


def checked(value: object) -> str:
    return " checked" if bool(value) else ""


def e(value: object) -> str:
    return html.escape(str(value), quote=True)


def render_form(values: dict[str, object]) -> str:
    modes = [MODE_DOTS, MODE_STROKES, MODE_EDGES]
    mode_options = "\n".join(f'<option value="{e(m)}"{selected(values["mode"], m)}>{e(m)}</option>' for m in modes)
    size_options = "\n".join(f'<option value="{size}"{selected(values["max_size"], size)}>{size}</option>' for size in [256, 384, 512, 640, 768, 900])
    return f"""
<form method="post" enctype="multipart/form-data">
  <div class="row">
    <div>
      <label>image</label>
      <input type="file" name="image" accept="image/png,image/jpeg" required>
    </div>
    <div>
      <label>mode</label>
      <select name="mode">{mode_options}</select>
    </div>
    <div>
      <label>max side</label>
      <select name="max_size">{size_options}</select>
    </div>
    <div>
      <label>seed</label>
      <input type="number" name="seed" min="0" max="2147483647" value="{e(values['seed'])}">
    </div>
    <div><button type="submit">run</button></div>
  </div>

  <div class="row">
    <div><label>dots / lines</label><input type="number" name="num_samples" min="1" max="50000" value="{e(values['num_samples'])}"></div>
    <div><label>contrast</label><input type="number" name="gamma" min="0.2" max="5" step="0.05" value="{e(values['gamma'])}"></div>
    <div><label>edge weight</label><input type="number" name="edge_weight" min="0" max="8" step="0.1" value="{e(values['edge_weight'])}"></div>
    <div><label>batch</label><input type="number" name="batch_size" min="1" max="10000" value="{e(values['batch_size'])}"></div>
    <div><label>frame delay ms</label><input type="number" name="frame_delay" min="10" max="2000" value="{e(values['frame_delay'])}"></div>
    <div style="padding-bottom: 9px;"><input type="checkbox" name="invert" value="1"{checked(values['invert'])}> invert</div>
  </div>

  <div class="row">
    <div><label>background</label><input type="text" name="background_color" value="{e(values['background_color'])}"></div>
    <div><label>ink</label><input type="text" name="drawing_color" value="{e(values['drawing_color'])}"></div>
    <div><label>dot size</label><input type="number" name="dot_size" min="0.1" max="20" step="0.1" value="{e(values['dot_size'])}"></div>
    <div><label>dot opacity</label><input type="number" name="dot_opacity" min="0.05" max="1" step="0.05" value="{e(values['dot_opacity'])}"></div>
    <div><label>line length</label><input type="number" name="stroke_length" min="2" max="80" value="{e(values['stroke_length'])}"></div>
    <div><label>jitter</label><input type="number" name="stroke_jitter" min="0" max="12" step="0.1" value="{e(values['stroke_jitter'])}"></div>
    <div><label>line width</label><input type="number" name="stroke_width" min="1" max="8" value="{e(values['stroke_width'])}"></div>
    <div><label>line opacity</label><input type="number" name="stroke_opacity" min="0.05" max="1" step="0.05" value="{e(values['stroke_opacity'])}"></div>
  </div>
</form>
"""


def render_result(result: dict[str, object], values: dict[str, object]) -> str:
    frames = result["frames"]
    assert isinstance(frames, list)
    frames_json = json.dumps(frames)
    preview = e(result["preview_text"])
    return f"""
<div class="grid2">
  <div class="box">
    <h2>original</h2>
    <img src="{e(result['original_url'])}" alt="original image">
    <h3>grayscale</h3>
    <img src="{e(result['gray_url'])}" alt="grayscale image">
  </div>
  <div class="box">
    <h2>reconstruction</h2>
    <img id="recon" src="{e(frames[0])}" alt="reconstruction">
    <div class="downloads">
      <button type="button" onclick="playFrames()">play</button>
      <button type="button" onclick="pauseFrames()">pause</button>
      <button type="button" onclick="resetFrames()">reset</button>
      <button type="button" onclick="finalFrame()">final</button>
    </div>
    <div class="downloads">
      <a class="button" href="{e(result['output_download'])}">download image</a>
      <a class="button" href="{e(result['equations_download'])}">download equations</a>
      <a class="button" href="{e(result['metadata_download'])}">download metadata</a>
    </div>
    <p class="small" id="frameText">frame 1 / {len(frames)}</p>
  </div>
</div>

<div class="box">
  <h2>equation / coordinate preview</h2>
  <textarea readonly>{preview}</textarea>
</div>

<div class="box math">
  <h2>what it is doing</h2>
  <div class="muted">
    it turns the image into grayscale, treats dark pixels as more important, samples from that distribution,
    and then draws the samples as dots or short random-walk line segments.
    for brownian lines, the equation file lists each tiny segment it drew.
  </div>
</div>

<script>
const frames = {frames_json};
const delay = {int(values['frame_delay'])};
let i = 0;
let timer = null;
const img = document.getElementById("recon");
const text = document.getElementById("frameText");
function showFrame(k) {{
  i = Math.max(0, Math.min(k, frames.length - 1));
  img.src = frames[i];
  text.textContent = "frame " + (i + 1) + " / " + frames.length;
}}
function playFrames() {{
  pauseFrames();
  timer = setInterval(function () {{
    if (i >= frames.length - 1) {{ pauseFrames(); return; }}
    showFrame(i + 1);
  }}, delay);
}}
function pauseFrames() {{
  if (timer !== null) {{ clearInterval(timer); timer = null; }}
}}
function resetFrames() {{ pauseFrames(); showFrame(0); }}
function finalFrame() {{ pauseFrames(); showFrame(frames.length - 1); }}
</script>
"""


def render_page(values: dict[str, object] | None = None, result: dict[str, object] | None = None, error: str | None = None) -> bytes:
    values = defaults() if values is None else values
    error_html = f'<div class="error">{e(error)}</div>' if error else ""
    result_html = render_result(result, values) if result else '<div class="box"><h2>nothing yet</h2><p class="small">upload an image and hit run</p></div>'
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>brownian brush</title>
  {CSS}
</head>
<body>
  <h1>brownian brush</h1>
  <p>turns an image into dots or messy little lines using probability</p>
  {error_html}
  {render_form(values)}
  <hr>
  {result_html}
</body>
</html>
"""
    return page.encode("utf-8")


def output_path(job_id: str, filename: str) -> Path | None:
    if not job_id.isalnum():
        return None
    filename = unquote(filename)
    root = OUTPUT_DIR.resolve()
    path = (OUTPUT_DIR / job_id / filename).resolve()
    if root not in path.parents or not path.is_file():
        return None
    return path


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        print(format % args)

    def send_bytes(self, body: bytes, status: int = 200, content_type: str = "text/html; charset=utf-8") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_local_file(self, path: Path, *, as_attachment: bool = False, download_name: str | None = None) -> None:
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(path.stat().st_size))
        if as_attachment:
            name = download_name or path.name
            self.send_header("Content-Disposition", f'attachment; filename="{name}"')
        self.end_headers()
        with path.open("rb") as fh:
            shutil.copyfileobj(fh, self.wfile)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            self.send_bytes(render_page())
            return

        if path.startswith("/outputs/"):
            parts = path.strip("/").split("/", 2)
            if len(parts) != 3:
                self.send_error(404)
                return
            local = output_path(parts[1], parts[2])
            if local is None:
                self.send_error(404)
                return
            self.send_local_file(local)
            return

        if path.startswith("/download/"):
            parts = path.strip("/").split("/")
            if len(parts) != 3:
                self.send_error(404)
                return
            _, job_id, kind = parts
            names = {
                "image": "brownian_brush_output.png",
                "equations": "brownian_brush_equations.txt",
                "metadata": "brownian_brush_metadata.json",
                "side_by_side": "side_by_side.png",
            }
            if kind not in names:
                self.send_error(404)
                return
            local = output_path(job_id, names[kind])
            if local is None:
                self.send_error(404)
                return
            self.send_local_file(local, as_attachment=True, download_name=names[kind])
            return

        self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/":
            self.send_error(404)
            return

        form = parse_multipart(self.headers, self.rfile)
        values = parse_values(form)
        try:
            result = build_result(values, form)
            self.send_bytes(render_page(values=values, result=result))
        except Exception as exc:
            self.send_bytes(render_page(values=values, error=str(exc)), status=400)


def main() -> None:
    host = "127.0.0.1"
    port = 8000

    try:
        server = ThreadingHTTPServer((host, port), Handler)
    except OSError:
        server = ThreadingHTTPServer((host, 0), Handler)
        port = server.server_address[1]

    url = f"http://{host}:{port}"

    print(f"brownian brush running at {url}")
    print("press ctrl c to stop")

    threading.Timer(0.75, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


        def main() -> None:
    host = "127.0.0.1"
    port = 8000

    try:
        server = ThreadingHTTPServer((host, port), Handler)
    except OSError:
        server = ThreadingHTTPServer((host, 0), Handler)
        port = server.server_address[1]

    url = f"http://{host}:{port}"

    print(f"brownian brush running at {url}")
    print("press ctrl c to stop")

    threading.Timer(0.75, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
