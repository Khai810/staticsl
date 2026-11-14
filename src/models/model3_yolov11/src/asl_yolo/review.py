from __future__ import annotations
from datetime import datetime, timezone
import io
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .core import digest, read_json, read_rgb, write_json
from .vision import clamp_box


def prepare_annotation_review(manifest, landmarks, env):
    path = Path(env["review_dir"]) / "review.json"
    if path.exists():
        saved = read_json(path)
        if saved["manifest"] != manifest["fingerprint"] or saved["landmarks_signature"] != landmarks["signature"]:
            raise ValueError("Existing review is incompatible; use new work_name")
        return path
    write_json(path, {"manifest": manifest["fingerprint"], "landmarks_signature": landmarks["signature"],
        "items": {r["sample_id"]: {"state": "pending", "boxes": landmarks["items"][r["sample_id"]]["boxes"],
                    "flags": landmarks["items"][r["sample_id"]]["flags"], "reviewer": "", "note": ""} for r in manifest["rows"]}})
    return path


def save_decision(path, row, state, boxes, reviewer, note=""):
    if state not in ("approved", "negative", "pending"):
        raise ValueError("Use approved, negative, or pending")
    if state != "pending" and not reviewer.strip():
        raise ValueError("Reviewer name required")
    boxes = [clamp_box(b, row["width"], row["height"]) for b in boxes]
    data = read_json(path)
    previous = data["items"][row["sample_id"]]
    if state == "approved" and not boxes:
        raise ValueError("Approved positive requires at least one box")
    if state == "negative" and (boxes or not note.strip()):
        raise ValueError("Confirmed no-hand image requires zero boxes and a reason")
    if state == "approved" and previous["flags"] and not note.strip():
        raise ValueError("Explain resolution of flags before approval")
    data["items"][row["sample_id"]] = {"state": state, "boxes": boxes, "flags": previous["flags"],
        "reviewer": reviewer.strip(), "note": note, "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "image_sha256": row["sha256"], "history": previous.get("history", []) + [{k: v for k, v in previous.items() if k != "history"}]}
    write_json(path, data)


def require_reviewed(manifest, review_path):
    data = read_json(review_path)
    if data["manifest"] != manifest["fingerprint"] or set(data["items"]) != {r["sample_id"] for r in manifest["rows"]}:
        raise ValueError("Review/manifest mismatch")
    pending = []
    for row in manifest["rows"]:
        item = data["items"][row["sample_id"]]
        if item["state"] not in ("approved", "negative") or not item.get("reviewer") or item.get("image_sha256") != row["sha256"]:
            pending.append(row["sample_id"])
            continue
        if item["state"] == "negative":
            if item["boxes"] or not item.get("note"):
                raise ValueError("Negative requires confirmed no-hand reason")
        elif not item["boxes"] or (item["flags"] and not item.get("note")):
            raise ValueError("Unresolved positive annotation")
        for box in item["boxes"]:
            if not np.allclose(box, clamp_box(box, row["width"], row["height"])):
                raise ValueError("Box extends beyond image")
    if pending:
        raise RuntimeError(f"Review gate: {len(pending)} images unresolved across train/val/test. Open review UI; no training/export yet.")
    return data


def review_widget(manifest, dataset, review_path):
    """Colab/Windows notebook UI: image selection + editable box sliders, no IDs to type."""
    import ipywidgets as w
    from IPython.display import display
    rows = manifest["rows"]
    reviewer = w.Text(description="Người duyệt")
    select = w.Dropdown(options=[(f"{i+1}: {r['split']} / {r['class_name']} / {Path(r['path']).name}", i) for i, r in enumerate(rows)], layout=w.Layout(width="95%"))
    canvas = w.Image(format="png")
    info, status = w.HTML(), w.HTML()
    note = w.Textarea(description="Ghi chú")
    box_index = w.Dropdown(description="Box", options=[])
    sliders = [w.IntSlider(description=n, continuous_update=False) for n in ("x1", "y1", "x2", "y2")]
    buttons = {n: w.Button(description=n) for n in ("Thêm box", "Xóa box", "Lưu box", "Duyệt tay", "Xác nhận không tay", "Để chờ", "Ảnh tiếp")}
    state = {"boxes": [], "loading": False}
    def paint():
        row = rows[select.value]
        im = Image.fromarray(read_rgb(Path(dataset["extracted"]) / row["path"]))
        draw = ImageDraw.Draw(im)
        for i, box in enumerate(state["boxes"]):
            draw.rectangle(box, outline="red", width=2)
            draw.text((box[0], box[1]), str(i + 1), fill="red")
        im.thumbnail((720, 560))
        stream = io.BytesIO()
        im.save(stream, format="PNG")
        canvas.value = stream.getvalue()
    def load_box(*args):
        if state["loading"] or box_index.value is None:
            return
        for slider, value in zip(sliders, state["boxes"][box_index.value]):
            slider.value = int(value)
    def refresh_boxes():
        box_index.options = [(str(i + 1), i) for i in range(len(state["boxes"]))]
        load_box()
        paint()
    def load(*args):
        state["loading"] = True
        row = rows[select.value]
        item = read_json(review_path)["items"][row["sample_id"]]
        state["boxes"] = [list(b) for b in item["boxes"]]
        note.value = item["note"]
        for slider, maximum in zip(sliders, [row["width"], row["height"], row["width"], row["height"]]):
            slider.max = maximum
        info.value = f"{item['state']} | flags: {', '.join(item['flags'])} | {row['width']} x {row['height']}"
        state["loading"] = False
        refresh_boxes()
    def action(name):
        try:
            row = rows[select.value]
            if name == "Thêm box":
                state["boxes"].append([0, 0, row["width"], row["height"]])
            elif name == "Xóa box" and box_index.value is not None:
                state["boxes"].pop(box_index.value)
            elif name == "Lưu box" and box_index.value is not None:
                state["boxes"][box_index.value] = clamp_box([s.value for s in sliders], row["width"], row["height"])
            elif name == "Ảnh tiếp":
                select.value = (select.value + 1) % len(rows)
            elif name in ("Duyệt tay", "Xác nhận không tay", "Để chờ"):
                mode = {"Duyệt tay": "approved", "Xác nhận không tay": "negative", "Để chờ": "pending"}[name]
                save_decision(review_path, row, mode, state["boxes"], reviewer.value, note.value)
                status.value = "Đã lưu quyết định."
            refresh_boxes()
        except Exception as exc:
            import html
            status.value = html.escape(str(exc))
    select.observe(load, names="value")
    box_index.observe(load_box, names="value")
    for name, button in buttons.items():
        button.on_click(lambda _, name=name: action(name))
    load()
    display(w.VBox([reviewer, select, info, canvas, box_index, *sliders, note, w.HBox(list(buttons.values())[:3]), w.HBox(list(buttons.values())[3:]), status]))


def review_opencv(manifest, dataset, review_path, reviewer):
    import cv2
    rows = manifest["rows"]
    index = 0
    drawing = {"start": None, "boxes": []}
    def mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing["start"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drawing["start"]:
            x0, y0 = drawing["start"]
            drawing["boxes"].append([min(x0, x), min(y0, y), max(x0, x), max(y0, y)])
            drawing["start"] = None
    try:
        cv2.namedWindow("ASL review", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("ASL review", mouse)
        while index < len(rows):
            row = rows[index]
            item = read_json(review_path)["items"][row["sample_id"]]
            rgb = read_rgb(Path(dataset["extracted"]) / row["path"])
            scale = min(900 / row["width"], 650 / row["height"], 2)
            original = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), None, fx=scale, fy=scale)
            drawing["boxes"] = [[v * scale for v in b] for b in item["boxes"]]
            print(f"{index+1}/{len(rows)} {row['split']} {row['class_name']} {row['path']} | {item['state']} | {item['flags']}")
            while True:
                frame = original.copy()
                for b in drawing["boxes"]:
                    cv2.rectangle(frame, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (0, 255, 0), 2)
                cv2.imshow("ASL review", frame)
                key = cv2.waitKey(30) & 255
                if key == ord("q") or cv2.getWindowProperty("ASL review", cv2.WND_PROP_VISIBLE) < 1:
                    return
                if key == ord("c"):
                    drawing["boxes"] = []
                elif key == ord("u") and drawing["boxes"]:
                    drawing["boxes"].pop()
                elif key in (ord("a"), ord("n")):
                    state = "approved" if key == ord("a") else "negative"
                    note = input("Resolution / reason (required for flags or no-hand): ") if item["flags"] or state == "negative" else ""
                    try:
                        save_decision(review_path, row, state, [[v / scale for v in b] for b in drawing["boxes"]], reviewer, note)
                        index += 1
                        break
                    except ValueError as exc:
                        print(exc)
                elif key == ord("s"):
                    index += 1
                    break
                elif key == ord("b"):
                    index = max(0, index - 1)
                    break
    finally:
        cv2.destroyAllWindows()
