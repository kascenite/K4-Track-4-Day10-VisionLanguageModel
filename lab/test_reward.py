import json

from lab.reward import reward

GT = [{"label": "pothole", "bbox": (100, 100, 200, 200)},
      {"label": "crack", "bbox": (500, 500, 700, 600)}]


def ans(*boxes):
    return json.dumps([{"label": l, "bbox_2d": list(b)} for l, b in boxes])


def test_perfect():
    assert reward(ans(("pothole", (100, 100, 200, 200)), ("crack", (500, 500, 700, 600))), GT) == 1.0


def test_broken_json():
    assert reward('[{"label": "pothole", "bbox_2d": [1, 2', GT) == 0.0
    assert reward("no damage here", GT) == 0.0


def test_wrong_class_gets_only_format_reward():
    assert reward(ans(("manhole", (100, 100, 200, 200)), ("manhole", (500, 500, 700, 600))), GT) == 0.2


def test_extra_boxes_are_penalised():
    r_ok = reward(ans(("pothole", (100, 100, 200, 200)), ("crack", (500, 500, 700, 600))), GT)
    r_extra = reward(ans(("pothole", (100, 100, 200, 200)), ("crack", (500, 500, 700, 600)),
                         ("crack", (0, 0, 50, 50)), ("pothole", (800, 800, 900, 900))), GT)
    assert r_extra == 0.2 + 0.8 * 2 / 4 < r_ok


def test_empty():
    assert reward("[]", []) == 1.0
    assert reward("[]", GT) == 0.2
    assert reward(ans(("crack", (0, 0, 10, 10))), []) == 0.2


def test_giant_box_not_rewarded():
    assert reward(ans(("pothole", (0, 0, 1000, 1000)), ("crack", (0, 0, 1000, 1000))), GT) == 0.2


def test_partial_iou_and_fenced_json():
    r = reward("```json\n" + ans(("pothole", (100, 100, 200, 220))) + "\n```", GT[:1])
    assert abs(r - (0.2 + 0.8 * (100 * 100) / (100 * 120))) < 1e-9
