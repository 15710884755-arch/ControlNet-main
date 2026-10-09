import os
import re
import threading
from pathlib import Path


CHINESE_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TRANSLATION_MODEL = PROJECT_ROOT / "models" / "translation" / "opus-mt-zh-en"

# Common terms keep the UI useful when the optional translation model is absent.
FALLBACK_TRANSLATIONS = {
    "笑脸": "smiling face",
    "微笑": "smile",
    "表情": "facial expression",
    "人物": "person",
    "女人": "woman",
    "男人": "man",
    "女孩": "girl",
    "男孩": "boy",
    "儿童": "child",
    "猫": "cat",
    "小猫": "kitten",
    "狗": "dog",
    "小狗": "puppy",
    "鸟": "bird",
    "鱼": "fish",
    "老虎": "tiger",
    "狮子": "lion",
    "熊猫": "giant panda",
    "龙": "dragon",
    "机器人": "robot",
    "汽车": "car",
    "飞机": "airplane",
    "火车": "train",
    "房子": "house",
    "城堡": "castle",
    "城市": "city",
    "森林": "forest",
    "树木": "tree",
    "花朵": "flower",
    "玫瑰": "rose",
    "天空": "sky",
    "海洋": "ocean",
    "山": "mountain",
    "太阳": "sun",
    "月亮": "moon",
    "星星": "stars",
    "夜景": "night scene",
    "红色": "red",
    "蓝色": "blue",
    "绿色": "green",
    "黄色": "yellow",
    "黑色": "black",
    "白色": "white",
    "动漫": "anime",
    "写实": "photorealistic",
    "油画": "oil painting",
    "水彩": "watercolor",
    "像素": "pixel art",
}

LANDMARK_ENHANCEMENTS = (
    (
        ("广州塔", "小蛮腰", "guangzhou tower", "canton tower"),
        "Canton Tower, Guangzhou Tower, slender hyperboloid lattice TV tower, narrow twisted waist, tall antenna spire, Guangzhou city skyline, architectural landmark, photorealistic",
    ),
    (
        ("东方明珠", "oriental pearl tower"),
        "Oriental Pearl TV Tower, stacked spheres on a tall concrete pillar, Shanghai skyline, architectural landmark, photorealistic",
    ),
    (
        ("上海中心", "shanghai tower"),
        "Shanghai Tower, tall spiraling skyscraper, twisted glass facade, Shanghai skyline, architectural landmark, photorealistic",
    ),
    (
        ("埃菲尔铁塔", "eiffel tower"),
        "Eiffel Tower, iron lattice tower, wide base, narrow spire, Paris skyline, architectural landmark, photorealistic",
    ),
    (
        ("自由女神", "statue of liberty"),
        "Statue of Liberty, green copper statue, raised torch, crown, New York harbor, iconic landmark, photorealistic",
    ),
    (
        ("悉尼歌剧院", "sydney opera house"),
        "Sydney Opera House, white shell-shaped roof, harbor waterfront, iconic modern architecture, photorealistic",
    ),
)

_translator = None
_translator_path = None
_translator_lock = threading.Lock()


def contains_chinese(text):
    return bool(text and CHINESE_RE.search(text))


def _find_translation_model():
    configured_path = os.environ.get("CONTROLNET_PROMPT_TRANSLATOR_MODEL")
    candidates = [Path(configured_path)] if configured_path else []
    candidates.append(DEFAULT_TRANSLATION_MODEL)

    for candidate in candidates:
        if (
            candidate.is_dir()
            and (candidate / "config.json").is_file()
            and (candidate / "tokenizer_config.json").is_file()
            and (candidate / "source.spm").is_file()
            and (candidate / "target.spm").is_file()
            and (
                (candidate / "pytorch_model.bin").is_file()
                or (candidate / "model.safetensors").is_file()
            )
        ):
            return candidate
    return None


def _load_translator():
    global _translator, _translator_path

    model_path = _find_translation_model()
    if model_path is None:
        return None
    if _translator is not None and _translator_path == model_path:
        return _translator

    with _translator_lock:
        if _translator is not None and _translator_path == model_path:
            return _translator
        try:
            from transformers.models.marian.modeling_marian import MarianMTModel
            from transformers.models.marian.tokenization_marian import MarianTokenizer

            tokenizer = MarianTokenizer.from_pretrained(str(model_path), local_files_only=True)
            model = MarianMTModel.from_pretrained(str(model_path), local_files_only=True)
            model.to("cpu").eval()
            _translator = (tokenizer, model)
            _translator_path = model_path
        except Exception:
            _translator = None
            _translator_path = None
    return _translator


def _translate_with_model(text):
    translator = _load_translator()
    if translator is None:
        return None

    tokenizer, model = translator
    try:
        import torch

        batch = tokenizer([text], return_tensors="pt", padding=True)
        with torch.inference_mode():
            generated = model.generate(
                **batch,
                max_length=128,
                num_beams=4,
                early_stopping=True,
            )
        translated = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        return translated or None
    except Exception:
        return None


def _translate_with_fallback(text):
    stripped = text.strip()
    exact = FALLBACK_TRANSLATIONS.get(stripped)
    if exact:
        return exact

    translated = stripped
    replaced = False
    for chinese, english in sorted(FALLBACK_TRANSLATIONS.items(), key=lambda item: len(item[0]), reverse=True):
        if chinese in translated:
            translated = translated.replace(chinese, english)
            replaced = True
    translated = re.sub(r"\s+", " ", translated).strip(" ,")
    return translated if replaced else text


def _merge_prompt_phrases(base_prompt, enhancement):
    parts = [part.strip() for part in base_prompt.split(",") if part.strip()]
    known = {part.lower() for part in parts}
    for part in enhancement.split(","):
        phrase = part.strip()
        if phrase and phrase.lower() not in known:
            parts.append(phrase)
            known.add(phrase.lower())
    return ", ".join(parts)


def enhance_prompt_for_generation(prompt, translated_prompt=None):
    translated = (translated_prompt or prompt or "").strip()
    source = (prompt or "").lower()
    translated_lower = translated.lower()

    enhancements = []
    for triggers, description in LANDMARK_ENHANCEMENTS:
        if any(trigger.lower() in source or trigger.lower() in translated_lower for trigger in triggers):
            enhancements.append(description)

    for description in enhancements:
        translated = _merge_prompt_phrases(translated, description)
    return translated


def translate_prompt_to_english(prompt):
    if not contains_chinese(prompt):
        return prompt

    stripped = prompt.strip()
    exact = FALLBACK_TRANSLATIONS.get(stripped)
    if exact:
        return exact
    return _translate_with_model(stripped) or _translate_with_fallback(stripped)
