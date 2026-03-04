from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, List
import os

from jinja2 import Environment, FileSystemLoader, TemplateNotFound, select_autoescape
from markupsafe import escape

# Read config from ENV (can be replaced by core.config later)
PROMPT_DIR = os.getenv("PROMPT_DIR", "prompts")
PROMPT_LANG = os.getenv("PROMPT_LANG", "ru")
PROMPT_VARIANT = os.getenv("PROMPT_VARIANT", "v1")
PROMPT_STRICT = os.getenv("PROMPT_STRICT", "true").lower() in {"1", "true", "yes", "y"}
PROMPT_CITE = os.getenv("PROMPT_CITE", "true").lower() in {"1", "true", "yes", "y"}


def _prompt_dir() -> Path:
    configured = Path(PROMPT_DIR)
    if configured.is_absolute():
        return configured

    cwd_candidate = Path.cwd() / configured
    if cwd_candidate.exists():
        return cwd_candidate

    module_base = Path(__file__).resolve().parents[2] / "prompts"
    return module_base


@lru_cache(maxsize=1)
def _jinja_env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(_prompt_dir())),
        autoescape=select_autoescape(enabled_extensions=(), default_for_string=True),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env


def _template_name(lang: str, variant: str) -> str:
    return f"system_{lang}_{variant}.j2"


@lru_cache(maxsize=128)
def get_system_instruction(
    lang: str | None = None,
    variant: str | None = None,
    cite: bool | None = None,
    strict: bool | None = None,
    extra_vars: Dict[str, Any] | None = None,
) -> str:
    lang = lang or PROMPT_LANG
    variant = variant or PROMPT_VARIANT
    cite = PROMPT_CITE if cite is None else cite
    strict = PROMPT_STRICT if strict is None else strict

    env = _jinja_env()
    tpl_name = _template_name(lang, variant)
    try:
        tpl = env.get_template(tpl_name)
    except TemplateNotFound:
        try:
            tpl = env.get_template(f"system_{lang}.j2")
        except TemplateNotFound:
            base = (
                "Ты — помощник, который отвечает на вопросы по внутренним документам. "
                "Говори кратко и по существу."
            )
            if strict:
                base += " Если информации недостаточно — честно скажи об этом."
            if cite:
                base += " Добавь список источников из переданных фрагментов."
            return base

    vars: Dict[str, Any] = {"cite": cite, "strict": strict, "language": lang}
    if extra_vars:
        vars.update(extra_vars)
    return tpl.render(**vars).strip()


def build_user_prompt(question: str, contexts: List[str], system_instruction: str) -> str:
    # Escape user input to prevent prompt injection via HTML/template chars
    safe_question = str(escape(question))
    ctx = "\n---\n".join(contexts)
    return f"{system_instruction}\n\nВопрос: {safe_question}\n\nКонтекстные фрагменты:\n{ctx}\n\nОтвет:"
