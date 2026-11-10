"""Telegram-обработчики (aiogram 3). Вся логика — в SaqBolCore; здесь только UI."""
from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, Message
from aiogram.utils.keyboard import InlineKeyboardBuilder, ReplyKeyboardBuilder

from saqbol.bot.logic import CheckResult, SaqBolCore
from saqbol.bot.texts import LANG_NAMES, all_variants, t

log = logging.getLogger("saqbol.bot")
router = Router(name="saqbol")
# бот работает только в личных чатах: в группах он не читает и не анализирует переписку
router.message.filter(F.chat.type == "private")


class Flow(StatesGroup):
    clarify = State()  # ждём ответ на единственный уточняющий вопрос
    voice_confirm = State()  # ждём подтверждения/исправления распознанного текста


# ---------- клавиатуры ----------
def main_menu(lang: str):
    kb = ReplyKeyboardBuilder()
    kb.button(text=t(lang, "btn_check"))
    kb.button(text=t(lang, "btn_link"))
    kb.button(text=t(lang, "btn_contact"))
    kb.button(text=t(lang, "btn_delete"))
    kb.adjust(1, 2, 1)
    return kb.as_markup(resize_keyboard=True)


def lang_kb():
    kb = InlineKeyboardBuilder()
    for code, name in LANG_NAMES.items():
        kb.button(text=name, callback_data=f"lang:{code}")
    return kb.as_markup()


def result_kb(lang: str, has_relative: bool):
    kb = InlineKeyboardBuilder()
    kb.button(text=t(lang, "btn_wrong"), callback_data="fb:wrong")
    kb.button(text=t(lang, "btn_ok"), callback_data="fb:ok")
    if has_relative:
        kb.button(text=t(lang, "btn_contact"), callback_data="contact")
    kb.adjust(2, 1)
    return kb.as_markup()


def yes_no_kb(lang: str, yes_key: str, yes_data: str, no_key: str, no_data: str):
    kb = InlineKeyboardBuilder()
    kb.button(text=t(lang, yes_key), callback_data=yes_data)
    kb.button(text=t(lang, no_key), callback_data=no_data)
    return kb.as_markup()


def settings_kb(lang: str):
    kb = InlineKeyboardBuilder()
    kb.button(text=t(lang, "btn_toggle_notify"), callback_data="set:notify")
    kb.button(text=t(lang, "btn_toggle_share"), callback_data="set:share")
    kb.button(text=t(lang, "btn_unlink"), callback_data="set:unlink")
    kb.button(text=LANG_NAMES["ru"] + " / " + LANG_NAMES["kk"], callback_data="set:lang")
    kb.adjust(1)
    return kb.as_markup()


async def deliver(bot: Bot, res: CheckResult) -> None:
    for out in res.outgoing:
        try:
            await bot.send_message(out.chat_id, out.text)
        except Exception as e:  # близкий мог заблокировать бота
            log.warning("delivery failed kind=%s err=%s", out.kind, type(e).__name__)


async def send_check(message: Message, core: SaqBolCore, state: FSMContext, bot: Bot, text: str,
                     allow_clarification: bool = True, original: str | None = None) -> None:
    uid = message.chat.id
    name = message.chat.full_name if message.chat else ""
    if original is not None:
        res = await asyncio.to_thread(core.check_with_clarification, uid, original, text, name)
    else:
        res = await asyncio.to_thread(core.check, uid, text, name, allow_clarification)
    lang = core.lang(uid)
    user = core.db.get_user(uid)
    data = {}
    if res.analysis:
        data = {"last_level": res.analysis.level, "last_model": res.analysis.model_name}
        # текст держим только в оперативной памяти и только при отдельном согласии
        if user and user.share_examples:
            data["last_text"] = text if original is None else f"{original}\n{text}"
    await state.clear()
    await state.update_data(**data)
    if res.ask_clarification:
        await state.set_state(Flow.clarify)
        await state.update_data(original=text)
    markup = result_kb(lang, bool(core.db.active_relatives(uid))) if res.analysis else None
    await message.answer(res.reply, reply_markup=markup)
    await deliver(bot, res)


# ---------- команды ----------
@router.message(CommandStart())
async def cmd_start(message: Message, command: CommandObject, core: SaqBolCore, state: FSMContext):
    await state.clear()
    uid = message.from_user.id
    core.db.ensure_user(uid)
    lang = core.lang(uid)
    payload = command.args or ""
    if payload.startswith("link_"):
        token = payload[5:]
        elder_id, err = core.inspect_invite(token)
        if err:
            await message.answer(t(lang, err))
            return
        try:
            chat = await message.bot.get_chat(elder_id)
            elder_name = chat.full_name
        except Exception:
            elder_name = "—"
        await message.answer(
            t(lang, "relative_offer", name=elder_name),
            reply_markup=yes_no_kb(lang, "btn_agree", f"link:accept:{token}", "btn_decline", "link:decline"),
        )
        return
    if core.db.get_user(uid).lang is None:
        await message.answer(t("ru", "choose_lang"), reply_markup=lang_kb())
        return
    await message.answer(t(lang, "start"), reply_markup=main_menu(lang))


@router.message(Command("help"))
async def cmd_help(message: Message, core: SaqBolCore):
    lang = core.lang(message.from_user.id)
    await message.answer(t(lang, "help"), reply_markup=main_menu(lang))


@router.message(Command("language"))
async def cmd_language(message: Message):
    await message.answer(t("ru", "choose_lang"), reply_markup=lang_kb())


@router.message(Command("settings"))
async def cmd_settings(message: Message, core: SaqBolCore):
    uid = message.from_user.id
    lang = core.lang(uid)
    await message.answer(core.settings_text(uid) + "\n\n" + t(lang, "share_explain"), reply_markup=settings_kb(lang))


@router.message(Command("privacy"))
async def cmd_privacy(message: Message, core: SaqBolCore):
    await message.answer(t(core.lang(message.from_user.id), "privacy"))


# ---------- кнопки главного меню ----------
@router.message(F.text.in_(all_variants("btn_check")))
async def btn_check(message: Message, core: SaqBolCore, state: FSMContext):
    await state.clear()
    await message.answer(t(core.lang(message.from_user.id), "ask_text"))


@router.message(F.text.in_(all_variants("btn_link")))
async def btn_link(message: Message, core: SaqBolCore):
    lang = core.lang(message.from_user.id)
    await message.answer(t(lang, "link_explain"),
                         reply_markup=yes_no_kb(lang, "btn_yes_allow", "link:allow", "btn_no", "link:no"))


@router.message(F.text.in_(all_variants("btn_contact")))
async def btn_contact(message: Message, core: SaqBolCore, bot: Bot):
    res = core.contact_relative(message.from_user.id, message.from_user.full_name)
    await message.answer(res.reply)
    await deliver(bot, res)


@router.message(F.text.in_(all_variants("btn_delete")))
async def btn_delete(message: Message, core: SaqBolCore):
    lang = core.lang(message.from_user.id)
    await message.answer(t(lang, "delete_confirm"),
                         reply_markup=yes_no_kb(lang, "btn_delete_yes", "del:yes", "btn_cancel", "del:no"))


# ---------- callback-кнопки ----------
@router.callback_query(F.data.startswith("lang:"))
async def cb_lang(cb: CallbackQuery, core: SaqBolCore):
    lang = cb.data.split(":", 1)[1]
    core.db.set_language(cb.from_user.id, lang)
    await cb.message.answer(t(lang, "lang_set"))
    await cb.message.answer(t(lang, "start"), reply_markup=main_menu(lang))
    await cb.answer()


@router.callback_query(F.data == "link:allow")
async def cb_link_allow(cb: CallbackQuery, core: SaqBolCore, bot: Bot):
    me = await bot.get_me()
    await cb.message.answer(core.create_invite(cb.from_user.id, me.username))
    await cb.answer()


@router.callback_query(F.data.in_({"link:no", "link:decline"}))
async def cb_link_no(cb: CallbackQuery, core: SaqBolCore):
    await cb.message.answer(t(core.lang(cb.from_user.id), "link_cancelled"))
    await cb.answer()


@router.callback_query(F.data.startswith("link:accept:"))
async def cb_link_accept(cb: CallbackQuery, core: SaqBolCore, bot: Bot):
    token = cb.data.split(":", 2)[2]
    rel_id, rel_name = cb.from_user.id, cb.from_user.full_name
    lang = core.lang(rel_id)
    link_id, elder_id, err = core.accept_invite(token, rel_id, rel_name)
    if err:
        await cb.message.answer(t(lang, err))
        await cb.answer()
        return
    elder_lang = core.lang(elder_id)
    try:
        elder_name = (await bot.get_chat(elder_id)).full_name
    except Exception:
        elder_name = "—"
    await cb.message.answer(t(lang, "relative_waiting", name=elder_name))
    await bot.send_message(
        elder_id, t(elder_lang, "elder_confirm", name=rel_name),
        reply_markup=yes_no_kb(elder_lang, "btn_confirm", f"link:confirm:{link_id}", "btn_reject", f"link:reject:{link_id}"),
    )
    await cb.answer()


@router.callback_query(F.data.startswith("link:confirm:"))
async def cb_link_confirm(cb: CallbackQuery, core: SaqBolCore, bot: Bot):
    link_id = int(cb.data.rsplit(":", 1)[1])
    elder_id = cb.from_user.id
    relative_id = core.elder_confirm(elder_id, link_id)
    if relative_id is None:
        await cb.answer(t(core.lang(elder_id), "link_err_not_found"), show_alert=True)
        return
    name = core.db.get_link(link_id)["relative_name"] or "—"
    await cb.message.answer(t(core.lang(elder_id), "link_active_elder", name=name),
                            reply_markup=main_menu(core.lang(elder_id)))
    await bot.send_message(relative_id, t(core.lang(relative_id), "link_active_relative"))
    await cb.answer()


@router.callback_query(F.data.startswith("link:reject:"))
async def cb_link_reject(cb: CallbackQuery, core: SaqBolCore):
    core.db.reject_link(cb.from_user.id, int(cb.data.rsplit(":", 1)[1]))
    await cb.message.answer(t(core.lang(cb.from_user.id), "link_rejected"))
    await cb.answer()


@router.callback_query(F.data == "contact")
async def cb_contact(cb: CallbackQuery, core: SaqBolCore, bot: Bot):
    res = core.contact_relative(cb.from_user.id, cb.from_user.full_name)
    await cb.message.answer(res.reply)
    await deliver(bot, res)
    await cb.answer()


@router.callback_query(F.data.startswith("fb:"))
async def cb_feedback(cb: CallbackQuery, core: SaqBolCore, state: FSMContext):
    kind = cb.data.split(":", 1)[1]
    data = await state.get_data()
    msg = core.feedback(cb.from_user.id, kind, data.get("last_level"), data.get("last_model"), data.get("last_text"))
    await state.update_data(last_text=None)
    await cb.answer(msg, show_alert=kind == "wrong")


@router.callback_query(F.data.startswith("del:"))
async def cb_delete(cb: CallbackQuery, core: SaqBolCore, state: FSMContext):
    lang = core.lang(cb.from_user.id)
    if cb.data == "del:yes":
        await state.clear()
        await cb.message.answer(core.delete_all(cb.from_user.id))
    else:
        await cb.message.answer(t(lang, "delete_cancelled"))
    await cb.answer()


@router.callback_query(F.data.startswith("set:"))
async def cb_settings(cb: CallbackQuery, core: SaqBolCore):
    uid = cb.from_user.id
    u = core.db.ensure_user(uid)
    action = cb.data.split(":", 1)[1]
    lang = core.lang(uid)
    if action == "notify":
        core.db.set_notify(uid, not u.notify_enabled)
    elif action == "share":
        core.db.set_share_examples(uid, not u.share_examples)
    elif action == "unlink":
        core.db.unlink_all(uid)
        await cb.message.answer(t(lang, "unlinked"))
    elif action == "lang":
        await cb.message.answer(t(lang, "choose_lang"), reply_markup=lang_kb())
        await cb.answer()
        return
    await cb.message.answer(core.settings_text(uid), reply_markup=settings_kb(lang))
    await cb.answer()


@router.callback_query(F.data == "voice:check")
async def cb_voice_check(cb: CallbackQuery, core: SaqBolCore, state: FSMContext, bot: Bot):
    data = await state.get_data()
    text = data.get("voice_text", "")
    await cb.answer()
    await send_check(cb.message, core, state, bot, text)


# ---------- содержимое ----------
@router.message(Flow.clarify, F.text)
async def on_clarification(message: Message, core: SaqBolCore, state: FSMContext, bot: Bot):
    data = await state.get_data()
    await send_check(message, core, state, bot, message.text, original=data.get("original", ""))


@router.message(Flow.voice_confirm, F.text)
async def on_voice_correction(message: Message, core: SaqBolCore, state: FSMContext, bot: Bot):
    await send_check(message, core, state, bot, message.text)  # пользователь прислал исправленный текст


@router.message(F.text)
async def on_text(message: Message, core: SaqBolCore, state: FSMContext, bot: Bot):
    if message.text.startswith("/"):
        await message.answer(t(core.lang(message.from_user.id), "help"))
        return
    await send_check(message, core, state, bot, message.text)


@router.message(F.caption)
async def on_caption(message: Message, core: SaqBolCore, state: FSMContext, bot: Bot):
    await send_check(message, core, state, bot, message.caption)


@router.message(F.voice)
async def on_voice(message: Message, core: SaqBolCore, state: FSMContext, bot: Bot):
    from saqbol.voice.asr import get_transcriber, temporary_audio_path

    lang = core.lang(message.from_user.id)
    if not core.settings.voice_enabled:
        await message.answer(t(lang, "voice_disabled"))
        return
    if (message.voice.duration or 0) > core.settings.max_voice_seconds:
        await message.answer(t(lang, "voice_too_long", sec=core.settings.max_voice_seconds))
        return
    try:
        with temporary_audio_path(".ogg") as path:  # файл удаляется при выходе из блока
            await bot.download(message.voice, destination=path)
            transcriber = get_transcriber(core.settings.whisper_model)
            tr = await asyncio.to_thread(transcriber.transcribe, path)
        log.info("asr latency_s=%s", tr.asr_latency_s)
    except Exception as e:
        log.warning("asr failed: %s", type(e).__name__)
        await message.answer(t(lang, "voice_failed"))
        return
    if not tr.text:
        await message.answer(t(lang, "voice_failed"))
        return
    await state.set_state(Flow.voice_confirm)
    await state.update_data(voice_text=tr.text)
    kb = InlineKeyboardBuilder()
    kb.button(text=t(lang, "btn_check_this"), callback_data="voice:check")
    await message.answer(t(lang, "voice_recognized", text=tr.text), reply_markup=kb.as_markup())


@router.message()
async def on_unsupported(message: Message, core: SaqBolCore):
    await message.answer(t(core.lang(message.from_user.id), "unsupported"))
