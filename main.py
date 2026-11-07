import os
import json
import time
import asyncio
import re
from typing import Dict, List, Any, Optional

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star
from astrbot.api.provider import ProviderRequest
from astrbot.api import logger, AstrBotConfig

PLUGIN_NAME = "astrbot_plugin_slang_miner"

class SlangMinerPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.plugin_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_file = os.path.join(self.plugin_dir, "slang_dict.json")
        
        # 运行时内存数据
        self.buffers: Dict[str, List[Dict[str, Any]]] = {} # group_id -> list of msgs
        self.slangs: Dict[str, Dict[str, Any]] = self._load_dict()
        self._lock = asyncio.Lock()
        self._extracting = False

    def _load_dict(self) -> Dict[str, Dict[str, Any]]:
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"[{PLUGIN_NAME}] 加载黑话词典失败: {e}")
        return {}

    @staticmethod
    def _response_text(res: Any) -> str:
        """兼容不同 AstrBot / Provider 版本的返回体，统一取出纯文本"""
        # 1. 标准 LLMResponse.completion_text
        text = getattr(res, "completion_text", None)
        if isinstance(text, str) and text.strip():
            return text
        # 2. 直接返回 str 的 provider
        if isinstance(res, str):
            return res
        # 3. 老版本 / 第三方 provider 的 raw_completion
        raw = getattr(res, "raw_completion", None)
        if isinstance(raw, str) and raw.strip():
            return raw
        # 4. OpenAI 风格 ChatCompletion 对象
        choices = getattr(raw, "choices", None)
        if choices:
            message = getattr(choices[0], "message", None)
            content = getattr(message, "content", None)
            if isinstance(content, str):
                return content
        # 5. 兜底
        for attr in ("result", "text", "content"):
            val = getattr(res, attr, None)
            if isinstance(val, str) and val.strip():
                return val
        return ""

    def _save_dict(self):
        try:
            with open(self.data_file, "w", encoding="utf-8") as f:
                json.dump(self.slangs, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"[{PLUGIN_NAME}] 保存黑话词典失败: {e}")

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def on_group_message(self, event: AstrMessageEvent):
        """旁路偷听群消息"""
        if not self.config.get("enabled", True):
            return

        group_id = str(event.message_obj.group_id)
        target_groups = self.config.get("target_groups", [])
        if target_groups and group_id not in [str(g) for g in target_groups]:
            return

        text = event.message_str.strip()
        # 忽略过短、指令类或纯表情消息
        if len(text) < 2 or text.startswith(("/", "!", "大魔王", "唯笑")):
            return

        sender_id = str(event.get_sender_id())
        sender_name = event.get_sender_name() or sender_id

        # 压入环形缓冲
        if group_id not in self.buffers:
            self.buffers[group_id] = []

        self.buffers[group_id].append({
            "sender_id": sender_id,
            "sender_name": sender_name,
            "text": text,
            "time": time.time()
        })

        # 检查是否达到批处理阈值
        threshold = self.config.get("buffer_size", 40)
        if len(self.buffers[group_id]) >= threshold and not self._extracting:
            # 异步启动萃取任务，完全不阻塞当前事件
            batch = self.buffers[group_id][:threshold]
            self.buffers[group_id] = self.buffers[group_id][threshold:]
            asyncio.create_task(self._process_batch(group_id, batch))

    async def _process_batch(self, group_id: str, batch: List[Dict[str, Any]]):
        """调用背景轻量 LLM 提炼黑话/梗"""
        async with self._lock:
            self._extracting = True
            try:
                # 构造对话文本流
                lines = []
                for m in batch:
                    lines.append(f"{m['sender_name']}: {m['text']}")
                chat_log = "\n".join(lines)

                prompt = (
                    "你是一个资深网络亚文化与群聊切片分析专家。以下是一段群聊消息记录：\n"
                    "```\n"
                    f"{chat_log}\n"
                    "```\n"
                    "请分析并提取其中【本群特有的高频词、黑话、自创缩写、特定梗或内部代称】。\n"
                    "排除通用的日常词汇、普通网络流行语（如笑死、确实、牛逼等）。\n"
                    "如果发现特定梗，请给出简短的含义推测与解释。\n"
                    "严格以 JSON 格式输出，不要有任何多余文字或 markdown 标记，格式如下：\n"
                    "[\n"
                    "  {\"term\": \"词语或梗名\", \"meaning\": \"群内含义简要推断\", \"example\": \"原文例句\"}\n"
                    "]\n"
                    "如果没有明显特有梗，返回空数组 []。"
                )

                provider_id = self.config.get("analysis_provider_id")
                # 动态获取 LLM provider
                provider = None
                if provider_id:
                    provider = self.context.get_provider_by_id(provider_id)
                if not provider:
                    # 回退到默认 LLM
                    provider = self.context.get_using_provider()

                if not provider:
                    logger.warning(f"[{PLUGIN_NAME}] 未找到可用的 LLM Provider 进行黑话萃取")
                    return

                res = await provider.text_chat(prompt)
                raw_text = self._response_text(res).strip()
                if not raw_text:
                    logger.warning(f"[{PLUGIN_NAME}] 萃取返回空文本，跳过本轮")
                    return
                
                # 尝试清洗与解析 JSON
                clean_json = re.sub(r"^```json\s*|^```\s*|```$", "", raw_text, flags=re.MULTILINE).strip()
                try:
                    data = json.loads(clean_json)
                    if isinstance(data, list):
                        updated = False
                        for item in data:
                            term = item.get("term", "").strip()
                            meaning = item.get("meaning", "").strip()
                            example = item.get("example", "").strip()

                            if not term or len(term) > 20:
                                continue

                            if term in self.slangs:
                                self.slangs[term]["count"] = self.slangs[term].get("count", 0) + 1
                                self.slangs[term]["last_seen"] = int(time.time())
                                # 保留更新解释
                                if meaning:
                                    self.slangs[term]["meaning"] = meaning
                            else:
                                self.slangs[term] = {
                                    "term": term,
                                    "meaning": meaning,
                                    "example": example,
                                    "count": 1,
                                    "first_seen": int(time.time()),
                                    "last_seen": int(time.time()),
                                    "group_id": group_id
                                }
                            updated = True
                            logger.info(f"[{PLUGIN_NAME}] 潜伏捕获新黑话/梗: {term} -> {meaning} (频次: {self.slangs[term]['count']})")

                        if updated:
                            self._save_dict()
                except Exception as parse_err:
                    logger.debug(f"[{PLUGIN_NAME}] 萃取结果解析跳过: {parse_err}")

            except Exception as e:
                logger.error(f"[{PLUGIN_NAME}] 异步萃取过程异常: {e}")
            finally:
                self._extracting = False

    @filter.on_llm_request()
    async def inject_slang_context(self, event: AstrMessageEvent, req: ProviderRequest):
        """当发生正常回复时，若群友消息命中了已转正的黑话，自动将含义以隐蔽背景挂载进上下文"""
        if not self.config.get("auto_inject", True):
            return

        # 私聊直接跳过群黑话注入，避免污染私聊上下文
        if event.is_private_chat():
            return

        msg = event.message_str
        if not msg:
            return

        min_conf = self.config.get("min_confidence_to_inject", 3)
        hits = []

        # 扫描命中的黑话词条
        for term, data in self.slangs.items():
            if data.get("count", 0) >= min_conf:
                if term in msg:
                    hits.append(f"【{term}】：{data.get('meaning', '群内特定梗')}")

        if hits:
            hint = f"\n[后台群语境背景参考（群友黑话与调侃，自然理解即可，无需刻意声明）]:\n" + "\n".join(hits)
            # 优雅地追加进 system prompt
            req.system_prompt += hint
            logger.debug(f"[{PLUGIN_NAME}] 命中群友黑话并注入上下文: {[h.split('：')[0] for h in hits]}")
