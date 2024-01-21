from transformers import BitsAndBytesConfig
import torch

FEW_SHOT_PROMPTS = (
    "你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。"
    "USER: 翻譯成文言文：\n雅裏惱怒地說： 從前在福山田獵時，你誣陷獵官，現在又說這種話。\n答案："
    "ASSISTANT: 雅裏怒曰： 昔畋於福山，卿誣獵官，今復有此言。"
    "USER: 泰山對土石沒有好惡之心，所以能夠形成它的高大；江海對細流不加選擇，所以能夠形成它的富有。\n翻譯成古文："
    "ASSISTANT: 太山不立好惡，故能成其高；江海不擇小助，故能成其富。"
    "USER: 辛未，命吳堅為左丞相兼樞密使，常楙參知政事。\n把這句話翻譯成現代文。"
    "ASSISTANT: 初五，命令吳堅為左承相兼樞密使，常增為參知政事。"
    "USER: 議雖不從，天下鹹重其言。\n翻譯成白話文："
    "ASSISTANT: 他的建議雖然不被采納，但天下都很敬重他的話。"
    "USER: {instruction} "
    "ASSISTANT:"
)

FORMAT_PROMPTS = (
    "你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。並且以繁體中文回答。"
    "USER: {instruction} "
    "ASSISTANT:"
)

ORIGNAL_PROMPTS = (
    "你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。USER: {instruction} ASSISTANT:"
)

NO_PROMPT = ("{instruction}")



def get_prompt(instruction: str) -> str:
    '''Format the instruction as a prompt for LLM.'''
    prompt = FORMAT_PROMPTS.format(instruction=instruction)
    
    return prompt


def get_bnb_config() -> BitsAndBytesConfig:
    '''Get the BitsAndBytesConfig.'''
    
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
