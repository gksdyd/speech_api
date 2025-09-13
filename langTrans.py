from googletrans import Translator
from punctuators.models import PunctCapSegModelONNX

translator = Translator()
async def trans_ko_to_eng(text: str):
  try:
    sentences = add_punctuation(text)
    full_text = " ".join(sentences)
    result = await translator.translate(full_text, src='ko', dest='en')
    return result.text
  except Exception as e:
    print(f"번역 오류: {e}")
    return -1

# import onnxruntime as ort
# print("available:", ort.get_available_providers())

model = PunctCapSegModelONNX.from_pretrained(
    "pcs_47lang",
    ort_providers=["CPUExecutionProvider"]   # ★ 문자열이 아니라 리스트
)
def add_punctuation(text: str):
  return model.infer([text])[0]

async def trans_text(separate_text):
  result = await trans_ko_to_eng(separate_text)
  if result == -1:
    return -1

  return result