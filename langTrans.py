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
    return None

model = PunctCapSegModelONNX.from_pretrained("pcs_47lang")
def add_punctuation(text: str):
  return model.infer([text])[0]

async def trans_text(separate_text):
  result = []
  for text in separate_text:
    temp = await trans_ko_to_eng(text[1])
    if temp is not None:
      result.append([text[0], temp])
  return result