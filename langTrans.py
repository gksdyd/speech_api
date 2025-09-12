from googletrans import Translator
from punctuators.models import PunctCapSegModelONNX

translator = Translator()
async def trans_ko_to_eng(text: str, debug: bool = False):
  try:
    sentences = add_punctuation(text)
    full_text = " ".join(sentences)
    result = await translator.translate(full_text, src='ko', dest='en')
    return result.text
  except Exception as e:
    if debug:
      print(f"번역 오류: {e}")
    return -1

model = PunctCapSegModelONNX.from_pretrained("pcs_47lang")
def add_punctuation(text: str):
  return model.infer([text])[0]

async def trans_text(separate_text, debug: bool = False):
  result = await trans_ko_to_eng(separate_text, debug)
  if result == -1:
    return -1

  return result