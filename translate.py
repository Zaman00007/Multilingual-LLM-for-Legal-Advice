from googletrans import Translator

translator = Translator()
text_hi = "भारतीय संविधान का अनुच्छेद 32 क्या है?"
translated = translator.translate(text_hi, src='hi', dest='en')
print(translated.text)
