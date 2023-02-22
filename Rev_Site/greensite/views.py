from django.shortcuts import render
from .forms import UserForm

from keras.models import load_model
import pickle
from keras.utils import pad_sequences
import langid
from googletrans import Translator
import re

model = load_model("greensite/GA17.h5",compile = False)
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
with open('/TokenizerReviews.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
translator = Translator()

def index(request):
    return render(request,"greensite/index.html")

def get_reviews(request):
    review_text = ""
    final_text = ""
    if request.method == "POST":
        #stbutton = request.POST.get("submit")
        form = UserForm(request.POST or None)

        #form.cleaned_data["reviews_2 "]
        if form.is_valid():
            review_text = form.cleaned_data.get("text_reviews")
        else:
            review_text = ""
    if review_text != "":
        if langid.classify(review_text)[0] != "en":
            review_text = translator.translate(review_text).text.lower()
        review_text = re.sub(r"[^a-z]"," ",review_text)
        review_text = tokenizer.texts_to_sequences([review_text])
        review_text = pad_sequences(review_text, maxlen=350)
        result = model.predict(review_text)[0]
        result = list(result).index(max(result))
        if result > 5:
            final_text = f"Review evaluation - {result} \n The review is positive"
        else:
            final_text = f"Review evaluation - {result} \n The review is negative"

    data = {"reviews_2": final_text}
    return render(request, "greensite/index.html", context=data)