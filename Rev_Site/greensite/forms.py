from django import forms

class UserForm(forms.Form):
    text_reviews = forms.CharField(label = "reviews",max_length=4000)