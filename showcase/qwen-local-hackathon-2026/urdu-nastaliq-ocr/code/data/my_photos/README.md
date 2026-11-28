# Your own test photos (the "real world" test)

Put photos you take yourself here. They are **test only** and never used for training.

For each photo add a text file with the **exact** Urdu text it contains, same name:

```text
data/my_photos/
  newspaper_01.jpg
  newspaper_01.txt      <- exact Urdu text in that photo
  book_page_03.jpg
  book_page_03.txt
```

Tips:
- Start with **single lines or 1–3 line crops** (books, newspaper headlines, signboards, notices). Crop on your phone before saving.
- Mix sources: newspaper, book, government notice, shop sign, medicine leaflet, school book.
- Mix conditions: good light, slightly tilted, low light. That's what real users do.
- 30–50 photos are enough for an honest real-world number.

Then rebuild the manifest: `python -m uocr.data.build`
