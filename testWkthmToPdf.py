import pdfkit

fileName = "Test.pdf"

options = {
    "page-size": "A4",
    "margin-top": "5mm",
    "margin-right": "5mm",
    "margin-bottom": "5mm",
    "margin-left": "5mm",
    "encoding": "UTF-8",
    "enable-local-file-access": True,
}
config = None

pdfkit.from_file("debugFile.html",fileName, options=options, configuration=config)