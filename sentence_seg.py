import pysbd

class SentenceSegmentor:
    def __init__(self) -> None:
        self.seg = pysbd.Segmenter(language="zh", clean=False)
    def cut_sentence(self, text):
        return self.seg.segment(text)