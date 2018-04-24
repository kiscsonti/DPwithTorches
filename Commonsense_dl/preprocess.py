
from xml.etree import ElementTree


class Text(object):

    def __init__(self, text):
        self.text = text
        self.questions = []

    def __iadd__(self, other):
        self.questions.append(other)
        return self

    def __str__(self):
        print('xxxTEXTxxx\n', self.text)
        for q in self.questions:
            print(q)
        return ""

    def get_len(self):
        return len(self.questions)


class Question(object):

    def __init__(self, question, a1, a2):
        self.question = question
        self.a1 = a1
        self.a2 = a2

    def __str__(self):
        print('QUESTION\n', self.question, '\n', self.a1, '\n', self.a2)
        return ""

class Corpus(object):

    def __init__(self):
        self.texts = []
        self.hatar = 0
        # nem biztos hogy ez kell
        self.vocab = []

    def __iadd__(self, other: Text):
        self.texts.append(other)
        return self

    def set_hatar(self):
        self.hatar = len(self.texts)

    def __str__(self):
        for t in self.texts:
            print(t)
        return ""

    def get_len(self):
        osszeg = 0
        for item in self.texts[:self.hatar]:
            osszeg += item.get_len()
        return osszeg


"""
def get_vocabulary_and_data(*args):
    vocabulary = set()
    datas = []

    global vectorizer
    analyze = vectorizer.build_analyzer()

    for elem in args:
        data = MyData()
        print(elem)
        parser = ElementTree.parse(elem)
        instances = parser.findall('instance')

        for inst in instances:
            text = inst.find('text').text
            corpus = [text]
            questions = inst.find('questions').findall('question')

            for q in questions:
                data_answer = []
                ans = q.findall('answer')
                targ = ans[0].attrib['correct']
                if targ == "True":
                    targ = 1
                else:
                    targ = 0
                data.target.append(targ)
                corpus.append(ans[0].attrib['text'])
                corpus.append(ans[1].attrib['text'])

                data_answer.append(ans[0].attrib['text'])
                data_answer.append(ans[1].attrib['text'])
                data.data.append(data_answer)

            for line in corpus:
                tmp_voc = analyze(line)
                vocabulary.update(tmp_voc)

        datas.append(data)

    print("Vocabulary Done")

    return vocabulary, datas
"""

def get_data(*args):
    corp = Corpus()
    for elem in args:

        print(elem)
        parser = ElementTree.parse(elem)
        instances = parser.findall('instance')

        for inst in instances:
            text = inst.find('text').text
            t = Text(text=text)
            questions = inst.find('questions').findall('question')

            for q in questions:
                data_answer = []
                ans = q.findall('answer')
                targ = ans[0].attrib['correct']
                if targ == "True":
                    targ = 1
                else:
                    targ = 0

                q = Question(question=q.attrib['text'], a1=(ans[0].attrib['text'], targ), a2=(ans[1].attrib['text'], 1-targ))
                t += q
            corp += t
        if corp.hatar == 0:
            corp.set_hatar()

    print("Reading Done")

    return corp
