import spacy
nlp = spacy.load('vi_spacy_model')
doc = nlp('Vợ tôi, chắc cũng như nhiều phụ nữ hiện đại khác, thường đứng trước tủ quần áo khổng lồ của mình và than rất thực lòng rằng không có gì để mặc ngày mai cả')
for token in doc:
    print(token.text)