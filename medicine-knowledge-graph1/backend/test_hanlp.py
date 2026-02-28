from pyhanlp import HanLP

text = "感冒的症状是什么？"
segmented = HanLP(text)
print(segmented)