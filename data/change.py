input_csv = "/home/hjs/study/Multitask-Recommendation-Library/data/AliExpress_US/AliExpress_US/test.csv"
output_txt = "/home/hjs/study/Multitask-Recommendation-Library/data/AliExpress_US/AliExpress_US/test.txt"

with open(input_csv, "r", encoding="utf-8") as infile, open(output_txt, "w", encoding="utf-8") as outfile:
    for line in infile:
        outfile.write(line.replace(",", " "))  # 쉼표(,)를 공백(" ")으로 변환하여 저장

print(f"변환 완료: {output_txt}")
