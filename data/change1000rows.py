# 원본 CSV 파일 경로
csv_file_path = "/home/hjs/study/Multitask-Recommendation-Library/data/AliExpress_US/AliExpress_US/train.csv"

# 새로운 CSV 파일 경로
csv_output_path = "/home/hjs/study/Multitask-Recommendation-Library/data/AliExpress_US/AliExpress_US/train_1000.csv"

# 1000행만 저장하는 코드
with open(csv_file_path, "r", encoding="utf-8") as infile, open(csv_output_path, "w", encoding="utf-8") as outfile:
    for i in range(1001):  # 헤더 + 1000개 행
        line = infile.readline()
        if not line:  # 파일 끝나면 종료
            break
        outfile.write(line)

print(f"✅ 1000행 CSV 저장 완료: {csv_output_path}")
