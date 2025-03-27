import csv
import random

# 🔹 입력 파일 & 출력 파일 경로
input_file = "./AliExpress_US/test.csv"  # 원본 데이터
output_file = "./AliExpress_US/test_new.csv"  # 새로운 데이터

# 🔹 설정값 (저장할 샘플 수)
MAX_SAMPLES = 1000  # 최종 생성할 데이터 개수
ZERO_SAMPLE_RATIO = 0.01  # 클릭 & 구매 0인 데이터 샘플링 비율

# 🔹 카운트 변수
selected_count = 0
zero_click_count = 0

# 🔹 새 파일에 저장 (실시간으로 한 줄씩 처리)
with open(input_file, "r") as infile, open(output_file, "w", newline="") as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    header = next(reader)  # 첫 줄 (컬럼 이름) 읽기
    writer.writerow(header)  # 헤더 저장

    for row in reader:
        click = int(row[-2])  # click 열 (뒤에서 2번째)
        conversion = int(row[-1])  # conversion 열 (마지막 열)

        if click == 1 or conversion == 1:
            writer.writerow(row)  # 클릭 or 구매한 경우 즉시 저장
            selected_count += 1
        elif random.random() < ZERO_SAMPLE_RATIO:  # 일부만 샘플링하여 저장
            writer.writerow(row)
            zero_click_count += 1

        # 최대 샘플 개수 도달하면 중지
        if selected_count + zero_click_count >= MAX_SAMPLES:
            break

print(f"✅ {output_file} 생성 완료! 최종 샘플 개수: {selected_count + zero_click_count}")
print(f"📊 클릭 1 또는 구매 1 데이터 개수: {selected_count}")
print(f"📊 샘플링된 클릭 0 & 구매 0 데이터 개수: {zero_click_count}")
