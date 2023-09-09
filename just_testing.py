from PIL import Image
import matplotlib.pyplot as plt

# 이미지를 표시할 화면 크기 설정
num_images = 20  # 이미지 파일 개수 (0.png부터 20.png까지)
columns = 5      # 한 행에 표시할 이미지 개수
rows = (num_images // columns)

# 이미지를 표시할 새로운 화면 생성
fig = plt.figure(figsize=(10, 8))

for i in range(num_images):
    filename = f"{i}.png"
    try:
        img = Image.open(filename)
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(img)
        plt.axis('off')  # 이미지 축 숨기기
        plt.title(filename)
    except FileNotFoundError:
        print(f"파일 {filename}을 찾을 수 없습니다.")

plt.show()