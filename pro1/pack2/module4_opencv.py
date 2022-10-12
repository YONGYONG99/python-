# opencv(Computer vision) : 이미지/영상/동양상 처리에 사용할 수 있는 오픈 소스 라이브러리
# pip install opencv-python

import cv2
# print(cv.__verstion__)

# 이미지 읽기
img1 = cv2.imread('./sajin.jpg')
print(type(img1))

# img1 = cv2.imread('./sajin.jpg', cv2.IMREAD_COLOR) # 채널(channel) : 3 (R,G,B)
# img1 = cv2.imread('./sajin.jpg', cv2.IMREAD_GRAYSCALE) # 채널(channel) : 1
img1 = cv2.imread('./sajin.jpg', cv2.IMREAD_REDUCED_COLOR_2)
cv2.imshow('image test', img1)
cv2.waitKey()
cv2.destroyAllWindows()

# 이미지를 다른 이름으로 저장
cv2.imwrite('./sajin2.jpg', img1)

# 이미지 크기 조절
img2 = cv2.resize(img1, (320,100), interpolation=cv2.INTER_AREA)
cv2.imwrite('./sajin3.jpg', img2)

# 이미지 상하좌우 대칭 (Flip)
# a = cv2.imshow('image rotation', cv2.flip(img1, flipCode=0)) # 이미지 뒤집기
a = cv2.imshow('image rotation', cv2.flip(img1, flipCode=1)) 
cv2.waitKey()
cv2.destroyAllWindows()
