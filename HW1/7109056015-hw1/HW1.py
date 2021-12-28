from PIL import Image
import numpy as np

#算八參數
def get_eight():
    #舊座標
    old_x = [135,1367,7,1475]
    old_y = [345,377,745,771]
    #新座標
    x = [0,1440,0,1440]
    y = [0,0,450,450]

    A = np.array([[x[0], y[0], x[0]*y[0], 1],
                [x[1], y[1], x[1]*y[1], 1],
                [x[2], y[2], x[2]*y[2], 1],
                [x[3], y[3], x[3]*y[3], 1]])

    B1 = np.array([old_x[0],old_x[1],old_x[2],old_x[3]])
    B2 = np.array([old_y[0],old_y[1],old_y[2],old_y[3]])

    X1 = np.linalg.solve(A, B1)
    X2 = np.linalg.solve(A, B2)

    eight = np.hstack((X1,X2)) #合併X1 X2

    print(eight)
    return eight
    

def transform(image1,image2,eight): #舊圖、新圖、八參數

    # 取得寬度、高度
    width1, height1 = image1.size
    width2, height2 = image2.size

    for i in range(width2):
        for j in range(height2):
            
            # 通過八參數取得點座標
            x1 = (eight[0]*i) + (eight[1]*j) + (eight[2]*i*j) + eight[3]
            y1 = (eight[4]*i) + (eight[5]*j) + (eight[6]*i*j) + eight[7]

            # x,y的整數部分
            x = int(x1)
            y = int(y1)

            # 計算距離
            a = x1 - x
            b = y1 - y

            # 依照距離調整權重
            R = int((1-a) * (1-b) * pixel1[x, y][0] + a * (1-b) * pixel1[x+1, y][0] + (1-a) * b * pixel1[x, y+1][0] + a * b * pixel1[x+1, y+1][0]) 
            G = int((1-a) * (1-b) * pixel1[x, y][1] + a * (1-b) * pixel1[x+1, y][1] + (1-a) * b * pixel1[x, y+1][1] + a * b * pixel1[x+1, y+1][1])
            B = int((1-a) * (1-b) * pixel1[x, y][2] + a * (1-b) * pixel1[x+1, y][2] + (1-a) * b * pixel1[x, y+1][2] + a * b * pixel1[x+1, y+1][2])

            # 給新圖像素的值
            pixel2[i, j] = (R, G, B)

#讀圖
image = Image.open('IMG.jpg')

#算參數
eight = get_eight()

# 初始化新圖                     寬度、高度
output = Image.new(image.mode, (1440, 450))

# 讀取pixel
pixel1 = image.load()
pixel2 = output.load()

# 進行轉換
transform(image,output,eight)

# 儲存新圖
output.save("output.jpg")