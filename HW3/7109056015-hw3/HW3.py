from PIL import Image
    
# 讀原圖 
image = Image.open('my_image.png').convert('1')
image_pixel = image.load()
weight, height = image.size

# 生成補集用的圖
complement_img = Image.new(image.mode, image.size)
complement_pixel = complement_img.load()

# 做出原圖的補集
for i in range(0, weight):
    for j in range(0, height):
        complement_pixel[i, j] = 255 - image_pixel[i, j]
complement_img.save("complement.jpg")

# 創建全白的圖作為初始圖
temp_img = Image.new(image.mode, image.size, 255) 
temp_pixel = temp_img.load()
temp_pixel[330,255] = 0 #將初始點[330,255] 設成黑色
temp_img.save("temp.jpg")

check = True
while(check):
    # 讀取temp圖片
    temp_img = Image.open('temp.jpg').convert('1')
    temp_pixel = temp_img.load()

    # 生成此次迴圈用的圖片
    nextTemp = Image.new(temp_img.mode, temp_img.size, 255) #全白
    nextTemp_pixel = nextTemp.load()

    # dilation
    for i in range(0, weight):
        for j in range(0, height):
            if(temp_pixel[i, j]<=10):
                nextTemp_pixel[i, j-1] = 0
                nextTemp_pixel[i-1, j] = 0
                nextTemp_pixel[i, j] = 0
                nextTemp_pixel[i+1, j] = 0
                nextTemp_pixel[i, j+1] = 0

    # 跟補集圖作交集
    for i in range(0, weight):
        for j in range(0, height):
            if(abs(nextTemp_pixel[i, j] - complement_pixel[i, j]) > 10):
                nextTemp_pixel[i, j] = 255      

    nextTemp.save("temp.jpg")
    
    # 檢查是否和上一次的結果一樣
    check = False
    for i in range(0, weight):
        for j in range(0, height):
            if(nextTemp_pixel[i, j] != temp_pixel[i, j]):
                check = True
                break
# 迴圈結束儲存最後圖                
nextTemp.save("loop_end.jpg")

# 將最後圖與原圖做聯集
result_img = Image.new(image.mode, image.size)
result_pixel = result_img.load()
for i in range(0, weight):
    for j in range(0, height):
        if((image_pixel[i, j] == 0 or nextTemp_pixel[i, j] == 0)):
            result_pixel[i, j] = 0
        else:
            result_pixel[i, j] = 255

result_img.save("result.jpg")