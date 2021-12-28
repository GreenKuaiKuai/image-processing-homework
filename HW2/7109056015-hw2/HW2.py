from PIL import Image, ImageDraw, ImageFilter

# 取出某pixel附近八格的pixel值
def getpixel(x, y, img, num = 3):
    pixelList = []
    for i in range(-(int(num/2)),int(num/2)+1):
        for j in range(-(int(num/2)), int(num/2)+1):
            pixel = img.getpixel( (x+j, y+i) )
            pixelList.append(pixel)
    return pixelList

def laplacian_Mask(x, y, img):
    pixel = getpixel(x, y, img)
    
    result = 8*pixel[4] - (pixel[0] + pixel[1] + pixel[2] + pixel[3] + pixel[5] + pixel[6] + pixel[7] + pixel[8] )
    if result > 255: 
        result = 255 
    elif result < 0: 
        result = 0

    return int(result)

# 邊緣檢測
def sobel(x,y,img):
    pixel = getpixel(x, y, img)
    result  = abs(-(pixel[0]) + pixel[2] - 2 * pixel[3] + 2 * pixel[5] - pixel[6] + pixel[8]) + \
                abs(-(pixel[0]) - 2 * pixel[1] - pixel[2] + pixel[6] + 2 * pixel[7] + pixel[8])
    if result > 255: 
        result = 255 
    elif result < 0: 
        result = 0
    return int(result)

# 模糊
def blur(x,y,img):
    pixel = getpixel(x, y, img)
    result = (pixel[0] + pixel[1] + pixel[2] + pixel[3] + pixel[4] + pixel[5] + pixel[6] + pixel[7] + pixel[8]) / 9
    if result > 255: 
        result = 255 
    elif result < 0: 
        result = 0
    return int(result)

#轉灰階
def gray(img):
    width, height = img.size

    gray = Image.new( "L", (width,height) , (0))  
    draw = ImageDraw.Draw(gray)

    for x in range(0, width):       
        for y in range(0, height):
            grayscale = (img.getpixel((x,y))[0] + img.getpixel((x,y))[1] + img.getpixel((x,y))[2]) / 3
            draw.point((x,y), fill=int(grayscale))
    
    gray.save('gray_image.jpg')
    return gray


#######################################################################

# rgb to gray
print('rgb to gray...')
rbgImage = Image.open('my_image.jpg')
image = gray(rbgImage)

# image = Image.open('gray_image.jpg')
width, height = image.size 

## Laplacian Mask
print('Laplacian...')
laplacianImage = Image.new( "L", (width,height) , 0)  # mode, size, color
laplacianDraw = ImageDraw.Draw(laplacianImage)

for x in range(1, width - 1):       
    for y in range(1, height - 1):
        pixel = laplacian_Mask(x, y, image)
        laplacianDraw.point((x,y), fill=pixel)

laplacianImage.save( "laplacian_mask.jpg" )

# 做邊緣檢測
print('Sobel...')
sobelImage = Image.new( "L", (width,height) , (0))  
sobelDraw = ImageDraw.Draw(sobelImage)

for x in range(1, width - 1):       
    for y in range(1, height - 1):
        pixel = sobel(x, y, image)
        sobelDraw.point((x,y), fill=pixel)
sobelImage.save( "sobelImage.jpg" )

# 模糊
print('Blur...')
blurImage = Image.new( "L", (width,height) , (0))  
blurDraw = ImageDraw.Draw(blurImage)

for x in range(1, width - 1):       
    for y in range(1, height - 1):
        pixel = blur(x, y, sobelImage)
        blurDraw.point((x,y), fill=pixel)
blurImage.save( "blurImage.jpg" )

# 正規化後乘laplacian 再加回去原圖
print('Normalization...')
normalizationImage = Image.new( "L", (width,height) , (0))  
normalizationDraw = ImageDraw.Draw(normalizationImage)

for x in range(0, width):       
    for y in range(0, height):
        pixel = (blurImage.getpixel((x,y)) / 255) * laplacianImage.getpixel((x,y)) + image.getpixel((x,y)) 
        if pixel > 255: 
            pixel = 255 
        elif pixel < 0: 
            pixel = 0
        normalizationDraw.point((x,y), fill=int(pixel))
normalizationImage.save( "result.jpg" )
