import cv2
import os
import numpy as np

IMAGE_PATH = r'D:\Study\wengzq\竞赛相关\orc'
save_path = r'D:\Study\wengzq\竞赛相关\orc\result'


def WShow(w_name, img):
    cv2.imshow(w_name, img)
    k = cv2.waitKey()
    if k == 7:
        cv2.destroyAllWindows()


def CleanDirt(c_list):
    clean = []
    c_list = np.asarray(c_list)

    for i in c_list:
        if i - np.mean(c_list) < 350:
            clean.append(i)
    return clean


class ImageDetect(object):
    def __init__(self, r_path):
        super(ImageDetect, self).__init__()
        self.r_path = r_path
        self.img_path = self.readImg()
        self.num = len(self.img_path)

    def readImg(self):
        img_list = []
        for file in os.listdir(self.r_path):
            if file.split('.')[-1] == 'png':
                img = cv2.imdecode(np.fromfile(os.path.join(self.r_path, file), dtype=np.uint8), -1)
                img_list.append(img)
        return img_list

    def tackleImg(self):
        """我也不知道为什么线条为啥画的这么清楚的数据处理
        output:线条异常清晰的图片，我傻了"""
        container = np.zeros((1000, 1000))
        container.fill(255)
        img_list = []
        for im in self.img_path:
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

            blur = cv2.GaussianBlur(gray, (35, 35), 0)
            img = cv2.add(gray[:, :], (gray[:, :] - blur[:, :]) * 2)  # 锐化图片
            ret, img = cv2.threshold(img, 252, 255, cv2.THRESH_TOZERO)

            binary = cv2.Canny(img, 50, 150, apertureSize=3)

            contours, h = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(container, contours, -1, (0, 0, 255), 30)
            c_img = container
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(c_img, contours, -1, (255, 0, 255), 3)
            c_img.reshape(gray.shape)

            show_i = img * c_img
            show_i = img - show_i
            show_i[show_i < 0] = 0

            gray1 = gray[:420, :]
            gray2 = gray[420:, :]

            copy_img_1 = show_i[:420, :]
            copy_img_2 = show_i[420:, :]
            target1, _ = self.getCorr(gray1, copy_img_1)
            target2, _ = self.getCorr(gray2, copy_img_2)

            img_list.append(target1)
            img_list.append(target2)
        return img_list

    def getCorr(self, origin, img):
        size = img.shape
        back = np.zeros((420, 1000), dtype=np.uint8)
        back.fill(255)
        col = []
        row = []
        # 对row方向进行扫描
        for i in range(size[0]):
            if np.mean(img[i, :], axis=0) > 3:
                row.append(i)
        # 对col方向扫描
        for i in range(size[1]):
            if np.mean(img[:, i], axis=0) > 4:
                col.append(i)

        row = CleanDirt(row)
        col = CleanDirt(col)

        back[row[0]:row[-1], col[0]:col[-1]] = 0
        mask = back
        target = origin[row[0]:row[-1], col[0] - 2:col[-1] + 2]
        return target, mask

    def getImg(self, s_path):
        img_list = self.tackleImg()
        try:
            for n, img in enumerate(img_list):
                name = str(n) + '.jpg'
                path = os.path.join(s_path, name)
                cv2.imencode('.jpg', img)[1].tofile(path)
        except IOError as e:
            print("An exception occurred")


if __name__ == '__main__':
    i = ImageDetect(IMAGE_PATH)
    i.getImg(save_path)
