# ディレクトリ変更
import os
print(os.getcwd())
os.chdir('C:/Users/shas/Documents/PythonTrial/201129')

# ----コピペ開始----------------------------------------------------------------------

# ライブラリの読み込み
import glob
import logging
import datetime
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sns
import cv2
from PIL import Image
from PIL import ImageEnhance

# ----画像の差分を取得----
# 全ての画像ファイルのパスを取得
pathList = glob.glob("./*.jpg")
imgFileNum = len(pathList)
# for i in range(imgFileNum):
#     pathList[i] = os.path.abspath(pathList[i])

# 背景画像の指定
pathBack = pathList[0]
pathFront = pathList[1:]

# 出力のファイルパスを設定（手動）
os.mkdir('.\\output')
pathOut = []
for i in range(imgFileNum - 1):
    pathOut.append(".\\output" + pathFront[i][1:-4] + "_out" + pathFront[i][-4:])

# 画像の読み込みと重ね合わせ
imgBack = cv2.cvtColor(cv2.imread(pathBack), cv2.COLOR_BGR2RGB)

for i in range(imgFileNum - 1):
    imgFront = cv2.cvtColor(cv2.imread(pathFront[i]), cv2.COLOR_BGR2RGB)
    imgDif = imgFront.astype(int) - imgBack.astype(int)
    # 型変換してunit8にしておく
    imgDifAbs = np.abs(imgDif).astype("u1")
    cv2.imwrite(pathOut[i], imgDifAbs)


# ----コピペ終了----------------------------------------------------------------------

# 文字の領域をチェックする
pHidari = 24
pUe = 42

# ----コピペ開始----------------------------------------------------------------------

#ログの設定
logger = logging.getLogger('LoggingTest')
logger.setLevel(10)
sh = logging.StreamHandler()
logger.addHandler(sh)
timeNow = datetime.datetime.now()
logFileName = str(timeNow.year) + "_" + str(timeNow.month) + str(timeNow.day) + "_" + str(timeNow.hour) + str(timeNow.minute) + ".log"
fh = logging.FileHandler(logFileName)
logger.addHandler(fh)

# 全ての画像ファイルのパスを取得
pathOut = glob.glob(".\\output\\*.jpg")
os.mkdir('.\\merged')

mojiLoc = (pHidari, pUe, pHidari+20, pUe+20)
arrStart = []
for i in range(len(pathOut)):
    imgTarget = Image.open(pathOut[i])
    roiColor = imgTarget.crop(mojiLoc).getdata()
    pixelSum = []
    for j in range(len(roiColor)):
        pixelSum.extend(list(roiColor[j]))
    pixelSum = [pixelSum[0::3], pixelSum[1::3], pixelSum[2::3]]
    pixelSum2 = [0, 0, 0]
    for k in range(3):
        pixelSum2[k] = sum([pixelSum[k][x] for x in range(len(roiColor)) if pixelSum[k][x] > 50])
    if pixelSum2[0] + pixelSum2[1] > max(4000, 2*pixelSum2[2]):
        arrStart.append(str(i))

arrStart
len(arrStart)

# ----コピペ終了----------------------------------------------------------------------

# arrStartに不具合がないかチェック
# 差分をチェックする位置の設定
margin = 5
margin2 = 25

# コントラストの引き上げ度合い
coef = 7

# ----コピペ開始----------------------------------------------------------------------

# ----画像のつなぎあわせ----
# 画像を読み込む関数
def imgReading(a):
    img = cv2.imread(a, cv2.IMREAD_GRAYSCALE)
    imgRGB = cv2.imread(a)
    return img, imgRGB

# 画像の上下左右の境界に葉っぱがあるかを判定する関数
class leafExist:
    # pos：判定に使う縦横の直線の座標
    # arr：直線状にある画素の明るさの和
    # flag：判定結果
    def __init__(self):
        self.pos = [0,0,0,0]
        self.arr = [0,0,0,0]
        self.flag = [0,0,0,0]
    def confirm(self, connect, conShape, margin):
        self.pos = [margin, conShape[0]-margin, margin, conShape[1]-margin]
        checkArr = [connect[self.pos[0],:], connect[self.pos[1],:], connect[:,self.pos[2]], connect[:,self.pos[3]]]
        for i in range(4):
            self.arr[i] = np.sum(np.square(checkArr[i]))
        arrAve = sum(self.arr) / len(self.arr)
        if len([x for x in self.arr if x > 5000]) == 0:
            self.flag = [0.1, 0.1, 0.1, 0.1]
        for i in range(4):
            if self.arr[i] > 25000:
                self.flag[i] = 1
            elif self.arr[i] > int(1.2 * arrAve):
                self.flag[i] = 1
            elif self.arr[i] > int(0.9 * arrAve):
                self.flag[i] = 0.1

#差分を計算する関数
def sumTateCalc(connect, conShape, basis, basShape, pos, j, margin3, flag):
    flagDrv = flag*(basShape[1]-conShape[1])
    smCoef = np.ones(3)/3
    conRange = connect[pos + margin3,:]
    basRange = basis[j + margin3,flagDrv:conShape[1] + flagDrv]
    basSmooth = np.convolve(basRange, smCoef, mode = 'same')
    basSmooth = basSmooth.astype(np.uint8)
    difsq = np.square(conRange - basSmooth)
    calItems = difsq / (conRange + 1)
    calSum = np.sum(calItems)
    calSum = calSum / 4
    return calSum

def sumYokoCalc(connect, conShape, basis, basShape, pos, j, margin3, flag):
    flagDrv = flag*(basShape[0]-conShape[0])
    smCoef = np.ones(3)/3
    conRange = connect[:,pos + margin3]
    basRange = basis[flagDrv:conShape[0] + flagDrv,j + margin3]
    basSmooth = np.convolve(basRange, smCoef, mode = 'same')
    basSmooth = basSmooth.astype(np.uint8)
    difsq = np.square(conRange - basSmooth)
    calItems = difsq / (conRange + 1)
    calSum = np.sum(calItems)
    calSum = calSum / 3
    return calSum

#差分を定量化する関数
def sumUe(connect, conShape, basis, basShape, pos, margin2):
    # minSumは（辺の方向）、（basis側でつながるxy座標）、（差分）を格納する
    minSum = [[0, 0, 0, 9999], [0, 0, 0, 9999]]
    for j in range(margin2, basShape[0] - margin2):
        calSum = sumTateCalc(connect, conShape, basis, basShape, pos, j, 0, 0)
        calSum = calSum + sumTateCalc(connect, conShape, basis, basShape, pos, j, margin2, 0)
        if calSum < minSum[0][3]:
            minSum[0][1] = j - pos
            minSum[0][3] = round(calSum, 2)
    if basShape[1] > conShape[1]:
        for j in range(margin2, basShape[0] - margin2):
            calSum = sumTateCalc(connect, conShape, basis, basShape, pos, j, 0, 1)
            calSum = calSum + sumTateCalc(connect, conShape, basis, basShape, pos, j, margin2, 1)
            if calSum < minSum[1][3]:
                minSum[1][1] = j - pos
                minSum[1][2] = basShape[1] - conShape[1]
                minSum[1][3] = round(calSum, 2)
    return minSum

def sumShita(connect, conShape, basis, basShape, pos, margin2):
    # minSumは（辺の方向）、（basis側でつながるxy座標）、（差分）を格納する
    minSum = [[1, 0, 0, 9999]]
    for j in range(margin2, basShape[0] - margin2):
        calSum = sumTateCalc(connect, conShape, basis, basShape, pos, j, 0, 1)
        calSum = calSum + sumTateCalc(connect, conShape, basis, basShape, pos, j, 0-margin2, 1)
        if calSum < minSum[0][3]:
            minSum[0][1] = j - pos
            minSum[0][2] = basShape[1] - conShape[1]
            minSum[0][3] = round(calSum, 2)
    return minSum

def sumHidari(connect, conShape, basis, basShape, pos, margin2):
    # minSumは（辺の方向）、（basis側でつながるxy座標）、（差分）を格納する
    minSum = [[2, 0, 0, 9999], [2, 0, 0, 9999]]
    for j in range(margin2, basShape[1] - margin2):
        calSum = sumYokoCalc(connect, conShape, basis, basShape, pos, j, 0, 0)
        calSum = calSum + sumYokoCalc(connect, conShape, basis, basShape, pos, j, margin2, 0)
        if calSum < minSum[0][3]:
            minSum[0][2] = j - pos
            minSum[0][3] = round(calSum, 2)
    if basShape[0] > conShape[0]:
        for j in range(margin2, basShape[1] - margin2):
            calSum = sumYokoCalc(connect, conShape, basis, basShape, pos, j, 0, 1)
            calSum = calSum + sumYokoCalc(connect, conShape, basis, basShape, pos, j, margin2, 1)
            if calSum < minSum[1][3]:
                minSum[1][2] = j - pos
                minSum[1][1] = basShape[0] - conShape[0]
                minSum[1][3] = round(calSum, 2)
    return minSum

# 葉っぱがある境界について、元になる画像との一致部分を当てにいく関数
def locMatching(leafExist, connect, conShape, basis, basShape, margin2):
    connectPos = []
    for i in range(3):
        if leafExist.flag[i] != 0:
            if i == 0:
                sumVal = sumUe(connect, conShape, basis, basShape, leafExist.pos[i], margin2)
            elif i == 1:
                sumVal = sumShita(connect, conShape, basis, basShape, leafExist.pos[i], margin2)
            elif i == 2:
                sumVal = sumHidari(connect, conShape, basis, basShape, leafExist.pos[i], margin2)
            connectPos.extend(sumVal)
    return connectPos

# 候補を絞り込む関数
def leafFlagSet(conPosArr, conShape, basShape, leafFlagPre, leafFlag):
    conEdge = [conPosArr[1], 0-(conShape[0] + conPosArr[1]), conPosArr[2], 0-(conShape[1] + conPosArr[2])]
    basEdge = [0, basShape[0], 0, basShape[1]]
    leafFlagTrans = leafFlagPre
    leafFlagNew = [0,0,0,0]
    for j in range(4):
        if conEdge[j] + basEdge[j] < 0:
            if leafFlagPre[j] == 0:
                conPosArr[3] = 9999
            else:
                leafFlagNew[j] = leafFlag[j]
                if j in [0,1]:
                    for k in [2,3]:
                        leafFlagNew[k] = max(leafFlagNew[k], leafFlag[k])
                elif j in [2,3]:
                    for k in [0,1]:
                        leafFlagNew[k] = max(leafFlagNew[k], leafFlag[k])
        else:
            leafFlagNew[j] = max(leafFlagPre[j], leafFlagNew[j])
    return conPosArr, leafFlagNew

def locSelecting(connectPos, conShape, basShape, leagFlagPre, leafFlag): 
    criteria = [99999, 9]
    leafFlagTrans = [0,0,0,0]
    leafFlagNew = [0,0,0,0]
    for i in range(len(connectPos)):
        if connectPos[i][3] == 9999:
            continue
        elif connectPos[i][2] < 0:
            continue
        elif connectPos[i][1] <= 0 and connectPos[i][2] == 0:
            continue
        else:
            connectPos[i], leafFlagTrans = leafFlagSet(connectPos[i], conShape, basShape, leagFlagPre, leafFlag)
            if connectPos[i][3] < criteria[0]:
                criteria[0] = connectPos[i][3]
                criteria[1] = i
                leafFlagNew = leafFlagTrans
    if criteria[1] == 9:
        connectPosFin = [0, 0, 0, 0]
    else:
        connectPosFin = connectPos[criteria[1]]
    return connectPosFin, leafFlagNew

# 画像を連結する関数
def imgMerging(cPF, connectRGB, conShape, basisRGB, basShape):
    #if cPF[3] == 0:
    #    return
    # 結合後の画像サイズに相当するキャンバスを用意する
    if cPF[1] < 0:
        height = basShape[0] - cPF[1]
    else:
        height = max(basShape[0], cPF[1] + conShape[0])
    if cPF[2] < 0:
        width = basShape[1] - cPF[2]
    else:
        width = max(basShape[1], cPF[2] + conShape[1])
    imgBlk = np.zeros((height, width, 3), dtype = np.uint8)
    # 画像の結合
    if cPF[1] < 0:
        imgBlk[0 - cPF[1]:,:] = basisRGB
        imgBlk[:conShape[0], cPF[2]:cPF[2] + conShape[1]] = connectRGB
    elif cPF[1] == 0 and cPF[2] < 0:
        imgBlk[:,0 - cPF[2]:] = basisRGB
        imgBlk[:conShape[0], :conShape[1]] = connectRGB
    elif cPF[1] >= 0 and cPF[2] >= 0:
        imgBlk[:basShape[0],:basShape[1]] = basisRGB
        imgBlk[cPF[1]:cPF[1] + conShape[0],cPF[2]:cPF[2] + conShape[1]] = connectRGB
    return imgBlk

# とりまとめの関数
def autoMerging(pathOut, fileList, margin, margin2):
    for k in range(len(fileList)-1):
        if k == 0:
            input = pathOut[fileList[k]]
            imgBasis, imgBasisRGB = imgReading(input)
            imgConnect, imgConnectRGB = imgReading(pathOut[fileList[k+1]])
            leafPrpPre = leafExist()
            leafPrpPre.confirm(imgBasis, imgBasis.shape, margin)
        else:
            imgBasisRGB = imgFin
            imgBasis = cv2.cvtColor(imgBasisRGB, cv2.COLOR_BGR2GRAY)
            imgConnect, imgConnectRGB = imgReading(pathOut[fileList[k+1]])
            leafPrpPre.flag = leafFlagNew
        leafPrp = leafExist()
        leafPrp.confirm(imgConnect, imgConnect.shape, margin)
        logger.info(leafPrp.arr)
        logger.info(leafPrp.flag)
        connectPos = locMatching(leafPrp, imgConnect, imgConnect.shape, imgBasis, imgBasis.shape, margin2)
        # 写真が２枚中の２枚目でなければ左の結合は行わない
        if k == 0 and len(fileList) != 2:
            connectPos = [connectPos[x] for x in range(len(connectPos)) if connectPos[x][0] == 0]
        logger.info(connectPos)
        connectPosFin, leafFlagNew = locSelecting(connectPos, imgConnect.shape, imgBasis.shape, leafPrpPre.flag, leafPrp.flag)
        logger.info(leafFlagNew)
        imgFin = imgMerging(connectPosFin, imgConnectRGB, imgConnect.shape, imgBasisRGB, imgBasis.shape)
        pathLoc = ".\\merged" + pathOut[fileList[k+1]][8:-8] + "_merged" + pathOut[fileList[k+1]][-4:]
        if k == len(fileList)-2:
            cv2.imwrite(pathLoc, imgFin)
            logger.info("Export: " + pathLoc)

for i in range(0, len(arrStart)):
    logger.info("arrStart:" + str(i))
    if i == len(arrStart) - 1:
        next = len(pathOut) - 1
    else:
        next = arrStart[i+1]
    if int(next) - int(arrStart[i]) == 1:
        img = cv2.imread(pathOut[int(arrStart[i])])
        pathLoc = ".\\merged" + pathOut[int(arrStart[i])][8:]
        cv2.imwrite(pathLoc, img)
    else:
        fileList = list(range(int(arrStart[i]), int(next)))
        autoMerging(pathOut, fileList, margin, margin2)

#ログの終了
logger.removeHandler(sh)
logger.removeHandler(fh)
logging.shutdown()

# 全ての画像ファイルのパスを取得
pathList = glob.glob("./merged/*.jpg")
imgFileNum = len(pathList)

# 出力のファイルパスを設定（手動）
os.mkdir('.\\highcontrast')
pathOut = []
for i in range(imgFileNum - 1):
    pathOut.append(".\\highcontrast" + pathList[i][8:])

for i in range(imgFileNum-1):
    img = Image.open(pathList[i])
    img = ImageEnhance.Contrast(img)
    newImage = img.enhance(coef)
    newImage.save(pathOut[i])
    if i % 20 == 0:
        print(i)

# ----コピペ終了----------------------------------------------------------------------

# 画像がちゃんとつながってるか手動チェック

# ----コピペ開始----------------------------------------------------------------------

# ----画像の差分を取得----
# コントラスト上げた画像ファイル（削除後）のパスを取得
os.chdir("./merged")
pathPrev = glob.glob("./*.jpg")
os.chdir("../highcontrast")
pathAfter = glob.glob("./*.jpg")
os.chdir("..")
imgFileNumPrev = len(pathPrev)
imgFileNumAfter = len(pathAfter)

# 出力のファイルパスを設定（手動）
os.mkdir('.\\merged_deleted')

for i in range(imgFileNumAfter - 1):
    for j in range(imgFileNumPrev - 1):
        if pathAfter[i] == pathPrev[j]:
            fileLoc = ".\\merged" + pathAfter[i]
            img = Image.open(fileLoc)
            img.save(".\\merged_deleted" + pathAfter[i])
    if i % 20 == 0:
        print(i)

