import numpy as np
import cv2
import math
from scipy import ndimage
from PIL import Image

from skimage import filters
from skimage import color

# def getUCIQE(img_RGB):
#     # if img_RGB.dtype != np.uint8:
#     #     img_RGB = img_RGB.astype(np.uint8)
#     img_BGR = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2BGR)
#     img_LAB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2LAB)
#     img_LAB = np.array(img_LAB,dtype=np.float64)
#     # Trained coefficients are c1=0.4680, c2=0.2745, c3=0.2576 according to paper.
#     coe_Metric = [0.4680, 0.2745, 0.2576]

#     img_lum = img_LAB[:,:,0]/255.0
#     img_a = img_LAB[:,:,1]/255.0
#     img_b = img_LAB[:,:,2]/255.0

#     # item-1
#     chroma = np.sqrt(np.square(img_a)+np.square(img_b))
#     sigma_c = np.std(chroma)

#     # item-2
#     img_lum = img_lum.flatten()
#     sorted_index = np.argsort(img_lum)
#     top_index = sorted_index[int(len(img_lum)*0.99)]
#     bottom_index = sorted_index[int(len(img_lum)*0.01)]
#     con_lum = img_lum[top_index] - img_lum[bottom_index]

#     # item-3
#     chroma = chroma.flatten()
#     sat = np.divide(chroma, img_lum, out=np.zeros_like(chroma, dtype=np.float64), where=img_lum!=0)
#     avg_sat = np.mean(sat)

#     uciqe = sigma_c*coe_Metric[0] + con_lum*coe_Metric[1] + avg_sat*coe_Metric[2]
#     return uciqe

def getUCIQE(img_RGB):
    if img_RGB.dtype != np.uint8:
        img_RGB = img_RGB.astype(np.uint8)
    img_lab = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2LAB)  # Transform to Lab color space


    coe_metric = [0.4680, 0.2745, 0.2576]      # Obtained coefficients are: c1=0.4680, c2=0.2745, c3=0.2576.
    img_lum = img_lab[..., 0]/255
    img_a = img_lab[..., 1]/255
    img_b = img_lab[..., 2]/255

    img_chr = np.sqrt(np.square(img_a)+np.square(img_b))              # Chroma

    img_sat = img_chr/np.sqrt(np.square(img_chr)+np.square(img_lum))  # Saturation
    aver_sat = np.mean(img_sat)                                       # Average of saturation

    aver_chr = np.mean(img_chr)                                       # Average of Chroma

    var_chr = np.sqrt(np.mean(abs(1-np.square(aver_chr/img_chr))))    # Variance of Chroma

    dtype = img_lum.dtype                                             # Determine the type of img_lum
    if dtype == 'uint8':
        nbins = 256
    else:
        nbins = 65536

    hist, bins = np.histogram(img_lum, nbins)                        # Contrast of luminance
    cdf = np.cumsum(hist)/np.sum(hist)

    ilow = np.where(cdf > 0.0100)
    ihigh = np.where(cdf >= 0.9900)
    tol = [(ilow[0][0]-1)/(nbins-1), (ihigh[0][0]-1)/(nbins-1)]
    con_lum = tol[1]-tol[0]

    quality_val = coe_metric[0]*var_chr+coe_metric[1]*con_lum + coe_metric[2]*aver_sat         # get final quality value

    return quality_val


# def getUCIQE(img_RGB):
#     if img_RGB.dtype != np.uint8:
#         img_RGB = img_RGB.astype(np.uint8)
#     img_BGR = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2BGR)
#     img_LAB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2LAB)
#     img_LAB = np.array(img_LAB,dtype=np.float64)

#     coe_Metric = [0.4680, 0.2745, 0.2576]

#     img_lum = img_LAB[:,:,0]/255.0
#     img_a = img_LAB[:,:,1]/255.0
#     img_b = img_LAB[:,:,2]/255.0

#     chroma = np.sqrt(np.square(img_a)+np.square(img_b))
#     sigma_c = np.std(chroma)

#     img_lum = img_lum.flatten()
#     sorted_index = np.argsort(img_lum)
#     top_index = sorted_index[int(len(img_lum)*0.99)]
#     bottom_index = sorted_index[int(len(img_lum)*0.01)]
#     con_lum = img_lum[top_index] - img_lum[bottom_index]

#     chroma = chroma.flatten()
#     sat = np.divide(chroma, img_lum, out=np.zeros_like(chroma, dtype=np.float64), where=img_lum!=0)
#     avg_sat = np.mean(sat)

#     uciqe = sigma_c*coe_Metric[0] + con_lum*coe_Metric[1] + avg_sat*coe_Metric[2]
#     return uciqe


















#================================================================================

# def getUIQM(a):
#     rgb = a
#     lab = color.rgb2lab(a)
#     gray = color.rgb2gray(a)
#     # UCIQE
#     c1 = 0.4680
#     c2 = 0.2745
#     c3 = 0.2576
#     l = lab[:,:,0]

#     #1st term
#     chroma = (lab[:,:,1]**2 + lab[:,:,2]**2)**0.5
#     uc = np.mean(chroma)
#     sc = (np.mean((chroma - uc)**2))**0.5

#     #2nd term
#     top = np.int(np.round(0.01*l.shape[0]*l.shape[1]))
#     sl = np.sort(l,axis=None)
#     isl = sl[::-1]
#     conl = np.mean(isl[::top])-np.mean(sl[::top])

#     #3rd term
#     satur = []
#     chroma1 = chroma.flatten()
#     l1 = l.flatten()
#     for i in range(len(l1)):
#         if chroma1[i] == 0: satur.append(0)
#         elif l1[i] == 0: satur.append(0)
#         else: satur.append(chroma1[i] / l1[i])

#     us = np.mean(satur)

#     uciqe = c1 * sc + c2 * conl + c3 * us

#     # UIQM
#     p1 = 0.0282
#     p2 = 0.2953
#     p3 = 3.5753

#     #1st term UICM
#     rg = rgb[:,:,0] - rgb[:,:,1]
#     yb = (rgb[:,:,0] + rgb[:,:,1]) / 2 - rgb[:,:,2]
#     rgl = np.sort(rg,axis=None)
#     ybl = np.sort(yb,axis=None)
#     al1 = 0.1
#     al2 = 0.1
#     T1 = np.int(al1 * len(rgl))
#     T2 = np.int(al2 * len(rgl))
#     rgl_tr = rgl[T1:-T2]
#     ybl_tr = ybl[T1:-T2]

#     urg = np.mean(rgl_tr)
#     s2rg = np.mean((rgl_tr - urg) ** 2)
#     uyb = np.mean(ybl_tr)
#     s2yb = np.mean((ybl_tr- uyb) ** 2)

#     uicm =-0.0268 * np.sqrt(urg**2 + uyb**2) + 0.1586 * np.sqrt(s2rg + s2yb)

#     #2nd term UISM (k1k2=8x8)
#     Rsobel = rgb[:,:,0] * filters.sobel(rgb[:,:,0])
#     Gsobel = rgb[:,:,1] * filters.sobel(rgb[:,:,1])
#     Bsobel = rgb[:,:,2] * filters.sobel(rgb[:,:,2])

#     Rsobel=np.round(Rsobel).astype(np.uint8)
#     Gsobel=np.round(Gsobel).astype(np.uint8)
#     Bsobel=np.round(Bsobel).astype(np.uint8)

#     Reme = eme(Rsobel)
#     Geme = eme(Gsobel)
#     Beme = eme(Bsobel)

#     uism = 0.299 * Reme + 0.587 * Geme + 0.114 * Beme

#     #3rd term UIConM
#     uiconm = logamee(gray)

#     uiqm = p1 * uicm + p2 * uism + p3 * uiconm
#     return uiqm

# def eme(ch,blocksize=8):

#     num_x = math.ceil(ch.shape[0] / blocksize)
#     num_y = math.ceil(ch.shape[1] / blocksize)

#     eme = 0
#     w = 2. / (num_x * num_y)
#     for i in range(num_x):

#         xlb = i * blocksize
#         if i < num_x - 1:
#             xrb = (i+1) * blocksize
#         else:
#             xrb = ch.shape[0]

#         for j in range(num_y):

#             ylb = j * blocksize
#             if j < num_y - 1:
#                 yrb = (j+1) * blocksize
#             else:
#                 yrb = ch.shape[1]

#             block = ch[xlb:xrb,ylb:yrb]

#             blockmin = np.float(np.min(block))
#             blockmax = np.float(np.max(block))

#             # # old version
#             # if blockmin == 0.0: eme += 0
#             # elif blockmax == 0.0: eme += 0
#             # else: eme += w * math.log(blockmax / blockmin)

#             # new version
#             if blockmin == 0: blockmin+=1
#             if blockmax == 0: blockmax+=1
#             eme += w * math.log(blockmax / blockmin)
#     return eme

# def plipsum(i,j,gamma=1026):
#     return i + j - i * j / gamma

# def plipsub(i,j,k=1026):
#     return k * (i - j) / (k - j)

# def plipmult(c,j,gamma=1026):
#     return gamma - gamma * (1 - j / gamma)**c

# def logamee(ch,blocksize=8):

#     num_x = math.ceil(ch.shape[0] / blocksize)
#     num_y = math.ceil(ch.shape[1] / blocksize)

#     s = 0
#     w = 1. / (num_x * num_y)
#     for i in range(num_x):

#         xlb = i * blocksize
#         if i < num_x - 1:
#             xrb = (i+1) * blocksize
#         else:
#             xrb = ch.shape[0]

#         for j in range(num_y):

#             ylb = j * blocksize
#             if j < num_y - 1:
#                 yrb = (j+1) * blocksize
#             else:
#                 yrb = ch.shape[1]

#             block = ch[xlb:xrb,ylb:yrb]
#             blockmin = np.float(np.min(block))
#             blockmax = np.float(np.max(block))

#             top = plipsub(blockmax,blockmin)
#             bottom = plipsum(blockmax,blockmin)

#             m = top/ (bottom + 0.00001)
#             if m ==0.:
#                 s+=0
#             else:
#                 s += (m) * np.log(m)

#     return plipmult(w,s)

# -------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
from scipy import ndimage
from PIL import Image
import numpy as np
import math

def mu_a(x, alpha_L=0.1, alpha_R=0.1):
    """
      Calculates the asymetric alpha-trimmed mean
    """
    # sort pixels by intensity - for clipping
    x = sorted(x)
    # get number of pixels
    K = len(x)
    # calculate T alpha L and T alpha R
    T_a_L = math.ceil(alpha_L*K)
    T_a_R = math.floor(alpha_R*K)
    # calculate mu_alpha weight
    weight = (1/(K-T_a_L-T_a_R))
    # loop through flattened image starting at T_a_L+1 and ending at K-T_a_R
    s   = int(T_a_L+1)
    e   = int(K-T_a_R)
    val = sum(x[s:e])
    val = weight*val
    return val

def s_a(x, mu):
    val = 0
    for pixel in x:
        val += math.pow((pixel-mu), 2)
    return val/len(x)

def _uicm(x):
    R = x[:,:,0].flatten()
    G = x[:,:,1].flatten()
    B = x[:,:,2].flatten()
    RG = R-G
    YB = ((R+G)/2)-B
    mu_a_RG = mu_a(RG)
    mu_a_YB = mu_a(YB)
    s_a_RG = s_a(RG, mu_a_RG)
    s_a_YB = s_a(YB, mu_a_YB)
    l = math.sqrt( (math.pow(mu_a_RG,2)+math.pow(mu_a_YB,2)) )
    r = math.sqrt(s_a_RG+s_a_YB)
    return (-0.0268*l)+(0.1586*r)


def sobel(x):
    dx = ndimage.sobel(x,0)
    dy = ndimage.sobel(x,1)
    mag = np.hypot(dx, dy)
    mag *= 255.0 / np.max(mag)
    return mag

# def eme(x, window_size):
#     """
#       Enhancement measure estimation
#       x.shape[0] = height
#       x.shape[1] = width
#     """
#     # if 4 blocks, then 2x2...etc.
#     k1 = x.shape[1]//window_size
#     k2 = x.shape[0]//window_size
#     # weight
#     w = 2./(k1*k2)
#     blocksize_x = window_size
#     blocksize_y = window_size
#     # make sure image is divisible by window_size - doesn't matter if we cut out some pixels
#     x = x[:blocksize_y*k2, :blocksize_x*k1]
#     val = 0
#     for l in range(k1):
#         for k in range(k2):
#             block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1)]
#             max_ = np.max(block)
#             min_ = np.min(block)
#             # bound checks, can't do log(0)
#             if min_ == 0.0: val += 0
#             elif max_ == 0.0: val += 0
#             else: val += math.log(max_/min_)
#     return w*val

def eme(ch, blocksize=10):
    num_x = ch.shape[0] // blocksize
    num_y = ch.shape[1] // blocksize

    eme_value = 0
    w = 2.0 / (num_x * num_y)

    for i in range(num_x):
        xlb = i * blocksize
        xrb = (i + 1) * blocksize if i < num_x - 1 else ch.shape[0]

        for j in range(num_y):
            ylb = j * blocksize
            yrb = (j + 1) * blocksize if j < num_y - 1 else ch.shape[1]

            block = ch[xlb:xrb, ylb:yrb]
            blockmin = block.min()
            blockmax = block.max()

            if blockmin == 0: eme_value += 0
            elif blockmax == 0: eme_value += 0
            else:eme_value += w * np.log((blockmax)/(blockmin ))

    return eme_value

# def eme(x, window_size):
#     """
#     Enhancement measure estimation
#     x.shape[0] = height
#     x.shape[1] = width
#     """
#     # if 4 blocks, then 2x2...etc.
#     k1 = x.shape[1]/window_size
#     k2 = x.shape[0]/window_size
#     # weight
#     w = 2./(k1*k2)
#     blocksize_x = window_size
#     blocksize_y = window_size
#     # make sure image is divisible by window_size - doesn't matter if we cut out some pixels
#     x = x[0:int(blocksize_y*k2), 0:int(blocksize_x*k1)]
#     val = 0
#     k1 = int(k1)
#     k2 = int(k2)
#     for l in range(k1):
#         for k in range(k2):
#             block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1)]
#             max_ = np.max(block)
#             min_ = np.min(block)
#             # bound checks, can't do log(0)
#             if min_ == 0.0: val += 0
#             elif max_ == 0.0: val += 0
#             else: val += math.log(max_/min_)
#     return w*val

def _uism(x):
    """
      Underwater Image Sharpness Measure
    """
    # get image channels
    R = x[:,:,0]
    G = x[:,:,1]
    B = x[:,:,2]
    # first apply Sobel edge detector to each RGB component
    Rs = sobel(R)
    Gs = sobel(G)
    Bs = sobel(B)
    # multiply the edges detected for each channel by the channel itself
    R_edge_map = np.multiply(Rs, R)
    G_edge_map = np.multiply(Gs, G)
    B_edge_map = np.multiply(Bs, B)
    # get eme for each channel
    r_eme = eme(R_edge_map)
    g_eme = eme(G_edge_map)
    b_eme = eme(B_edge_map)
    # coefficients
    lambda_r = 0.299
    lambda_g = 0.587
    lambda_b = 0.144
    return (lambda_r*r_eme) + (lambda_g*g_eme) + (lambda_b*b_eme)

def plip_g(x,mu=1026.0):
    return mu-x

def plip_theta(g1, g2, k):
    g1 = plip_g(g1)
    g2 = plip_g(g2)
    return k*((g1-g2)/(k-g2))

def plip_cross(g1, g2, gamma):
    g1 = plip_g(g1)
    g2 = plip_g(g2)
    return g1+g2-((g1*g2)/(gamma))

def plip_diag(c, g, gamma):
    g = plip_g(g)
    return gamma - (gamma * math.pow((1 - (g/gamma) ), c) )

def plip_multiplication(g1, g2):
    return plip_phiInverse(plip_phi(g1) * plip_phi(g2))
    #return plip_phiInverse(plip_phi(plip_g(g1)) * plip_phi(plip_g(g2)))

def plip_phiInverse(g):
    plip_lambda = 1026.0
    plip_beta   = 1.0
    return plip_lambda * (1 - math.pow(math.exp(-g / plip_lambda), 1 / plip_beta));

def plip_phi(g):
    plip_lambda = 1026.0
    plip_beta   = 1.0
    return -plip_lambda * math.pow(math.log(1 - g / plip_lambda), plip_beta)

def _uiconm(x, window_size):
    """
      Underwater image contrast measure
      https://github.com/tkrahn108/UIQM/blob/master/src/uiconm.cpp
      https://ieeexplore.ieee.org/abstract/document/5609219
    """
    plip_lambda = 1026.0
    plip_gamma  = 1026.0
    plip_beta   = 1.0
    plip_mu     = 1026.0
    plip_k      = 1026.0
    # if 4 blocks, then 2x2...etc.
    k1 = x.shape[1]//window_size
    k2 = x.shape[0]//window_size
    # weight
    w = -1./(k1*k2)
    blocksize_x = window_size
    blocksize_y = window_size
    # make sure image is divisible by window_size - doesn't matter if we cut out some pixels
    x = x[:blocksize_y*k2, :blocksize_x*k1]
    # entropy scale - higher helps with randomness
    alpha = 1
    val = 0
    for l in range(k1):
        for k in range(k2):
            block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1), :]
            max_ = np.max(block)
            min_ = np.min(block)
            top = max_-min_
            bot = max_+min_
            if math.isnan(top) or math.isnan(bot) or bot == 0.0 or top == 0.0: val += 0.0
            else: val += alpha*math.pow((top/bot),alpha) * math.log(top/bot)
            #try: val += plip_multiplication((top/bot),math.log(top/bot))
    return w*val



def getUIQM(img_RGB):
    """
      Function to return UIQM to be called from other programs
      x: image
    """
    if img_RGB.dtype != np.uint8:
        img_RGB = img_RGB.astype(np.uint8)
    # img_RGB = (img_RGB - img_RGB.max()) / (img_RGB.max() - img_RGB.min()) * 255
    # img_RGB = Image.fromarray(img_RGB).resize((256, 256))
    x = np.array(img_RGB).astype(np.float32)
    c1 = 0.0282; c2 = 0.2953; c3 = 3.5753
    uicm   = _uicm(x)
    uism   = _uism(x)
    uiconm = _uiconm(x, 10)
    uiqm = (c1*uicm) + (c2*uism) + (c3*uiconm)
    return uiqm


#----------------------------------------------------------------------------------


#----------------------------------------------------------------------------------

# import numpy as np
# import cv2
# from scipy import ndimage
# import math
# from skimage import color, filters

# # 1. 计算 UICM (色度失真指标)
# def mu_a(x, alpha_L=0.1, alpha_R=0.1):
#     x = sorted(x)
#     K = len(x)
#     T_a_L = math.ceil(alpha_L * K)
#     T_a_R = math.floor(alpha_R * K)
#     weight = (1 / (K - T_a_L - T_a_R))
#     val = sum(x[T_a_L+1:K-T_a_R])
#     return weight * val

# def s_a(x, mu):
#     return np.mean((x - mu) ** 2)

# def calculate_uicm(image):
#     R = image[:, :, 0].flatten()
#     G = image[:, :, 1].flatten()
#     B = image[:, :, 2].flatten()

#     RG = R - G
#     YB = 0.5 * (R + G) - B

#     # 避免 NaN
#     RG = np.nan_to_num(RG, nan=0.0, posinf=0.0, neginf=0.0)
#     YB = np.nan_to_num(YB, nan=0.0, posinf=0.0, neginf=0.0)

#     mu_a_RG = mu_a(RG)
#     mu_a_YB = mu_a(YB)

#     s_a_RG = s_a(RG, mu_a_RG)
#     s_a_YB = s_a(YB, mu_a_YB)

#     l = np.sqrt(mu_a_RG ** 2 + mu_a_YB ** 2)
#     r = np.sqrt(s_a_RG + s_a_YB)

#     uicm = (-0.0268 * l) + (0.1586 * r)
#     if np.isnan(uicm):
#         uicm = 0  # 避免 NaN
#     return uicm

# # 2. 计算 UISM (锐度指标)
# def calculate_uism(image):
#     R = image[:, :, 0]
#     G = image[:, :, 1]
#     B = image[:, :, 2]

#     Rsobel = filters.sobel(R)
#     Gsobel = filters.sobel(G)
#     Bsobel = filters.sobel(B)

#     Reme = eme(R * Rsobel, 8)
#     Geme = eme(G * Gsobel, 8)
#     Beme = eme(B * Bsobel, 8)

#     uism = 0.299 * Reme + 0.587 * Geme + 0.114 * Beme

#     if np.isnan(uism):
#         uism = 0  # 避免 NaN
#     return uism

# # 3. 计算 UIConM (对比度指标) - 使用 PLIP 理论
# def plipsum(i, j, gamma=1026):
#     return i + j - i * j / gamma

# def plipsub(i, j, k=1026):
#     return k * (i - j) / (k - j)

# def plipmult(c, j, gamma=1026):
#     return gamma - gamma * (1 - j / gamma) ** c

# def calculate_uiconm(image, window_size=8):
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     num_x = math.ceil(gray.shape[0] / window_size)
#     num_y = math.ceil(gray.shape[1] / window_size)

#     s = 0
#     w = 1.0 / (num_x * num_y)
#     epsilon = 1e-5  # 小值避免除以零

#     for i in range(num_x):
#         xlb = i * window_size
#         xrb = (i + 1) * window_size if i < num_x - 1 else gray.shape[0]

#         for j in range(num_y):
#             ylb = j * window_size
#             yrb = (j + 1) * window_size if j < num_y - 1 else gray.shape[1]

#             block = gray[xlb:xrb, ylb:yrb]
#             blockmin = np.min(block)
#             blockmax = np.max(block)

#             if blockmin == 0: blockmin += epsilon
#             if blockmax == 0: blockmax += epsilon

#             top = plipsub(blockmax, blockmin)
#             bottom = plipsum(blockmax, blockmin)

#             m = top / (bottom + epsilon)

#             if m > 0:
#                 s += m * np.log(m)

#     uiconm = plipmult(w, s)
#     if np.isnan(uiconm):
#         uiconm = 0  # 避免 NaN
#     return uiconm

# # 4. EME 函数 - 用于局部对比度增强
# def eme(ch, blocksize=8):
#     num_x = math.ceil(ch.shape[0] / blocksize)
#     num_y = math.ceil(ch.shape[1] / blocksize)

#     eme_value = 0
#     w = 2.0 / (num_x * num_y)
#     epsilon = 1e-5  # 添加一个小的 epsilon 值避免除以零

#     for i in range(num_x):
#         xlb = i * blocksize
#         xrb = (i + 1) * blocksize if i < num_x - 1 else ch.shape[0]

#         for j in range(num_y):
#             ylb = j * blocksize
#             yrb = (j + 1) * blocksize if j < num_y - 1 else ch.shape[1]

#             block = ch[xlb:xrb, ylb:yrb]
#             blockmin = float(np.min(block))
#             blockmax = float(np.max(block))

#             if blockmin == 0: blockmin += epsilon
#             if blockmax == 0: blockmax += epsilon

#             # 避免 NaN 计算
#             if blockmax > blockmin:
#                 eme_value += w * np.log(blockmax / blockmin + epsilon)

#     return eme_value

# # 5. 计算最终的 UIQM
# def getUIQM(image):
#     uicm = calculate_uicm(image)
#     uism = calculate_uism(image)
#     uiconm = calculate_uiconm(image, 8)

#     # UIQM 权重
#     p1, p2, p3 = 0.0282, 0.2953, 3.5753
#     uiqm = (p1 * uicm) + (p2 * uism) + (p3 * uiconm)
#     return uiqm



#----------------------------------------------------------------------------------

from typing import Optional
import torch
import math
import numpy as np
from scipy import ndimage
import time


class UIQM():
    metric = 'UIQM'

    def __init__(self,
            gt_key: str = 'gt_img',
            pred_key: str = 'pred_img',
            collect_device: str = 'cpu',
            prefix: Optional[str] = None,
            crop_border=0,
            input_order='CHW',
            convert_to=None) -> None:
        self.name = "UIQM"
        super().__init__()

        self.crop_border = crop_border
        self.input_order = input_order
        self.convert_to = convert_to

    def sobel(self, x):
        dx = ndimage.sobel(x,0)
        dy = ndimage.sobel(x,1)
        mag = np.hypot(dx, dy)
        mag *= 255.0 / np.max(mag)
        return mag

    def get_values(self,blocks,top,bot,k1,k2,alpha):
        values = alpha * torch.pow((top / bot), alpha) * torch.log(top / bot)
        values[blocks[k1:k1+10, k2:k2+10, :] == 0] = 0.0
        return values

    def _uiconm(self, x, window_size):
        k1x = x.shape[1] / window_size
        k2x = x.shape[0] / window_size
        w = -1. / (k1x * k2x)
        blocksize_x = window_size
        blocksize_y = window_size
        blocks = x[0:int(blocksize_y * k2x), 0:int(blocksize_x * k1x)]
        k1, k2 = int(k1x), int(k2x)
        alpha = 1

        val = 0.0
        for l in range(k1):
            for k in range(k2):
                block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1), :]
                max_ = np.max(block)
                min_ = np.min(block)
                top = max_-min_
                bot = max_+min_
                if math.isnan(top) or math.isnan(bot) or bot == 0.0 or top == 0.0: val += 0.0
                else: val += alpha*math.pow((top/bot),alpha) * math.log(top/bot)
                #try: val += plip_multiplication((top/bot),math.log(top/bot))
        return w*val

    def mu_a(self, x, alpha_l = 0.1, alpha_r = 0.1):
        """
        Calculates the asymetric alpha-trimmed mean
        """
        # sort pixels by intensity - for clipping
        x = sorted(x)
        # get number of pixels
        K = len(x)
        # calculate T alpha L and T alpha R
        T_a_L = math.ceil(alpha_l*K)
        T_a_R = math.floor(alpha_r*K)
        # calculate mu_alpha weight
        weight = (1/(K-T_a_L-T_a_R))
        # loop through flattened image starting at T_a_L+1 and ending at K-T_a_R
        s   = int(T_a_L+1)
        e   = int(K-T_a_R)
        val = sum(x[s:e])
        val = weight*val
        return val


    def s_a(self, x, mu):
        val = 0
        for pixel in x:
            val += math.pow((pixel-mu), 2)
        return val/len(x)

    def _uicm(self, x):
        r = x[:,:,0].flatten()
        g = x[:,:,1].flatten()
        b = x[:,:,2].flatten()
        rg = r - g
        yb = ((r + g) / 2) - b
        mu_a_rg = self.mu_a(rg)
        mu_a_yb = self.mu_a(yb)
        s_a_rg = self.s_a(rg, mu_a_rg)
        s_a_yb = self.s_a(yb, mu_a_yb)
        l = math.sqrt((mu_a_rg ** 2) + (mu_a_yb ** 2))
        r = math.sqrt(s_a_rg + s_a_yb)
        return (-0.0268 * l) + (0.1586 * r)

    def _eme1(self, x, window_size):
        """
        Enhancement measure estimation
        x.shape[0] = height
        x.shape[1] = width
        """
        # if 4 blocks, then 2x2...etc.
        k1 = x.shape[1]/window_size
        k2 = x.shape[0]/window_size
        # weight
        w = 2./(k1*k2)
        blocksize_x = window_size
        blocksize_y = window_size
        # make sure image is divisible by window_size - doesn't matter if we cut out some pixels
        x = x[0:int(blocksize_y*k2), 0:int(blocksize_x*k1)]
        val = 0
        k1 = int(k1)
        k2 = int(k2)
        for l in range(k1):
            for k in range(k2):
                block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1)]
                max_ = np.max(block)
                min_ = np.min(block)
                # bound checks, can't do log(0)
                if min_ == 0.0: val += 0
                elif max_ == 0.0: val += 0
                else: val += math.log(max_/min_)
        return w*val

    def _uism(self, x):
        r = x[:,:,0]
        g = x[:,:,1]
        b = x[:,:,2]
        rs = self.sobel(r)
        gs = self.sobel(g)
        bs = self.sobel(b)
        r_edge_map = np.multiply(rs, r)
        g_edge_map = np.multiply(gs, g)
        b_edge_map = np.multiply(bs, b)
        r_eme = self._eme1(r_edge_map, 10)
        g_eme = self._eme1(g_edge_map, 10)
        b_eme = self._eme1(b_edge_map, 10)
        lambda_r = 0.299
        lambda_g = 0.587
        lambda_b = 0.144
        return (lambda_r * r_eme) + (lambda_g * g_eme) + (lambda_b * b_eme)

    def calculate(self, x):

        x = x.astype(np.float32)
        c1 = 0.0282
        c2 = 0.2953
        c3 = 3.5753
        uicm = self._uicm(x)
        uism = self._uism(x)
        uiconm = self._uiconm(x, 10)
        uiqm = (c1 * uicm) + (c2 * uism) + (c3 * uiconm)

        return uiqm

    def process_image(self, pred):
        return self.calculate(pred)