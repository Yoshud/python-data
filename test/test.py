import numpy as np
import cv2
# import skimage.measure
def maxpol(mat):
    M, N = mat.shape
    K = 32
    L =32

    MK = M // K
    NL = N // L
    returned = mat[:MK*K, :NL*L].reshape(MK, K, NL, L).mean(axis=(1, 3), dtype=np.int32)
    return np.uint8(np.clip(returned, 0, 255))

def maxpoolAll(frame):
    B, G, R = cv2.split(frame)
    maxPooledB = maxpol(B)
    maxPooledG = maxpol(G)
    maxPooledR = maxpol(R)
    merged = cv2.merge((maxPooledB, maxPooledG, maxPooledR))
    return cv2.resize(merged, (B.shape[1], B.shape[0]))
    # return merged

cap = cv2.VideoCapture(0)
ret, last_frame = cap.read()
_, last_last_frame = cap.read()
frames = [last_frame, ]

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    maxpooled = maxpoolAll(frame)
    frames.append(frame)
    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    diff = maxpoolAll(last_frame) - maxpoolAll(frame)
    meanVal = cv2.mean(diff)
    shape = frame.shape

    print(cv2.norm(diff))
    # cv2.imshow('frame', diff-meanVal[:3])
    # cv2.imshow('frame', 2*frame - meanVal[:3])
    # cv2.imshow('frame', (frame) // 2)

    drowing = (np.int32(maxpoolAll(last_last_frame)) + np.int32(maxpoolAll(last_frame))) // 2

    cv2.imshow('frame', np.uint8(np.clip(drowing, 0, 255)) - frame)
    # cv2.imshow('frame', diff)

    # cv2.imshow('frame', drowing)
    print(drowing[:3])
    # print((frame + frame)[:3])

    last_frame = frame
    last_last_frame = last_frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()