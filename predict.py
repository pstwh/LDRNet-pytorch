import torch
import cv2
import numpy as np
import time

from model import LDRNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LDRNet()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

vc = cv2.VideoCapture(0)

pf = 0
nf = 0

while vc.isOpened():
    with torch.no_grad():
        nf = time.time()
        ret, frame = vc.read()
        img = cv2.resize(frame, (224, 224))
        img_show = np.copy(img)

        img = torch.tensor(img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        img = (img - 0.5 * 255) / (0.5 * 255)

        result = model(img)
        lines = result[1].detach().numpy()[0]
        result = result[0].detach().numpy()[0]
        coord = result[0:8]
        coord = [int(x * 224) for x in coord]
        lines = [int(x * 224) for x in lines]
        cv2.circle(img_show, (coord[0], coord[1]), 3, (0, 0, 255), -1)
        cv2.circle(img_show, (coord[2], coord[3]), 3, (0, 255, 255), -1)
        cv2.circle(img_show, (coord[4], coord[5]), 3, (255, 0, 0), -1)
        cv2.circle(img_show, (coord[6], coord[7]), 3, (0, 255, 0), -1)
        for i in range(0, 192, 2):
            cv2.circle(img_show, (lines[i], lines[i + 1]), 1, (127, 127, 127), -1)

        fps = 1 / (nf - pf)
        pf = nf
        fps = int(fps)
        fps = str(fps)

        cv2.putText(
            img_show,
            fps,
            (7, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (100, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow("Video", img_show)

        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break

vc.release()
cv2.destroyAllWindows()
