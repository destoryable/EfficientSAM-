from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import zipfile
models = {}
models['efficientsam_ti'] = build_efficient_sam_vitt()
with zipfile.ZipFile("weights/efficient_sam_vits.pt.zip", 'r') as zip_ref:
    zip_ref.extractall("weights")
sample_image_np = np.array(Image.open("figs/examples/dog1.jpg"))
sample_image_tensor = transforms.ToTensor()(sample_image_np)
 
def inference():
    for model_name, model in models.items():
        print('Running inference using ', model_name)
        predicted_logits, predicted_iou = model(
            sample_image_tensor[None, ...],
            input_points.to(torch.float32),
            input_labels.to(torch.float32),
        )
        sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
        predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
        predicted_logits = torch.take_along_dim(
            predicted_logits, sorted_ids[..., None, None], dim=2
        )
        # The masks are already sorted by their predicted IOUs.
        # The first dimension is the batch size (we have a single image. so it is 1).
        # The second dimension is the number of masks we want to generate (in this case, it is only 1)
        # The third dimension is the number of candidate masks output by the model.
        # For this demo we use the first mask.
        mask = torch.ge(predicted_logits[0, 0, 0, :, :], 0).cpu().detach().numpy()
        masked_image_np = sample_image_np.copy().astype(np.uint8) * mask[:,:,None]
        Image.fromarray(masked_image_np).save(f"figs/examples/dog1_{model_name}_mask_ui.png")
 
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        a.append(x)
        b.append(y)
        cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)
 
count = 0
 
for i in range(1):
    import cv2
    import numpy as np
    # 图片路径
    model_name = "efficientsam_ti"
    a = []
    b = []
    if count == 0:
        img = cv2.imread(f"figs/examples/dog1.jpg")
        cv2.imshow("image", img)
        cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
        # cv2.imshow("image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        positions = [[[[a[0], b[0]], [a[1], b[1]]]]]
        input_points = torch.tensor(positions)
        sample_image_np = np.array(img)
        sample_image_tensor = transforms.ToTensor()(sample_image_np)
        input_labels = torch.tensor([[[1, 1]]])
        inference()
        count += 1
        continue
    a = []
    b = []
    img = cv2.imread(f"figs/examples/dog1_{model_name}_mask.png")
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    positions = [[[[a[0],b[0]],[a[1],b[1]]]]]
    input_points = torch.tensor(positions)
    sample_image_np = np.array(img)
    sample_image_tensor = transforms.ToTensor()(sample_image_np)
    input_labels = torch.tensor([[[1, 1]]])
    print("---------------------------------------------")
    for i in range(len(a)):
        print(a[i], b[i])
    inference()