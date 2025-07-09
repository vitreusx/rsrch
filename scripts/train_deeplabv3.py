import train_unet
from torchvision.models import ResNet34_Weights
from train_unet import *

from rsrch.models import deeplabv3
from rsrch.models.resnet import adapt_tv_state_dict, resnet34


class SegModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        upsample: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.upsample = upsample

    def forward(self, input: Tensor):
        output = self.head(self.backbone(input))
        if self.upsample:
            output = tv_F.resize(output, input.shape[-2:])
        return output


class Trainer(train_unet.Trainer):
    project = "deeplabv3"

    def setup_data(self):
        voc_root = "./datasets/voc"
        MEAN = np.array([0.485, 0.456, 0.406])
        STD = np.array([0.229, 0.224, 0.225])

        class Data:
            def __init__(
                self,
                split: Literal["train", "val"],
                transforms: list,
            ):
                super().__init__()
                self.base = VOCSegmentation(voc_root, split=split)

                self.meta = self.base.meta()

                self.transform = A.Compose(
                    [
                        *transforms,
                        A.Normalize(MEAN, STD),
                        A.ToTensorV2(),
                    ]
                )

            def __len__(self):
                return len(self.base)

            def __getitem__(self, index: int):
                item = self.base[index]
                image = np.asarray(item["image"].convert("RGB"))
                labels = np.asarray(item["labels"])
                res = self.transform(image=image, mask=labels)
                return {
                    "image": res["image"],
                    "labels": res["mask"].to(torch.long),
                }

            def to_pil_image(self, image: Tensor):
                img_nd = image.moveaxis(0, -1).numpy(force=True)
                img_nd = img_nd * STD + MEAN
                img_nd = (255 * img_nd).astype(np.uint8)
                return Image.fromarray(img_nd)

        img_size = 256

        self.train_data = Data(
            split="train",
            transforms=[
                A.RandomResizedCrop(
                    (img_size, img_size),
                    scale=(0.08, 1.0),
                    ratio=(3 / 4, 4 / 3),
                ),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(p=0.5),
            ],
        )

        self.val_data = Data(
            split="val",
            transforms=[
                A.SmallestMaxSize(img_size),
                A.CenterCrop(img_size, img_size),
            ],
        )

        self.meta = self.train_data.meta

    def setup_model(self):
        backbone = resnet34(num_classes=None)
        tv_state = ResNet34_Weights.IMAGENET1K_V1.get_state_dict()
        backbone.load_state_dict(adapt_tv_state_dict(tv_state), strict=False)

        deeplabv3.convert_resnet_(backbone, [2, 4])
        self.model = SegModel(
            backbone=backbone,
            head=deeplabv3.ASPP(
                in_channels=backbone.num_channels[-1],
                out_channels=self.meta.num_classes,
                dilations=[6, 12, 18],
            ),
            upsample=True,
        )

        self.model = self.ddp.wrap_model(self.model)

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=3e-4)
        self.opt = ScaledOptimizer(self.opt, self.compute_dtype)


def main():
    cfg = Config()
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
