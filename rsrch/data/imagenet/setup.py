import tyro
from pydantic import BaseModel


class Args(BaseModel):
    data_root: str


def main():
    tyro.cli(Args)

    print("""ImageNet cannot be set up automatically.
You can download the ImageNet-1k subset from Kaggle:
https://www.kaggle.com/competitions/imagenet-object-localization-challenge/overview""")  # noqa: T201


if __name__ == "__main__":
    main()
